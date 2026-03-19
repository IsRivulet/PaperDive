"""OCR-based PDF Reader using local PaddleOCR.

Replaces the external edusys OCR API with a local PaddleOCR model.
PyMuPDF renders each page to a numpy array, PaddleOCR extracts text,
results are sorted by reading order (top-to-bottom, left-to-right).

Note: PaddleOCR does not preserve LaTeX formula syntax — math expressions
will be extracted as approximate Unicode text. For formula-preserving OCR,
a specialized math OCR service is still required.
"""

import logging
import threading
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Union
from uuid import uuid4

import fitz  # PyMuPDF
import numpy as np
from paddleocr import PaddleOCR

from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document.base import Document
from agno.knowledge.reader.pdf_reader import BasePDFReader
from agno.knowledge.types import ContentType

logger = logging.getLogger(__name__)


def _sort_boxes_by_reading_order(
    ocr_result: list,
    line_threshold: float = 10.0,
) -> list[tuple[list, str, float]]:
    """
    将 PaddleOCR 返回的文本块按阅读顺序排序。

    PaddleOCR 返回格式: [[bbox, (text, confidence)], ...]
    bbox 为四个角点坐标: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    排序策略：
    1. 取 bbox 顶边 y 坐标，差值在 line_threshold 内视为同一行
    2. 同一行内按 x 坐标从左到右排序
    3. 行间按 y 坐标从上到下排序
    """
    if not ocr_result:
        return []

    items = []
    for line in ocr_result:
        bbox, (text, conf) = line
        # 取上边两点的平均 y 作为行基准
        top_y = (bbox[0][1] + bbox[1][1]) / 2
        left_x = (bbox[0][0] + bbox[3][0]) / 2
        items.append((top_y, left_x, bbox, text, conf))

    # 按 top_y 聚合成行组
    items.sort(key=lambda t: t[0])
    groups: list[list] = []
    current_group: list = []
    current_y: float = -1

    for item in items:
        top_y = item[0]
        if current_y < 0 or abs(top_y - current_y) <= line_threshold:
            current_group.append(item)
            # 更新当前行基准为组内平均
            current_y = sum(i[0] for i in current_group) / len(current_group)
        else:
            groups.append(current_group)
            current_group = [item]
            current_y = top_y

    if current_group:
        groups.append(current_group)

    # 行内按 left_x 排序
    result = []
    for group in groups:
        group.sort(key=lambda t: t[1])
        for _, _, bbox, text, conf in group:
            result.append((bbox, text, conf))

    return result


class OcrPDFReader(BasePDFReader):
    """PDF reader using local PaddleOCR for text extraction.

    After calling read(), per-page OCR text is available via
    ``last_ocr_pages`` for downstream structure extraction.

    Args:
        lang:            PaddleOCR 语言代码，默认 "en"（英文数学论文）
                         中英混合论文可用 "ch"
        dpi:             渲染分辨率，建议 200-300（数学论文用 250）
        use_gpu:         是否使用 GPU 加速（需安装 paddlepaddle-gpu）
        use_angle_cls:   是否启用方向分类（扫描版倾斜文档建议开启）
        max_workers:     页面渲染的并行线程数（OCR 本身串行执行）
        split_on_pages:  同原 BasePDFReader 参数
        chunking_strategy: 同原 BasePDFReader 参数
    """

    def __init__(
        self,
        lang: str = "en",
        dpi: int = 250,
        use_gpu: bool = False,
        use_angle_cls: bool = False,
        max_workers: int = 4,
        *,
        split_on_pages: bool = True,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        **kwargs,
    ):
        self.lang = lang
        self.dpi = dpi
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls
        self.max_workers = max_workers
        self.last_ocr_pages: list[str] = []

        # PaddleOCR 不是线程安全的，用锁保护单例
        self._ocr_lock = threading.Lock()
        self._ocr_engine: Optional[PaddleOCR] = None

        super().__init__(
            split_on_pages=split_on_pages,
            chunking_strategy=chunking_strategy,
            **kwargs,
        )

    def _get_ocr_engine(self) -> PaddleOCR:
        """懒初始化 PaddleOCR 引擎（避免 import 时就加载模型）。"""
        if self._ocr_engine is None:
            logger.info(
                "初始化 PaddleOCR（lang=%s, gpu=%s）...", self.lang, self.use_gpu
            )
            self._ocr_engine = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,       # 抑制 PaddleOCR 自身的大量日志
                enable_mkldnn=True,   # CPU 加速
            )
        return self._ocr_engine

    @classmethod
    def get_supported_content_types(cls) -> List[ContentType]:
        return [ContentType.PDF]

    def _render_page_to_array(self, page: fitz.Page) -> np.ndarray:
        """将 PDF 页面渲染为 RGB numpy 数组（PaddleOCR 的输入格式）。"""
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
        # tobytes("raw") 返回连续 RGB 字节，转为 HxWx3 数组
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        return img_array

    def _ocr_single_page(self, page_index: int, img_array: np.ndarray) -> tuple[int, str]:
        """OCR 单页，返回 (page_index, text)。

        使用锁确保多线程渲染场景下 OCR 引擎的串行调用。
        """
        with self._ocr_lock:
            engine = self._get_ocr_engine()
            try:
                result = engine.ocr(img_array, cls=self.use_angle_cls)
            except Exception as e:
                logger.warning("Page %d: PaddleOCR 失败: %s", page_index + 1, e)
                return page_index, ""

        # result 是 [页面结果列表]，单图时取 result[0]
        page_result = result[0] if result and result[0] else []

        if not page_result:
            logger.debug("Page %d: 无识别结果", page_index + 1)
            return page_index, ""

        sorted_lines = _sort_boxes_by_reading_order(page_result)
        text = "\n".join(text for _, text, _ in sorted_lines)
        return page_index, text

    def _create_documents_with_metadata(
        self,
        pdf_content: List[str],
        doc_name: str,
        page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Document]:
        """创建带结构元数据的 Document 列表，然后分块。"""
        documents: List[Document] = []
        for i, page_text in enumerate(pdf_content):
            page_num = i + 1
            meta: Dict[str, Any] = {"page": page_num}
            if page_metadata and i in page_metadata:
                meta.update(page_metadata[i])
            documents.append(
                Document(
                    name=doc_name,
                    id=str(uuid4()),
                    meta_data=meta,
                    content=page_text,
                )
            )
        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents

    def read(
        self,
        pdf: Optional[Union[str, Path, IO[Any]]] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,
        page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Document]:
        """读取 PDF 并通过 PaddleOCR 提取文本。

        Args:
            pdf:           PDF 文件路径或文件对象
            name:          文档名称（可选）
            password:      PDF 密码（可选）
            page_metadata: 页级附加元数据，格式 {页面索引(0-based): {key: value}}
        """
        if pdf is None:
            logger.error("未提供 PDF")
            return []

        doc_name = self._get_doc_name(pdf, name)
        logger.debug("Reading (PaddleOCR): %s", doc_name)

        doc = fitz.open(pdf)
        if password:
            doc.authenticate(password)
        total_pages = len(doc)

        logger.info(
            "PaddleOCR: %s — %d 页, dpi=%d, lang=%s",
            doc_name, total_pages, self.dpi, self.lang,
        )

        # 并行渲染所有页面为 numpy 数组（渲染是 CPU/GPU 无状态操作，可并行）
        # OCR 推理本身串行（受 _ocr_lock 保护）
        from concurrent.futures import ThreadPoolExecutor, as_completed

        page_arrays: list[tuple[int, np.ndarray]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            render_futures = {
                pool.submit(self._render_page_to_array, page): i
                for i, page in enumerate(doc)
            }
            for future in as_completed(render_futures):
                i = render_futures[future]
                try:
                    arr = future.result()
                    page_arrays.append((i, arr))
                except Exception as e:
                    logger.warning("Page %d 渲染失败: %s", i + 1, e)
                    page_arrays.append((i, np.zeros((100, 100, 3), dtype=np.uint8)))
        doc.close()

        page_arrays.sort(key=lambda t: t[0])

        # OCR 推理：串行处理（PaddleOCR 非线程安全）
        pdf_content: list[str] = [""] * total_pages
        for i, arr in page_arrays:
            idx, text = self._ocr_single_page(i, arr)
            pdf_content[idx] = text
            logger.info("OCR: 第 %d/%d 页完成（%d 字符）", idx + 1, total_pages, len(text))

        self.last_ocr_pages = list(pdf_content)

        return self._create_documents_with_metadata(
            pdf_content,
            doc_name,
            page_metadata=page_metadata,
        )