
"""
arXiv LaTeX source downloader and reader.

Downloads the raw .tar.gz source from arxiv.org/e-print/{id},
extracts and merges all .tex files, cleans up preamble/bibliography,
then splits by section boundaries into Document chunks.

This completely avoids OCR — formulas are preserved as original LaTeX.
Falls back gracefully when source is unavailable.
"""

import io
import logging
import re
import tarfile
from pathlib import Path
from typing import IO, Any, List, Optional, Union
from uuid import uuid4

import httpx
from agno.knowledge.chunking.strategy import ChunkingStrategy
from agno.knowledge.document.base import Document
from agno.knowledge.reader.pdf_reader import BasePDFReader
from agno.knowledge.types import ContentType

logger = logging.getLogger(__name__)

# arXiv ID 格式：YYMM.NNNNN 或 YYMM.NNNNNvN
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def is_arxiv_id(name: str) -> bool:
    """判断字符串是否符合 arXiv ID 格式。"""
    return bool(ARXIV_ID_RE.match(name.strip()))


# ─────────────────────────────────────────────────────────────
# LaTeX 文本清洗工具
# ─────────────────────────────────────────────────────────────

def _strip_comments(tex: str) -> str:
    """删除 LaTeX 注释（% 开头的行）。"""
    return re.sub(r"(?m)^%.*$", "", tex)


def _strip_preamble(tex: str) -> str:
    """删除 \\begin{document} 之前的导言区。"""
    match = re.search(r"\\begin\{document\}", tex)
    if match:
        return tex[match.end():]
    return tex  # 找不到则保留全文（部分子文件无导言区）


def _strip_bibliography(tex: str) -> str:
    """删除参考文献部分（对结构提取无价值，且占大量 token）。"""
    # \\bibliography{...} 命令
    tex = re.sub(r"\\bibliography\{[^}]*\}", "", tex)
    # thebibliography 环境
    tex = re.sub(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        "",
        tex,
        flags=re.DOTALL,
    )
    # bibitem 列表
    tex = re.sub(r"\\bibitem\{[^}]*\}[^\\]*", "", tex)
    return tex


def _strip_figure_content(tex: str) -> str:
    """
    删除 figure/table 环境内的 \\includegraphics 等纯图片命令。
    保留 caption 和 label（结构提取需要）。
    """
    def clean_env(m):
        content = m.group(0)
        content = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", content)
        return content

    tex = re.sub(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
        clean_env,
        tex,
        flags=re.DOTALL,
    )
    return tex


# ─────────────────────────────────────────────────────────────
# 多文件 .tex 合并
# ─────────────────────────────────────────────────────────────

def _find_main_file(members: dict[str, str]) -> str | None:
    """找到含 \\documentclass 的主文件。"""
    for name, content in members.items():
        if r"\documentclass" in content:
            return name
    return None


def _resolve_inputs(tex: str, members: dict[str, str], visited: set[str]) -> str:
    """
    递归展开 \\input{} 和 \\include{} 命令。
    防止循环引用（visited 集合追踪已处理文件）。
    """
    def replace(m):
        fname = m.group(1).strip()
        if not fname.endswith(".tex"):
            fname += ".tex"

        # 尝试多种路径匹配（相对路径、仅文件名）
        content = (
            members.get(fname)
            or members.get(Path(fname).name)
            or members.get(fname.lstrip("./"))
        )
        if fname in visited or not content:
            return ""

        visited.add(fname)
        return _resolve_inputs(content, members, visited)

    return re.sub(r"\\(?:input|include)\{([^}]+)\}", replace, tex)


def _merge_tex_files(members: dict[str, str]) -> str:
    """将多个 .tex 文件合并为单个完整文档。"""
    main = _find_main_file(members)

    if main is None:
        # 找不到主文件：按字典序拼接所有文件
        logger.warning("找不到含 \\documentclass 的主文件，直接拼接所有 .tex 文件")
        return "\n\n".join(members.values())

    visited = {main}
    return _resolve_inputs(members[main], members, visited)


# ─────────────────────────────────────────────────────────────
# 下载与解析
# ─────────────────────────────────────────────────────────────

def download_and_parse_arxiv_source(arxiv_id: str, cache_dir: Path) -> str | None:
    """
    下载 arXiv 论文 LaTeX 源码，返回清洗后的完整 LaTeX 文本。

    流程：
    1. 命中本地缓存 → 直接读取
    2. 下载 https://arxiv.org/e-print/{arxiv_id}
    3. 解压 tar.gz，合并所有 .tex 文件
    4. 清洗（去注释、去导言区、去参考文献）

    Returns:
        清洗后的 LaTeX 字符串；以下情况返回 None：
        - 论文无源码（仅 PDF 投稿）
        - 网络错误
        - 解压/解析失败
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{arxiv_id}.tex"

    # 缓存命中
    if cache_file.exists() and cache_file.stat().st_size > 200:
        logger.info("[LaTeX源码] 缓存命中: %s", arxiv_id)
        return cache_file.read_text(encoding="utf-8", errors="replace")

    url = f"https://arxiv.org/e-print/{arxiv_id}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        print(f"[LaTeX源码] 正在下载 {arxiv_id} ...", flush=True)
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)

        if resp.status_code != 200:
            logger.warning(
                "[LaTeX源码] 下载失败 (HTTP %d): %s", resp.status_code, arxiv_id
            )
            return None

        raw = resp.content
        full_tex: str | None = None

        # tar.gz 格式（绝大多数论文）
        if raw[:2] == b"\x1f\x8b":
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
                members: dict[str, str] = {}
                for member in tar.getmembers():
                    if not member.name.endswith(".tex"):
                        continue
                    f = tar.extractfile(member)
                    if f:
                        text = f.read().decode("utf-8", errors="replace")
                        members[member.name] = _strip_comments(text)

            if not members:
                logger.warning("[LaTeX源码] 压缩包内无 .tex 文件: %s", arxiv_id)
                return None

            print(
                f"[LaTeX源码] 解压完成，找到 {len(members)} 个 .tex 文件",
                flush=True,
            )
            full_tex = _merge_tex_files(members)

        else:
            # 单个 .tex 文件（少数旧论文）
            try:
                full_tex = _strip_comments(raw.decode("utf-8", errors="replace"))
            except Exception as e:
                logger.warning("[LaTeX源码] 单文件解码失败: %s", e)
                return None

        if not full_tex or len(full_tex.strip()) < 200:
            logger.warning("[LaTeX源码] 提取内容过短，可能是纯 PDF 投稿: %s", arxiv_id)
            return None

        # 清洗
        full_tex = _strip_preamble(full_tex)
        full_tex = _strip_bibliography(full_tex)
        full_tex = _strip_figure_content(full_tex)
        full_tex = re.sub(r"\n{3,}", "\n\n", full_tex)  # 压缩多余空行
        full_tex = full_tex.strip()

        # 写缓存
        cache_file.write_text(full_tex, encoding="utf-8")
        print(
            f"[LaTeX源码] 下载完成: {arxiv_id} ({len(full_tex):,} 字符)",
            flush=True,
        )
        return full_tex

    except Exception as e:
        logger.warning("[LaTeX源码] 下载异常 %s: %s", arxiv_id, e)
        return None


# ─────────────────────────────────────────────────────────────
# 按章节切分
# ─────────────────────────────────────────────────────────────

# 匹配各级章节标题命令（正向前瞻，保留命令本身）
_SECTION_BOUNDARY_RE = re.compile(
    r"(?=\\(?:chapter|section|subsection|subsubsection|paragraph)\*?\{[^}]+\})"
)


def split_latex_by_section(
    latex: str,
    min_chars: int = 400,
    max_chars: int = 3500,
) -> list[str]:
    """
    按章节边界切分 LaTeX 文本，确保每块大小适合向量化。

    策略：
    1. 在 \\section / \\subsection 等命令处切分
    2. 过短的块合并到上一块（避免碎片化）
    3. 过长的块按段落二次切分（避免超出 embedding 上下文窗口）

    Returns:
        每块包含一个或多个章节的 LaTeX 文本列表。
    """
    raw_parts = [p.strip() for p in _SECTION_BOUNDARY_RE.split(latex) if p.strip()]

    if not raw_parts:
        # 无章节标记（如纯摘要文本）：直接按字符数切分
        return [latex[i : i + max_chars] for i in range(0, len(latex), max_chars)]

    # 合并过短的块
    chunks: list[str] = []
    current = ""

    for part in raw_parts:
        if not current:
            current = part
        elif len(current) + len(part) <= max_chars:
            current = current + "\n\n" + part
        else:
            chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    # 二次切分过长的块（按段落边界）
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
            continue

        paragraphs = chunk.split("\n\n")
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= max_chars or not buf:
                buf = (buf + "\n\n" + para).strip()
            else:
                result.append(buf)
                buf = para
        if buf:
            result.append(buf)

    # 合并尾部过短的块
    final: list[str] = []
    for chunk in result:
        if final and len(chunk) < min_chars:
            final[-1] = final[-1] + "\n\n" + chunk
        else:
            final.append(chunk)

    return final


# ─────────────────────────────────────────────────────────────
# agno Reader 封装
# ─────────────────────────────────────────────────────────────

class LaTeXSourceReader(BasePDFReader):
    """
    读取预下载的 .tex 文件，按章节分块返回 Document 列表。

    接口与 OcrPDFReader 完全一致：
    - read(path) → List[Document]
    - last_ocr_pages: list[str]  （各章节文本，供 structure_extractor 使用）

    用法：
        reader = LaTeXSourceReader(chunking_strategy=semantic_chunking)
        shared_knowledge.insert(name=arxiv_id, path=str(tex_path), reader=reader)
        pages = reader.last_ocr_pages  # 章节文本列表
    """

    def __init__(
        self,
        min_section_chars: int = 400,
        max_section_chars: int = 3500,
        *,
        split_on_pages: bool = True,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        **kwargs,
    ):
        self.min_section_chars = min_section_chars
        self.max_section_chars = max_section_chars
        # 与 OcrPDFReader 保持一致的接口，供 PaperDive 的 if reader.last_ocr_pages: 使用
        self.last_ocr_pages: list[str] = []
        super().__init__(
            split_on_pages=split_on_pages,
            chunking_strategy=chunking_strategy,
            **kwargs,
        )

    @classmethod
    def get_supported_content_types(cls) -> List[ContentType]:
        return [ContentType.PDF, ContentType.TEXT]

    def read(
        self,
        pdf: Optional[Union[str, Path, IO[Any]]] = None,
        name: Optional[str] = None,
        password: Optional[str] = None,          # 保持接口一致，实际忽略
        page_metadata: Optional[dict] = None,    # 保持接口一致，实际忽略
        **kwargs,
    ) -> List[Document]:
        if pdf is None:
            logger.error("LaTeXSourceReader: 未提供文件路径")
            return []

        path = Path(str(pdf))
        doc_name = name or path.stem

        try:
            latex = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("LaTeXSourceReader: 读取文件失败 %s: %s", path, e)
            return []

        sections = split_latex_by_section(
            latex,
            min_chars=self.min_section_chars,
            max_chars=self.max_section_chars,
        )

        if not sections:
            logger.warning("LaTeXSourceReader: 未提取到任何章节: %s", doc_name)
            return []

        # 暴露给 PaperDive 的结构提取流程
        self.last_ocr_pages = list(sections)

        documents = [
            Document(
                name=doc_name,
                id=str(uuid4()),
                meta_data={
                    "page": i + 1,          # 用"章节编号"模拟页码
                    "source": "latex",      # 标记来源，便于后续过滤
                },
                content=section_text,
            )
            for i, section_text in enumerate(sections)
        ]

        logger.info(
            "LaTeXSourceReader: %s → %d 个章节块", doc_name, len(documents)
        )

        if self.chunk:
            return self._build_chunked_documents(documents)
        return documents