import os
import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv
import httpx
import json
import sqlite3
import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR

from agno.agent import Agent
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.knowledge.chunking.semantic import SemanticChunking
from agno.knowledge.content import ContentStatus
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader  # kept as fallback
from ocr_pdf_reader import OcrPDFReader
from arxiv_source_reader import (
    LaTeXSourceReader,
    download_and_parse_arxiv_source,
    is_arxiv_id,
)
from structure_extractor import (
    extract_paper_structure,
    extract_paper_summary,
    find_section_for_page,
    find_elements_on_page,
    format_structure_for_display,
)
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.vectordb.lancedb import LanceDb, SearchType

load_dotenv()
# 抑制 numpy 除以零警告
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
# 将 agno 的日志级别设为 ERROR，屏蔽 WARNING 及以下
logging.getLogger("agno").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).parent
PAPERS_DIR = BASE_DIR / "arxiv_test" / "papers"
# LaTeX 源码缓存目录（与 papers/ 同级）
TEX_CACHE_DIR = BASE_DIR / "arxiv_test" / "tex_sources"
TEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

NOTES_DIR = BASE_DIR / "notes"
SQLITE_DB_FILE = str(BASE_DIR / "arxiv_test" / "state.db")
LANCEDB_URI = str(BASE_DIR / "arxiv_test" / "lancedb")
TEAM_SESSION_ID = "arxiv-team-v4"

PAPERS_DIR.mkdir(parents=True, exist_ok=True)

agent_db = SqliteDb(
    db_file=SQLITE_DB_FILE, session_table="agent_sessions", memory_table="agent_memory"
)


knowledge_db = SqliteDb(
    db_file=SQLITE_DB_FILE,
    knowledge_table="knowledge_contents",
    session_table="knowledge_sessions",
)


vector_db = LanceDb(
    uri=LANCEDB_URI,
    table_name="arxiv_paper_chunks",
    search_type=SearchType.vector,
    embedder=OllamaEmbedder(
        id="bge-m3",
        dimensions=1024,
        timeout=120.0,
    ),
)


# 把文章按语义自然分割成多个块，每个块的字符数尽量接近 1200
semantic_chunking = SemanticChunking(
    embedder=OllamaEmbedder(
        id="bge-m3",
        dimensions=1024,
        timeout=120.0,
    ),
    chunk_size=1200,
    similarity_threshold=0.6,
)

pdf_reader = OcrPDFReader(
    lang="en",          # 纯英文论文
    # lang="ch",        # 中英混合论文
    dpi=250,            # 数学论文建议 250
    use_gpu=False,      # 有 NVIDIA GPU 时改为 True
    use_angle_cls=False,# 扫描版倾斜文档才需要开启
    max_workers=4,
    chunking_strategy=semantic_chunking,
    split_on_pages=True,
)

latex_reader = LaTeXSourceReader(
    min_section_chars=400,
    max_section_chars=3500,
    chunking_strategy=semantic_chunking,  # 复用同一个分块策略
    split_on_pages=True,
)

#shared_llm = OpenAILike(
#    id=os.getenv("MODEL_ID", "coding-glm-4.7-free"),
#    api_key=os.getenv("OPENAI_API_KEY"),
#    base_url=os.getenv("BASE_URL", "https://aihubmix.com/v1"),
#    timeout=300.0,
#)

shared_llm = OpenAILike(
    id=os.getenv("LLM_MODEL_ID", "ecnu-plus"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://chat.ecnu.edu.cn/open/api/v1
"),
    timeout=300.0,
)

shared_knowledge = Knowledge(
    name="arxiv_library",
    vector_db=vector_db,
    contents_db=knowledge_db,
    readers={"pdf": pdf_reader},
    max_results=8,
)


def _get_indexed_names() -> set:
    """查询 knowledge_db 中已成功完成索引的论文名称集合。"""
    try:
        contents, _ = shared_knowledge.get_content()
        return {
            c.name for c in contents if c.status == ContentStatus.COMPLETED and c.name
        }
    except Exception as e:
        print(f"[Debug] _get_indexed_names 出错: {e}")
        return set()


def _cleanup_stuck_processing():
    """清理卡在 processing 状态的记录，避免重启后无法重新索引。"""
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.execute(
            "DELETE FROM knowledge_contents WHERE status = 'processing'"
        )
        n = cursor.rowcount
        conn.commit()
        conn.close()
        if n > 0:
            print(f"[清理] 已移除 {n} 条卡住的索引记录，将重新索引。", flush=True)
    except Exception:
        pass


_POLLUTION_PATTERNS = ["<function=", "</tool_call>", "<parameter="]


def _cleanup_polluted_session():
    """检测并清除 session 历史中被污染的工具调用记录。

    当 LLM 尝试调用未注册的工具时会输出原始文本（如 <function=...>），
    这些错误模式被持久化到 session 后会导致后续对话持续模仿。
    """
    try:
        conn = sqlite3.connect(SQLITE_DB_FILE)
        row = conn.execute(
            "SELECT runs FROM agent_sessions WHERE session_id = ?",
            (TEAM_SESSION_ID,),
        ).fetchone()
        if not row or not row[0]:
            conn.close()
            return
        runs_text = row[0]
        if any(p in runs_text for p in _POLLUTION_PATTERNS):
            conn.execute(
                "DELETE FROM agent_sessions WHERE session_id = ?",
                (TEAM_SESSION_ID,),
            )
            conn.commit()
            print(
                f"[清理] 检测到对话历史包含异常工具调用模式，已自动清除以确保工具正常执行。",
                flush=True,
            )
        conn.close()
    except Exception as e:
        print(f"[清理] session 检测失败（不影响使用）: {e}", flush=True)


# ─────────────────────────────────────────────────────────────
# 论文结构存储（SQLite）
# ─────────────────────────────────────────────────────────────



def _init_tables():
    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_structures ("
        "  paper_id TEXT PRIMARY KEY,"
        "  structure_json TEXT,"
        "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_pages ("
        "  paper_id TEXT,"
        "  page_num INTEGER,"
        "  content TEXT,"
        "  PRIMARY KEY (paper_id, page_num)"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_summaries ("
        "  paper_id TEXT PRIMARY KEY,"
        "  title TEXT,"
        "  abstract TEXT,"
        "  proof_approaches TEXT,"
        "  core_techniques TEXT,"
        "  field_tags TEXT,"
        "  content_tags TEXT,"
        "  technique_tags TEXT,"
        "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS paper_structural_elements ("
        "  id TEXT PRIMARY KEY,"          # paper_id + element_id，如 "2301.12345::thm3.1"
        "  paper_id TEXT NOT NULL,"
        "  element_type TEXT NOT NULL,"   # theorem / proof / definition / equation
        "  element_id TEXT NOT NULL,"     # 如 thm3.1、def2.1
        "  label TEXT,"                   # 显示名，如 "Theorem 3.1"
        "  content TEXT,"
        "  start_page INTEGER,"
        "  end_page INTEGER,"
        "  ref_ids TEXT"                  # JSON数组，关联的其他element_id（如定理→其证明）
        ")"
    )
    conn.execute(
    "CREATE TABLE IF NOT EXISTS paper_notations ("
    "  id TEXT PRIMARY KEY,"          # paper_id + "::" + latex（做唯一键）
    "  paper_id TEXT NOT NULL,"
    "  latex TEXT NOT NULL,"          # 原始 LaTeX，如 \\mathcal{F}
    "  ascii_repr TEXT,"              # ASCII 近似，如 F，方便模糊搜索
    "  meaning TEXT NOT NULL,"        # 含义说明
    "  first_page INTEGER,"           # 首次定义页码
    "  context TEXT"                  # 定义所在的原文片段（≤200字符）
    ")"
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pse_paper_type "
        "ON paper_structural_elements(paper_id, element_type)"
    )
    # 给 paper_structural_elements 补加 depends_on 列（幂等）
    try:
        conn.execute(
            "ALTER TABLE paper_structural_elements ADD COLUMN depends_on TEXT"
        )
    except sqlite3.OperationalError:
        pass  # 列已存在，忽略
    conn.commit()
    conn.close()

_init_tables()


def save_paper_structure(paper_id: str, structure: dict) -> None:
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.execute(
        "INSERT OR REPLACE INTO paper_structures (paper_id, structure_json) VALUES (?, ?)",
        (paper_id, json.dumps(structure, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def load_paper_structure(paper_id: str) -> dict | None:
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    row = conn.execute(
        "SELECT structure_json FROM paper_structures WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


# ── paper_pages CRUD ──────────────────────────────────────────

def save_paper_pages(paper_id: str, pages: list[str]) -> None:
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    for i, content in enumerate(pages):
        conn.execute(
            "INSERT OR REPLACE INTO paper_pages (paper_id, page_num, content) VALUES (?, ?, ?)",
            (paper_id, i + 1, content),
        )
    conn.commit()
    conn.close()


def load_paper_pages(paper_id: str, start: int = 1, end: int = 0) -> list[tuple[int, str]]:
    """Return list of (page_num, content). end=0 means start page only."""
    paper_id = str(paper_id)
    if end <= 0:
        end = start
    conn = sqlite3.connect(SQLITE_DB_FILE)
    rows = conn.execute(
        "SELECT page_num, content FROM paper_pages "
        "WHERE paper_id = ? AND page_num >= ? AND page_num <= ? ORDER BY page_num",
        (paper_id, start, end),
    ).fetchall()
    conn.close()
    return rows


def get_paper_page_count(paper_id: str) -> int:
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    row = conn.execute(
        "SELECT MAX(page_num) FROM paper_pages WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else 0


# ── paper_summaries CRUD ──────────────────────────────────────

def save_paper_summary(paper_id: str, summary: dict) -> None:
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.execute(
        "INSERT OR REPLACE INTO paper_summaries "
        "(paper_id, title, abstract, proof_approaches, core_techniques, "
        " field_tags, content_tags, technique_tags) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            paper_id,
            summary.get("title", ""),
            summary.get("abstract", ""),
            json.dumps(summary.get("proof_approaches", {}), ensure_ascii=False),
            json.dumps(summary.get("core_techniques", []), ensure_ascii=False),
            json.dumps(summary.get("field_tags", []), ensure_ascii=False),
            json.dumps(summary.get("content_tags", []), ensure_ascii=False),
            json.dumps(summary.get("technique_tags", []), ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def load_paper_summary(paper_id: str) -> dict | None:
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    row = conn.execute(
        "SELECT title, abstract, proof_approaches, core_techniques, "
        "field_tags, content_tags, technique_tags FROM paper_summaries WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "title": row[0] or "",
        "abstract": row[1] or "",
        "proof_approaches": json.loads(row[2]) if row[2] else {},
        "core_techniques": json.loads(row[3]) if row[3] else [],
        "field_tags": json.loads(row[4]) if row[4] else [],
        "content_tags": json.loads(row[5]) if row[5] else [],
        "technique_tags": json.loads(row[6]) if row[6] else [],
    }


def load_all_paper_summaries() -> list[dict]:
    conn = sqlite3.connect(SQLITE_DB_FILE)
    rows = conn.execute(
        "SELECT paper_id, title, field_tags, content_tags, technique_tags FROM paper_summaries"
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({
            "paper_id": r[0],
            "title": r[1] or "",
            "field_tags": json.loads(r[2]) if r[2] else [],
            "content_tags": json.loads(r[3]) if r[3] else [],
            "technique_tags": json.loads(r[4]) if r[4] else [],
        })
    return results

def _count_vector_chunks(paper_id: str) -> int:
    """统计指定论文在向量库中的块数，失败返回 -1"""
    paper_id = str(paper_id)
    try:
        import lancedb
        db = lancedb.connect(LANCEDB_URI)
        table = db.open_table("arxiv_paper_chunks")
        # 假设表中存在 'name' 列存储论文ID
        result = table.search().where(f"name = '{paper_id}'").limit(10000).to_pandas()
        return len(result)
    except Exception:
        return -1

def diagnose_paper(paper_id: str) -> str:
    """
    诊断某篇论文在本地库中的结构/摘要/原文状态（用于排查「结构数据暂时无法获取」等问题）。
    可在项目根目录运行: python -c "from PaperDive import diagnose_paper; print(diagnose_paper('sat-matching'))"
    """
    paper_id = str(paper_id)
    lines = [f"论文 ID: {paper_id}", "─" * 40]
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    lines.append(f"PDF 文件: {'存在' if pdf_path.exists() else '不存在'}")
    # 知识库向量
    try:
        count = _count_vector_chunks(paper_id)
        if count >= 0:
            lines.append(f"向量块数: {count}")
        else:
            lines.append("向量: 查询失败（可能是表不存在或无数据）")
    except Exception as e:
        lines.append(f"向量: 查询异常 — {e}")
    # 结构
    st = load_paper_structure(paper_id)
    if st is None:
        lines.append("结构: 无")
    else:
        n_sec = len(st.get("sections", []))
        n_thm = len(st.get("theorems", []))
        n_def = len(st.get("definitions", []))
        lines.append(f"结构: 有 — {n_sec} 章节, {n_thm} 定理/引理, {n_def} 定义")
        raw = json.dumps(st, ensure_ascii=False)[:300]
        lines.append(f"结构预览: {raw}...")
    # 摘要
    sm = load_paper_summary(paper_id)
    if sm is None:
        lines.append("摘要: 无")
    else:
        title = (sm.get("title") or "")[:60]
        lines.append(f"摘要: 有 — title: {title}...")
    # 原文页数
    n_pages = get_paper_page_count(paper_id)
    lines.append(f"原文页数: {n_pages}")
    lines.append("─" * 40)
    return "\n".join(lines)


def _extract_and_store_structure(paper_id: str, pages: list[str]) -> dict:
    """Run two-phase structure extraction, store result, return structure.
    On any failure (LLM/timeout/JSON), falls back to regex-only and still saves.
    """
    paper_id = str(paper_id)
    print(f"[结构] 正在提取 {paper_id} 的论文结构...", flush=True)
    structure = None
    try:
        structure = extract_paper_structure(pages, llm=shared_llm)
    except Exception as e:
        print(f"[结构] 完整提取异常，改用仅正则结果: {e}", flush=True)
        try:
            structure = extract_paper_structure(pages, llm=None)
        except Exception as e2:
            print(f"[结构] 正则提取也失败: {e2}", flush=True)
            structure = {"sections": [], "theorems": [], "proofs": [], "definitions": [], "key_equations": []}
    if structure is None:
        structure = {"sections": [], "theorems": [], "proofs": [], "definitions": [], "key_equations": []}
    try:
        save_paper_structure(paper_id, structure)
    except Exception as se:
        print(f"[结构] 写入数据库失败: {se}", flush=True)
        raise
    _save_structural_elements(paper_id, structure)
    n_thm = len(structure.get("theorems", []))
    n_def = len(structure.get("definitions", []))
    n_sec = len(structure.get("sections", []))
    print(
        f"[结构] 完成: {n_sec} 个章节, {n_thm} 个定理/引理, {n_def} 个定义",
        flush=True,
    )
    return structure

def _save_structural_elements(paper_id: str, structure: dict) -> None:
    """将结构中的各类元素拆开存入 paper_structural_elements 表。"""
    paper_id = str(paper_id)
    rows: list[tuple] = []

    # 建立 proof → theorem 的反向索引（用于填充 ref_ids）
    proof_refs: dict[str, list[str]] = {}
    for proof in structure.get("proofs", []):
        ref = proof.get("proof_of", "")
        if ref:
            proof_refs.setdefault(ref, []).append(proof.get("id", ""))

    for thm in structure.get("theorems", []):
        eid = thm.get("id", "")
        related_proofs = proof_refs.get(eid, [])
        rows.append((
            f"{paper_id}::{eid}",
            paper_id,
            thm.get("type", "theorem"),   # theorem / lemma / corollary 等
            eid,
            thm.get("label", eid),
            thm.get("statement", ""),
            thm.get("page"),
            thm.get("page"),
            json.dumps(related_proofs, ensure_ascii=False),
        ))

    for proof in structure.get("proofs", []):
        eid = proof.get("id", "")
        rows.append((
            f"{paper_id}::{eid}",
            paper_id,
            "proof",
            eid,
            proof.get("label", eid),
            proof.get("content", ""),
            proof.get("page"),
            proof.get("end_page") or proof.get("page"),
            json.dumps([proof.get("proof_of", "")], ensure_ascii=False),
        ))

    for defn in structure.get("definitions", []):
        eid = defn.get("id", "")
        rows.append((
            f"{paper_id}::{eid}",
            paper_id,
            "definition",
            eid,
            defn.get("label", eid),
            defn.get("content", ""),
            defn.get("page"),
            defn.get("page"),
            "[]",
        ))

    for eq in structure.get("key_equations", []):
        eid = eq.get("id", "")
        rows.append((
            f"{paper_id}::{eid}",
            paper_id,
            "equation",
            eid,
            eq.get("label", eid),
            eq.get("content", ""),
            eq.get("page"),
            eq.get("page"),
            "[]",
        ))

    if not rows:
        return

    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.executemany(
        "INSERT OR REPLACE INTO paper_structural_elements "
        "(id, paper_id, element_type, element_id, label, content, start_page, end_page, ref_ids) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    
    conn.commit()
    conn.close()
    print(f"[结构单元] 已写入 {len(rows)} 条", flush=True)

def _save_dependencies(paper_id: str, dep_graph: dict[str, list[str]]) -> None:
    """
    Write depends_on lists back into paper_structural_elements.
    dep_graph: { proof_id: [dep_element_id, ...] }
    """
    if not dep_graph:
        return
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    for proof_id, deps in dep_graph.items():
        uid = f"{paper_id}::{proof_id}"
        conn.execute(
            "UPDATE paper_structural_elements SET depends_on = ? WHERE id = ?",
            (json.dumps(deps, ensure_ascii=False), uid),
        )
    conn.commit()
    conn.close()
    print(f"[依赖图] 已写入 {len(dep_graph)} 条 depends_on", flush=True)
    
def load_dependency_chain(
    paper_id: str,
    root_element_id: str,
    max_depth: int = 6,
) -> list[dict]:
    """
    BFS from root_element_id, following depends_on edges.
    Returns a list of element dicts in traversal order (root first).
    Cycles are handled via visited set.
    """
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)

    def fetch(eid: str) -> dict | None:
        uid = f"{paper_id}::{eid}"
        row = conn.execute(
            "SELECT element_type, element_id, label, content, "
            "start_page, ref_ids, depends_on "
            "FROM paper_structural_elements WHERE id = ?",
            (uid,),
        ).fetchone()
        if not row:
            return None
        return {
            "element_type": row[0],
            "element_id":   row[1],
            "label":        row[2],
            "content":      row[3],
            "start_page":   row[4],
            "ref_ids":      json.loads(row[5]) if row[5] else [],
            "depends_on":   json.loads(row[6]) if row[6] else [],
        }

    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(root_element_id, 0)]
    chain: list[dict] = []

    while queue:
        eid, depth = queue.pop(0)
        if eid in visited or depth > max_depth:
            continue
        visited.add(eid)

        node = fetch(eid)
        if not node:
            continue

        node["depth"] = depth
        chain.append(node)

        # 向下展开：如果是定理，先找其证明（通过 ref_ids）
        if node["element_type"] in ("theorem", "lemma", "corollary", "proposition"):
            for proof_id in node["ref_ids"]:
                if proof_id and proof_id not in visited:
                    queue.append((proof_id, depth + 1))

        # 向下展开：如果是证明，展开其 depends_on
        if node["element_type"] == "proof":
            for dep_id in node["depends_on"]:
                if dep_id and dep_id not in visited:
                    queue.append((dep_id, depth + 1))

    conn.close()
    return chain

def _extract_and_store_summary(paper_id: str, pages: list[str], structure: dict) -> dict | None:
    """Run LLM summary extraction, store result, return summary dict."""
    print(f"[摘要] 正在为 {paper_id} 生成高层摘要...", flush=True)
    paper_id = str(paper_id)
    summary = extract_paper_summary(pages, structure, llm=shared_llm)
    if summary:
        save_paper_summary(paper_id, summary)
        tags = (
            summary.get("field_tags", [])
            + summary.get("content_tags", [])
            + summary.get("technique_tags", [])
        )
        print(f"[摘要] 完成: 标签={', '.join(tags[:6])}", flush=True)
    else:
        print(f"[摘要] LLM 摘要提取失败，跳过", flush=True)
    return summary

def _run_post_index_pipeline(
    paper_id: str,
    active_reader,
    label: str = "",
) -> tuple[str, str]:
    """
    索引完成后的通用后处理：存原文、提取结构、提取摘要。

    Args:
        paper_id:      论文 ID
        active_reader: 刚完成索引的 reader（OcrPDFReader 或 LaTeXSourceReader）
        label:         日志前缀，如 "[LaTeX]" 或 "[OCR]"

    Returns:
        (structure_info, summary_info): 格式化的结果说明字符串，供拼接到返回消息。
    """
    structure_info = ""
    summary_info = ""

    if not active_reader.last_ocr_pages:
        return structure_info, summary_info

    try:
        save_paper_pages(paper_id, active_reader.last_ocr_pages)
        n = len(active_reader.last_ocr_pages)
        print(f"{label} 原文已存储（{n} 块）", flush=True)
    except Exception as e:
        print(f"{label} 原文存储失败: {e}", flush=True)

    structure = None
    try:
        structure = _extract_and_store_structure(paper_id, active_reader.last_ocr_pages)
        n_sec = len(structure.get("sections", []))
        n_thm = len(structure.get("theorems", []))
        n_def = len(structure.get("definitions", []))
        structure_info = f"\n结构提取：{n_sec} 个章节, {n_thm} 个定理/引理, {n_def} 个定义"
    except Exception as e:
        print(f"{label} 结构提取失败（不影响检索）: {e}", flush=True)
        
    # ── 新增：依赖图提取 ──────────────────────────────────
    try:
        from structure_extractor import extract_dependency_graph
        dep_graph = extract_dependency_graph(structure, llm=shared_llm)
        _save_dependencies(paper_id, dep_graph)
        n_edges = sum(len(v) for v in dep_graph.values())
        structure_info += f"\n依赖图：{n_edges} 条依赖边"
    except Exception as e:
        print(f"{label} 依赖图提取失败（不影响检索）: {e}", flush=True)
        
    try:
        sm = _extract_and_store_summary(paper_id, active_reader.last_ocr_pages, structure or {})
        if sm:
            tags = sm.get("field_tags", []) + sm.get("technique_tags", [])
            summary_info = f"\n摘要标签：{', '.join(tags[:4])}"
    except Exception as e:
        print(f"{label} 摘要提取失败（不影响检索）: {e}", flush=True)
        
    # —— 符号表提取 ——————————————————————————
    try:
        from structure_extractor import extract_notation_map
        symbols = extract_notation_map(active_reader.last_ocr_pages, llm=shared_llm)
        save_notation_map(paper_id, symbols)
        if symbols:
            notation_info = f"\n符号表：提取到 {len(symbols)} 个符号定义"
    except Exception as e:
        print(f"{label} 符号表提取失败（不影响检索）: {e}", flush=True)


    return structure_info, summary_info


# ─────────────────────────────────────────────────────────────
# 工具定义
# ─────────────────────────────────────────────────────────────

def save_notation_map(paper_id: str, symbols: list[dict]) -> None:
    paper_id = str(paper_id)
    if not symbols:
        return
    rows = []
    for s in symbols:
        latex = s.get("latex", "").strip()
        if not latex:
            continue
        uid = f"{paper_id}::{latex}"
        rows.append((
            uid,
            paper_id,
            latex,
            s.get("ascii_repr", ""),
            s.get("meaning", ""),
            s.get("first_page"),
            (s.get("context", "") or "")[:200],
        ))
    if not rows:
        return
    conn = sqlite3.connect(SQLITE_DB_FILE)
    conn.executemany(
        "INSERT OR REPLACE INTO paper_notations "
        "(id, paper_id, latex, ascii_repr, meaning, first_page, context) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"[符号表] 已写入 {len(rows)} 条", flush=True)


def load_notation_map(paper_id: str) -> list[dict]:
    paper_id = str(paper_id)
    conn = sqlite3.connect(SQLITE_DB_FILE)
    rows = conn.execute(
        "SELECT latex, ascii_repr, meaning, first_page, context "
        "FROM paper_notations WHERE paper_id = ? ORDER BY first_page",
        (paper_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "latex": r[0], "ascii_repr": r[1],
            "meaning": r[2], "first_page": r[3], "context": r[4],
        }
        for r in rows
    ]

def _perform_scan() -> str:
    """
    扫描本地论文文件夹，将尚未索引的新 PDF 写入向量知识库。
    已索引过的论文自动跳过，不重复处理，不删除历史数据。

    Returns:
        有新论文：索引结果列表。
        无新论文：现有论文列表 + 引导选项。
        文件夹为空：下载操作指引。
    """
    _cleanup_stuck_processing()
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    # 情况一：文件夹为空
    if not pdf_files:
        return (
            f"论文文件夹目前为空。\n\n"
            f"文件夹路径：{PAPERS_DIR}\n\n"
            "请按以下步骤添加论文：\n"
            "  1. 访问 https://arxiv.org 搜索感兴趣的论文\n"
            "  2. 点击论文页面右侧的 [Download PDF] 下载\n"
            "  3. 将 PDF 文件移入上述文件夹（或告诉我 arXiv ID，我自动下载）\n\n"
            "也可告诉我感兴趣的研究方向，我来帮你找值得读的论文！"
        )

    # 情况二：差集识别新论文
    indexed_names = _get_indexed_names()
    new_files = [f for f in pdf_files if f.stem not in indexed_names]

    # 情况三：无新论文，全部已索引
    if not new_files:
        lines = [
            "扫描完成，没有发现新论文。\n",
            f"知识库中已有以下 {len(indexed_names)} 篇论文：\n",
        ]
        for i, name in enumerate(sorted(indexed_names), 1):
            lines.append(f"  [{i}] {name}")
        lines += [
            "\n您可以：",
            "  • 直接提问，我将在所有论文中检索作答",
            "  • 说「研究第N篇」聚焦单篇深度问答",
            "  • 说「帮我找 XXX 方向的新论文」在 arXiv 搜索",
            "  • 说「加载 arXiv ID」自动下载并索引新论文",
        ]
        return "\n".join(lines)

    # 情况四：有新论文，执行追加索引
    success, failed = [], []
    total = len(new_files)
    print(f"发现 {total} 篇新论文，开始索引（大论文可能需数分钟）...\n", flush=True)

    for i, pdf_path in enumerate(new_files, 1):
        paper_id = pdf_path.stem
        print(f"[{i}/{total}] 正在索引: {paper_id} ...", flush=True)
        try:
            # ── 优先尝试 LaTeX 源码（仅当文件名符合 arXiv ID 格式）──
            active_reader = pdf_reader  # 默认
            used_latex = False

            if is_arxiv_id(paper_id):
                latex_source = download_and_parse_arxiv_source(paper_id, TEX_CACHE_DIR)
                if latex_source:
                    tex_path = TEX_CACHE_DIR / f"{paper_id}.tex"
                    shared_knowledge.insert(
                        name=paper_id,
                        path=str(tex_path),
                        reader=latex_reader,
                        skip_if_exists=True,
                    )
                    active_reader = latex_reader
                    used_latex = True
                    print(f"[{i}/{total}] ✅ LaTeX 源码索引完成: {paper_id}", flush=True)

            if not used_latex:
                # 回退：PDF + OCR
                shared_knowledge.insert(
                    name=paper_id,
                    path=str(pdf_path),
                    reader=pdf_reader,
                    skip_if_exists=True,
                )
                print(f"[{i}/{total}] OCR 索引完成: {paper_id}", flush=True)

            success.append(paper_id)

            # 后处理：存原文、提取结构、提取摘要
            label = "[LaTeX]" if used_latex else "[OCR]"
            _run_post_index_pipeline(paper_id, active_reader, label=label)

        except Exception as e:
            failed.append(f"{pdf_path.name}：{e}")
            print(f"[{i}/{total}] 失败: {paper_id} — {e}", flush=True)


    lines = [f" 已成功索引 {len(success)} 篇新论文：\n"]
    for i, pid in enumerate(success, 1):
        lines.append(f"  [{i}] {pid}")

    if failed:
        lines.append(f"\n以下 {len(failed)} 篇索引失败：")
        for err in failed:
            lines.append(f"  - {err}")

    all_indexed = _get_indexed_names()
    if len(all_indexed) > len(success):
        lines.append(f"\n知识库合计现有 {len(all_indexed)} 篇论文（含历史积累）。")

    lines.append("\n现在可以就任意论文提问了！")
    return "\n".join(lines)


@tool
def scan_and_index_new_papers() -> str:
    return _perform_scan()


@tool
def list_indexed_papers() -> str:
    """
    列出知识库中所有已成功索引的论文。

    【调用时机】：
    - 用户询问"有哪些论文"、"知识库里有什么"时

    Returns:
        已索引论文列表，含编号、文件名和本地文件状态。
    """
    indexed_names = _get_indexed_names()

    if not indexed_names:
        return (
            f"知识库目前为空。\n"
            f"请告诉我 arXiv ID，我自动下载；或将 PDF 放入 {PAPERS_DIR} 后说「扫描新论文」。"
        )

    lines = [f"知识库共有 {len(indexed_names)} 篇论文：\n"]
    for i, name in enumerate(sorted(indexed_names), 1):
        pdf_path = PAPERS_DIR / f"{name}.pdf"
        status = (
            "本地缓存存在" if pdf_path.exists() else " 源文件已移除（向量仍可检索）"
        )
        lines.append(f"  [{i}] {name}　{status}")

    lines += [
        "\n使用方式：",
        "  • 直接提问 → 跨所有论文检索作答",
        "  • 「研究第N篇」→ 定向深度问答",
    ]
    return "\n".join(lines)

@tool
def search_structural_elements(
    paper_id: str,
    query: str = "",
    element_type: str = "",
    element_id: str = "",
) -> str:
    """
    在结构单元表中精确查找定理、证明、定义、公式。

    【调用时机】（优先于 search_structured）：
    - 用户问"定理3.1是什么" → element_id="thm3.1" 或 element_id="3.1"
    - 用户问"定理3.1的证明" → element_type="proof", query="3.1"
    - 用户问"有哪些引理" → element_type="lemma"
    - 用户问某个具体定义 → element_type="definition", query="关键词"

    Args:
        paper_id:     论文 ID（必填）
        query:        关键词，模糊匹配 label 和 content（可选）
        element_type: 类型过滤：theorem / lemma / proof / definition / equation（可选）
        element_id:   精确匹配元素 ID 或编号，如 "thm3.1" 或 "3.1"（可选）

    Returns:
        匹配的结构单元列表，含完整内容和页码。
    """
    conn = sqlite3.connect(SQLITE_DB_FILE)

    conditions = ["paper_id = ?"]
    params: list = [paper_id]

    if element_type:
        # 支持 "theorem" 匹配 theorem/lemma/corollary
        if element_type == "theorem":
            conditions.append("element_type IN ('theorem', 'lemma', 'corollary', 'proposition')")
        else:
            conditions.append("element_type = ?")
            params.append(element_type)

    if element_id:
        # 支持模糊匹配，如输入 "3.1" 能匹配 "thm3.1"
        conditions.append("(element_id LIKE ? OR label LIKE ?)")
        params.extend([f"%{element_id}%", f"%{element_id}%"])

    if query:
        conditions.append("(label LIKE ? OR content LIKE ?)")
        params.extend([f"%{query}%", f"%{query}%"])

    sql = (
        "SELECT element_type, element_id, label, content, start_page, end_page, ref_ids "
        "FROM paper_structural_elements "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY start_page"
    )
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    if not rows:
        hint = ""
        if element_id:
            hint = f"（尝试去掉 element_id 过滤，或检查 ID 格式）"
        return (
            f"在论文 {paper_id} 中未找到匹配的结构单元。{hint}\n"
            f"建议：调用 get_paper_structure({paper_id!r}) 查看全部可用元素列表。"
        )

    lines = [f"找到 {len(rows)} 个结构单元：\n"]
    for etype, eid, label, content, sp, ep, ref_ids_json in rows:
        page_str = f"p.{sp}" if sp == ep or not ep else f"p.{sp}-{ep}"
        refs = json.loads(ref_ids_json) if ref_ids_json else []
        ref_str = f"  ↔ 关联：{', '.join(refs)}" if refs and any(refs) else ""

        lines.append(f"### [{etype.upper()}] {label} ({page_str})")
        lines.append(content[:600] + ("…" if len(content) > 600 else ""))
        if ref_str:
            lines.append(ref_str)
        lines.append("")

    return "\n".join(lines)



def _fetch_arxiv_title(arxiv_id: str) -> str | None:
    """获取 arXiv 论文的标题，失败时返回 None。"""
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(abs_url, headers=headers)
            if resp.status_code != 200:
                return None
            # 解析 HTML 中的标题（简单正则）
            import re

            match = re.search(
                r"<title>arXiv:[\w.]+\s+(.*?)</title>", resp.text, re.DOTALL
            )
            if match:
                title = match.group(1).strip()
                # 移除可能的多余空白和换行
                title = re.sub(r"\s+", " ", title)
                return title
    except Exception:
        pass
    return None


@tool
def load_paper_for_deep_analysis(
    arxiv_url_or_id: str | float | int, expected_title: str | None = None
) -> str:
    # ── 归一化 ID ────────────────────────────────────────────
    raw = str(arxiv_url_or_id).strip()
    if raw.startswith("http"):
        arxiv_id = raw.split("/abs/")[-1].split("/pdf/")[-1].removesuffix(".pdf")
    else:
        arxiv_id = raw.removesuffix(".pdf").strip()

    abs_url = f"https://arxiv.org/abs/{arxiv_id}"

    if not re.match(r"^[\w.\-]+$", arxiv_id):
        return f"非法的 arXiv ID 格式：「{arxiv_id}」\n合法示例：2301.12345 或 2301.12345v2"

    try:
        _cleanup_stuck_processing()

        if arxiv_id in _get_indexed_names():
            return (
                f"论文 **{arxiv_id}** 已在知识库中，无需重新索引。\n\n"
                f"可以直接提问，我将严格基于原文给出带引用的精准回答。"
            )

        # ── 标题获取（用于校验和返回信息）───────────────────
        title = _fetch_arxiv_title(arxiv_id)
        if title:
            print(f"[校验] 论文标题：{title}")
            if expected_title:
                import difflib
                ratio = difflib.SequenceMatcher(
                    None, expected_title.lower(), title.lower()
                ).ratio()
                if ratio < 0.6:
                    print(f"[警告] 标题差异较大：预期={expected_title}，实际={title}")
        else:
            print(f"[警告] 无法获取论文标题，请确认 arXiv ID {arxiv_id} 有效。")

        # ════════════════════════════════════════════════════
        # 路径一：LaTeX 源码（优先，仅限 arXiv ID 格式）
        # ════════════════════════════════════════════════════
        if is_arxiv_id(arxiv_id):
            latex_source = download_and_parse_arxiv_source(arxiv_id, TEX_CACHE_DIR)
        else:
            latex_source = None

        if latex_source:
            tex_path = TEX_CACHE_DIR / f"{arxiv_id}.tex"
            # tex_path 已由 download_and_parse_arxiv_source 写入

            print(f"[LaTeX] 开始向量化（{len(latex_source):,} 字符）...", flush=True)
            shared_knowledge.insert(
                name=arxiv_id,
                path=str(tex_path),
                reader=latex_reader,
                skip_if_exists=True,
            )

            structure_info, summary_info = _run_post_index_pipeline(
                arxiv_id, latex_reader, label="[LaTeX]"
            )

            title_line = f"\n论文标题：{title}" if title else ""
            return (
                f"论文 **{arxiv_id}** 已通过 **LaTeX 源码** 成功索引！{title_line}"
                f"{structure_info}{summary_info}\n\n"
                f"✅ 公式以原始 LaTeX 格式保留，检索精度最优。\n"
                f"原文链接：{abs_url}\n"
                f"源码缓存：{TEX_CACHE_DIR / f'{arxiv_id}.tex'}\n\n"
                f"现在可以就该论文的任何内容提问。"
            )

        # ════════════════════════════════════════════════════
        # 路径二：PDF + OCR（回退）
        # ════════════════════════════════════════════════════
        print(
            f"[回退] LaTeX 源码不可用，改用 PDF + PaddleOCR", flush=True
        )

        local_pdf_path = PAPERS_DIR / f"{arxiv_id}.pdf"
        if not local_pdf_path.exists():
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            print(f"[下载] 正在从 {pdf_url} 下载 PDF...", flush=True)
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                resp = client.get(pdf_url, headers=headers)

            if resp.status_code != 200:
                return (
                    f"LaTeX 源码和 PDF 均无法获取。\n"
                    f"PDF 下载失败（HTTP {resp.status_code}）：{pdf_url}\n"
                    f"请检查 arXiv ID 是否正确，或网络是否可访问 arxiv.org。"
                )

            local_pdf_path.write_bytes(resp.content)
            print(
                f"[下载] 已保存至 {local_pdf_path}（{len(resp.content) // 1024} KB）",
                flush=True,
            )
        else:
            print(f"[缓存] 命中本地 PDF {local_pdf_path}，跳过下载", flush=True)

        print(f"[OCR] 开始向量化 {arxiv_id}...", flush=True)
        shared_knowledge.insert(
            name=arxiv_id,
            path=str(local_pdf_path),
            reader=pdf_reader,
            skip_if_exists=True,
        )

        structure_info, summary_info = _run_post_index_pipeline(
            arxiv_id, pdf_reader, label="[OCR]"
        )

        title_line = f"\n论文标题：{title}" if title else ""
        return (
            f"论文 **{arxiv_id}** 已通过 **PDF + OCR** 成功索引。{title_line}"
            f"{structure_info}{summary_info}\n\n"
            f"⚠️ 注意：OCR 模式下公式以近似字符表示，LaTeX 语法不保留。\n"
            f"原文链接：{abs_url}\n"
            f"本地缓存：{local_pdf_path}\n\n"
            f"现在可以就该论文的任何内容提问。"
        )

    except Exception as e:
        return f"论文加载失败（{arxiv_id}）：{e}"


@tool
def search_arxiv_papers(query: str, max_results: int = 5) -> str:
    """
    在 arXiv 上检索最新学术论文，返回带 arXiv ID 的论文列表。

    【调用时机】：
    - 用户想探索某研究方向时
    - 用户说"帮我找 XXX 方向的论文"时
    - 知识库无论文，用户希望寻找新方向时

    检索后用户说「加载第N篇」，系统自动调用 load_paper_for_deep_analysis 完成下载索引。

    Args:
        query:       检索关键词（英文效果更佳）
        max_results: 返回数量（默认 3，最大 10）
    """
    import urllib.parse

    encoded_query = urllib.parse.quote(query.strip())
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded_query}"
        f"&start=0&max_results={min(max_results, 10)}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        if not entries:
            return f"未找到与「{query}」相关的论文，建议换用英文关键词重试。"

        lines = [f"arXiv 检索结果：「{query}」\n"]
        for i, entry in enumerate(entries, 1):
            title_elem = entry.find("atom:title", ns)
            title = (
                title_elem.text.strip().replace("\n", " ")
                if title_elem is not None and title_elem.text
                else "无标题"
            )
            summary_elem = entry.find("atom:summary", ns)
            summary = (
                summary_elem.text.strip().replace("\n", " ")
                if summary_elem is not None and summary_elem.text
                else "无摘要"
            )
            abs_elem = entry.find("atom:id", ns)
            abs_url = (
                abs_elem.text.strip() if abs_elem is not None and abs_elem.text else ""
            )
            arxiv_id = abs_url.split("/abs/")[-1] if abs_url else "未知ID"
            author_elems = entry.findall("atom:author", ns)
            authors = []
            for a in author_elems:
                name_elem = a.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            author_str = ", ".join(authors[:3]) if authors else "未知作者"
            if len(authors) > 3:
                author_str += "等"
            lines.append(
                f"**[{i}] {title}**\n"
                f"  - arXiv ID：`{arxiv_id}`\n"
                f"  - 作者：{author_str}\n"
                f"  - 摘要：{summary[:280]}…\n"
                f"  - 链接：{abs_url if abs_url else '无链接'}\n"
            )

        lines += [
            "─" * 48,
            " 说「加载第N篇」，我自动下载 PDF 并完成索引，全程无需手动操作。",
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"arXiv 检索失败：{e}"

# ─────────────────────────────────────────────────────────────
# 工具定义：笔记保存
# ─────────────────────────────────────────────────────────────

@tool
def save_note(filename: str, content: str) -> str:
    """
    将笔记内容保存到本地文件（Markdown 格式）。

    【调用时机】：
    - 用户要求保存总结、笔记或任何文本内容时。

    Args:
        filename: 文件名（不含扩展名），将自动添加 .md 后缀。
        content: 要保存的文本内容。

    Returns:
        成功或失败消息。
    """
    import os
    from pathlib import Path

    if not filename.endswith(".md"):
        filename += ".md"
    path = NOTES_DIR / filename
    try:
        path.write_text(content, encoding="utf-8")
        return f"笔记已保存至：{path}"
    except Exception as e:
        return f"保存失败：{e}"

@tool
def list_notes() -> str:
    """
    列出所有已保存的笔记文件。

    【调用时机】：
    - 用户询问“有哪些笔记”或“查看已保存的笔记”时。

    Returns:
        笔记文件列表，包含文件名和大小。
    """
    import os
    from pathlib import Path

    files = list(NOTES_DIR.glob("*.md"))
    if not files:
        return "暂无笔记文件。"
    lines = [f"共有 {len(files)} 个笔记文件："]
    for i, f in enumerate(sorted(files), 1):
        size = f.stat().st_size
        lines.append(f"  [{i}] {f.name} ({size} 字节)")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
# 工具定义：结构化检索
# ─────────────────────────────────────────────────────────────

@tool
def get_paper_structure(paper_id: str) -> str:
    """
    获取论文的结构化大纲：章节层级、定理/引理列表、定义、证明、关键公式。

    【调用时机】：
    - 用户说"论文结构是什么"、"有哪些定理"、"列出定义"时
    - 在深入提问某篇论文的具体定理/证明前，先调用此工具了解全貌

    Args:
        paper_id: 论文在知识库中的名称（通常是 arXiv ID，如 2301.12345）

    Returns:
        格式化的论文结构大纲（Markdown），包含章节、定理、定义、证明列表。
        若未找到结构数据，返回提示信息。
    """
    structure = load_paper_structure(paper_id)
    if structure is None:
        return (
            f"未找到论文 {paper_id} 的结构数据。\n"
            f"可能原因：该论文在结构提取功能上线前已索引。\n"
            f"建议：删除后重新加载该论文以触发结构提取。"
        )
    return format_structure_for_display(structure)

@tool
def search_structured(
    query: str,
    paper_id: str = "",
    element_type: str = "",
) -> str:
    """
    在知识库中按结构类型精准检索论文内容。

    【调用时机】：
    - 用户问"定理3.1是什么"→ element_type="theorem"
    - 用户问"定理3.1怎么证明的"→ element_type="proof"
    - 用户问"XX的定义"→ element_type="definition"
    - 需要精确定位特定类型内容时

    Args:
        query: 检索关键词（如 "theorem 3.1 statement", "proof of main theorem"）
        paper_id: 可选，限定在某篇论文内检索（论文名称/arXiv ID）
        element_type: 可选，按结构类型过滤。
                     可选值: theorem, proof, definition, equation

    Returns:
        匹配的文本块列表，带页码和结构类型标注。
    """
    filters: dict[str, any] = {}
    if paper_id:
        filters["name"] = paper_id
    if element_type:
        filters["element_types"] = element_type

    results = vector_db.search(query, limit=8, filters=filters if filters else None)

    if not results:
        filter_desc = ""
        if paper_id:
            filter_desc += f" 论文={paper_id}"
        if element_type:
            filter_desc += f" 类型={element_type}"
        return f"未找到匹配结果。查询: '{query}'{filter_desc}\n建议：尝试放宽过滤条件或换用不同关键词。"

    lines = [f"找到 {len(results)} 条结果：\n"]
    for i, doc in enumerate(results, 1):
        meta = doc.meta_data or {}
        page = meta.get("page", "?")
        section = meta.get("section", "")
        etypes = meta.get("element_types", "")
        header_parts = [f"p.{page}"]
        if section:
            header_parts.append(section)
        if etypes:
            header_parts.append(f"[{etypes}]")
        header = " | ".join(header_parts)

        content_preview = (doc.content or "")[:400]
        lines.append(f"**[{i}]** ({header})\n{content_preview}\n")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 工具定义：三级递进（全局索引 → 概要 → 深读）
# ─────────────────────────────────────────────────────────────


@tool
def browse_paper_catalog() -> str:
    """
    返回所有已索引论文的紧凑索引：每篇含标题和三维标签（领域/内容/技巧）。

    【调用时机】：
    - 用户提问时，先调用此工具浏览全局索引，挑出可能相关的论文
    - 用户问"有哪些论文"、"知识库里有什么"时

    Returns:
        紧凑论文目录，每篇 2 行（标题 + 三维标签），适合一次性通读筛选。
    """
    summaries = load_all_paper_summaries()
    if not summaries:
        indexed = _get_indexed_names()
        if not indexed:
            return "知识库为空，请先添加论文。"
        lines = [f"知识库有 {len(indexed)} 篇论文（尚无摘要信息，可能需要重新索引）："]
        for i, name in enumerate(sorted(indexed), 1):
            lines.append(f"  {i}. [{name}]")
        return "\n".join(lines)

    lines = [f"论文索引（共 {len(summaries)} 篇）：\n"]
    for i, s in enumerate(summaries, 1):
        pid = s["paper_id"]
        title = s["title"] or pid
        field = ", ".join(s.get("field_tags", [])) or "—"
        content = ", ".join(s.get("content_tags", [])) or "—"
        technique = ", ".join(s.get("technique_tags", [])) or "—"
        lines.append(f"{i}. [{pid}] {title}")
        lines.append(f"   领域: {field} | 内容: {content} | 技巧: {technique}")
    lines.append(f"\n提示：对感兴趣的论文调用 get_paper_overview(paper_id) 查看详细概要。")
    return "\n".join(lines)


@tool
def get_paper_overview(paper_id: str) -> str:
    """
    获取论文的详细概要：主要内容、章节结构、证明思路、核心技巧。

    【调用时机】：
    - 通过 browse_paper_catalog 筛出候选后，逐篇调用此工具精选
    - 用户问某篇论文"讲了什么"、"主要内容"、"创新点"时

    Args:
        paper_id: 论文在知识库中的名称（通常是 arXiv ID 或文件名）

    Returns:
        格式化的论文详细概要（Markdown）。
    """
    summary = load_paper_summary(paper_id)
    structure = load_paper_structure(paper_id)

    if not summary and not structure:
        return (
            f"未找到论文 {paper_id} 的概要或结构数据。\n"
            f"可能原因：该论文在此功能上线前已索引。\n"
            f"建议：删除后重新加载以触发摘要提取。"
        )

    lines: list[str] = []

    if summary:
        title = summary.get("title", paper_id)
        lines.append(f"# {title}\n")

        field_tags = summary.get("field_tags", [])
        content_tags = summary.get("content_tags", [])
        technique_tags = summary.get("technique_tags", [])
        if field_tags or content_tags or technique_tags:
            lines.append(f"**领域**: {', '.join(field_tags) if field_tags else '—'}")
            lines.append(f"**内容**: {', '.join(content_tags) if content_tags else '—'}")
            lines.append(f"**技巧**: {', '.join(technique_tags) if technique_tags else '—'}\n")

        abstract = summary.get("abstract", "")
        if abstract:
            lines.append(f"## 主要内容\n{abstract}\n")

        core_techniques = summary.get("core_techniques", [])
        if core_techniques:
            lines.append("## 核心技巧")
            for t in core_techniques:
                lines.append(f"- {t}")
            lines.append("")

        proof_approaches = summary.get("proof_approaches", {})
        if proof_approaches:
            lines.append("## 证明思路")
            for thm, approach in proof_approaches.items():
                lines.append(f"- **{thm}**: {approach}")
            lines.append("")

    if structure:
        sections = structure.get("sections", [])
        if sections:
            lines.append("## 章节大纲")
            for sec in sections:
                indent = "  " * (sec.get("level", 1) - 1)
                lines.append(f"{indent}- **{sec['id']}** {sec.get('title', '')} (p.{sec.get('page', '?')})")
            lines.append("")

        theorems = structure.get("theorems", [])
        if theorems:
            lines.append("## 定理/引理")
            for thm in theorems:
                stmt = thm.get("statement", "")[:120]
                lines.append(f"- **{thm.get('label', thm['id'])}** (p.{thm.get('page', '?')}): {stmt}")
            lines.append("")

    page_count = get_paper_page_count(paper_id)
    if page_count:
        lines.append(f"---\n全文共 {page_count} 页，可用 `read_paper_pages` 或 `read_paper_section` 深读原文。")

    return "\n".join(lines) if lines else f"论文 {paper_id} 的概要信息为空。"


@tool
def read_paper_pages(paper_id: str, start_page: int, end_page: int = 0) -> str:
    """
    读取论文指定页的 OCR 原文。

    【调用时机】：
    - 看完概要后需要深读某几页原文时
    - 需要查看某个证明、定义的完整内容时

    Args:
        paper_id: 论文 ID
        start_page: 起始页码（从 1 开始）
        end_page: 结束页码（包含），为 0 时只读 start_page 一页

    Returns:
        指定页的 OCR 原文。
    """
    paper_id = str(paper_id)
    
    if end_page <= 0:
        end_page = start_page
    if start_page < 1:
        return "页码必须从 1 开始。"
    if end_page - start_page > 10:
        return "一次最多读取 10 页，请缩小范围。"

    total = get_paper_page_count(paper_id)
    if total == 0:
        return f"未找到论文 {paper_id} 的原文数据。可能需要重新索引。"

    rows = load_paper_pages(paper_id, start_page, end_page)
    if not rows:
        return f"论文 {paper_id} 没有第 {start_page}-{end_page} 页的数据（全文共 {total} 页）。"

    lines = [f"论文 {paper_id} 第 {start_page}-{end_page} 页（共 {total} 页）：\n"]
    for page_num, content in rows:
        lines.append(f"--- 第 {page_num} 页 ---")
        lines.append(content if content else "[空白页]")
        lines.append("")

    return "\n".join(lines)


@tool
def read_paper_section(paper_id: str, section_id: str) -> str:
    """
    读取论文指定章节的原文（根据结构自动定位页码范围）。

    【调用时机】：
    - 看完概要后需要深读某个章节时
    - 用户说"读第3章"、"看 section 2.1"时

    Args:
        paper_id: 论文 ID
        section_id: 章节 ID（如 "sec1", "sec2.1"），可从 get_paper_overview 获取

    Returns:
        该章节覆盖页的 OCR 原文。
    """
    structure = load_paper_structure(paper_id)
    if not structure:
        return f"未找到论文 {paper_id} 的结构数据。"

    sections = structure.get("sections", [])
    target = None
    target_idx = -1
    for idx, sec in enumerate(sections):
        if sec["id"] == section_id or sec.get("title", "").lower() == section_id.lower():
            target = sec
            target_idx = idx
            break

    if not target:
        available = ", ".join(f"{s['id']}({s.get('title', '')})" for s in sections)
        return f"未找到章节 {section_id}。可用章节：{available}"

    start_page = target.get("page", 1)
    if target_idx + 1 < len(sections):
        end_page = sections[target_idx + 1].get("page", start_page-1)
    else:
        total = get_paper_page_count(paper_id)
        end_page = total if total > 0 else start_page

    end_page = min(end_page, start_page + 9)

    rows = load_paper_pages(paper_id, start_page, end_page)
    if not rows:
        return f"未找到论文 {paper_id} 第 {start_page}-{end_page} 页的原文数据。"

    sec_title = target.get("title", section_id)
    lines = [f"章节 [{section_id}] {sec_title}（p.{start_page}-{end_page}）：\n"]
    for page_num, content in rows:
        lines.append(f"--- 第 {page_num} 页 ---")
        lines.append(content if content else "[空白页]")
        lines.append("")
    
    if "notation" in targets or "all" in targets:
        conn.execute("DELETE FROM paper_notations WHERE paper_id = ?", (paper_id,))
        deleted.append("notation")
        
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 工具定义：删除 / 重索引
# ─────────────────────────────────────────────────────────────
_ALL_TARGETS = ["vector", "structure", "summary", "pages", "notation"]


def _delete_paper_data(paper_id: str, targets: list[str]) -> list[str]:
    """Delete specified data for a paper. Returns list of successfully deleted targets."""
    if "all" in targets:
        targets = list(_ALL_TARGETS)

    deleted: list[str] = []

    if "vector" in targets:
        try:
            contents, _ = shared_knowledge.get_content()
            for c in contents:
                if c.name == paper_id and c.id is not None:
                    shared_knowledge.remove_content_by_id(c.id)
            deleted.append("vector")
        except Exception as e:
            print(f"[删除] 向量数据删除失败: {e}", flush=True)

    conn = sqlite3.connect(SQLITE_DB_FILE)
    try:
        if "structure" in targets:
            conn.execute("DELETE FROM paper_structures WHERE paper_id = ?", (paper_id,))
            conn.execute("DELETE FROM paper_structural_elements WHERE paper_id = ?", (paper_id,))
            deleted.append("structure")
        if "summary" in targets:
            conn.execute("DELETE FROM paper_summaries WHERE paper_id = ?", (paper_id,))
            deleted.append("summary")
        if "pages" in targets:
            conn.execute("DELETE FROM paper_pages WHERE paper_id = ?", (paper_id,))
            deleted.append("pages")
        conn.commit()
    except Exception as e:
        print(f"[删除] SQLite 数据删除失败: {e}", flush=True)
    finally:
        conn.close()

    return deleted


@tool
def delete_paper_data(paper_id: str, targets: str = "all") -> str:
    """
    选择性删除论文的索引数据。

    【调用时机】：
    - 用户说"删除某篇论文"、"清除索引"时
    - 需要只更新部分数据（如重新生成摘要）时

    Args:
        paper_id: 论文 ID（如 sat-matching 或 2301.12345）
        targets: 要删除的数据类型，逗号分隔。可选值：
                 vector（向量嵌入）、structure（结构）、summary（摘要标签）、
                 pages（OCR原文）、all（全部，默认）
                 例如："summary,pages" 只删摘要和原文

    Returns:
        删除结果说明。
    """
    target_list = [t.strip() for t in targets.split(",") if t.strip()]
    if not target_list:
        target_list = ["all"]

    invalid = [t for t in target_list if t not in _ALL_TARGETS + ["all"]]
    if invalid:
        return f"无效的 targets: {invalid}。可选值: vector, structure, summary, pages, all"

    deleted = _delete_paper_data(paper_id, target_list)
    if not deleted:
        return f"论文 {paper_id} 没有找到可删除的数据。"

    return f"已删除论文 **{paper_id}** 的以下数据：{', '.join(deleted)}。\n可调用 `reindex_paper` 重新索引，或 `scan_and_index_new_papers` 扫描。"


@tool
def reindex_paper(paper_id: str) -> str:
    """
    删除论文全部索引数据并重新执行完整索引流程（OCR → 存原文 → 结构提取 → 摘要生成 → 向量嵌入）。

    【调用时机】：
    - 用户说"重新索引"、"重建索引"时
    - 旧论文缺少摘要/原文数据，需要走新流程时

    Args:
        paper_id: 论文 ID（如 sat-matching 或 2301.12345）

    Returns:
        重索引结果。
    """
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        return f"未找到论文 PDF 文件：{pdf_path}\n请确认 paper_id 正确，或先下载论文。"

    print(f"[重索引] 正在删除 {paper_id} 的旧数据...", flush=True)
    deleted = _delete_paper_data(paper_id, ["all"])
    print(f"[重索引] 已清除: {', '.join(deleted) if deleted else '无旧数据'}", flush=True)

    print(f"[重索引] 开始重新索引 {paper_id}...", flush=True)
    try:
        # 优先 LaTeX 源码
        active_reader = pdf_reader
        used_latex = False

        if is_arxiv_id(paper_id):
            latex_source = download_and_parse_arxiv_source(paper_id, TEX_CACHE_DIR)
            if latex_source:
                tex_path = TEX_CACHE_DIR / f"{paper_id}.tex"
                shared_knowledge.insert(
                    name=paper_id,
                    path=str(tex_path),
                    reader=latex_reader,
                    skip_if_exists=True,
                )
                active_reader = latex_reader
                used_latex = True

        if not used_latex:
            shared_knowledge.insert(
                name=paper_id,
                path=str(pdf_path),
                reader=pdf_reader,
                skip_if_exists=True,
            )

        source_label = "LaTeX 源码" if used_latex else "PDF + OCR"
        print(f"[重索引] 向量嵌入完成（{source_label}）", flush=True)

        structure_info, summary_info = _run_post_index_pipeline(
            paper_id, active_reader, label=f"[重索引/{source_label}]"
        )

        return (
            f"论文 **{paper_id}** 重新索引完成（{source_label}）！"
            f"{structure_info}{summary_info}\n\n"
            f"可用 `get_paper_overview('{paper_id}')` 查看概要。"
        )
    except Exception as e:
        return f"重索引失败：{e}"

@tool
def get_paper_notation(paper_id: str, query: str = "") -> str:
    """
    查询论文的符号表（notation map）。

    【调用时机】：
    - 遇到不认识的符号时（如 "\\mathcal{F} 是什么意思"）
    - 回答涉及公式前，先确认符号含义
    - 用户问"这篇文章里 X 代表什么"时

    Args:
        paper_id: 论文 ID
        query:    可选，按 latex 或 ascii 或含义关键词过滤；为空时返回全部

    Returns:
        符号定义列表，带首次出现页码和原文片段。
    """
    symbols = load_notation_map(paper_id)
    if not symbols:
        return (
            f"论文 {paper_id} 暂无符号表数据。\n"
            f"可能是旧版本索引，建议调用 reindex_paper('{paper_id}') 重建。"
        )

    if query:
        q = query.lower()
        symbols = [
            s for s in symbols
            if q in s["latex"].lower()
            or q in (s["ascii_repr"] or "").lower()
            or q in (s["meaning"] or "").lower()
        ]

    if not symbols:
        return f"未找到与「{query}」匹配的符号。"

    lines = [f"论文 {paper_id} 符号表（{len(symbols)} 条）：\n"]
    for s in symbols:
        page = f"p.{s['first_page']}" if s["first_page"] else ""
        ctx = f"\n  > {s['context']}" if s.get("context") else ""
        lines.append(f"- `{s['latex']}` ({s['ascii_repr']})  {page}")
        lines.append(f"  {s['meaning']}{ctx}")
    return "\n".join(lines)
# ─────────────────────────────────────────────────────────────
# Agent 构建（4 角色架构）
# ─────────────────────────────────────────────────────────────

# 1. arXiv Researcher — 外网检索 + 下载
arxiv_researcher = Agent(
    name="arXiv Researcher",
    role="负责在 arXiv 上检索、下载论文并触发索引",
    model=shared_llm,
    tools=[search_arxiv_papers, load_paper_for_deep_analysis],
    instructions=dedent("""
        你是论文检索助手。你必须严格遵守以下「两步走」工作流，绝不能自作主张跳步！

        **第一步：仅搜索（当用户询问某方向的论文时）**
        - 只能调用 `search_arxiv_papers`。
        - 【绝对红线】：获取到搜索结果后，必须**立即停止思考并输出结果给用户**，询问用户对哪篇感兴趣。**绝对禁止**在同一轮对话中紧接着调用 `load_paper_for_deep_analysis`。

        **第二步：仅下载（当用户明确说“下载/加载第N篇”或提供具体 arXiv ID 时）**
        - 从对话历史的搜索结果中提取对应的 arXiv ID。
        - 调用 `load_paper_for_deep_analysis` 进行下载和自动索引。
        
        记住：搜索和下载必须跨越两轮独立的对话！如果用户没有明确下达下载指令，你只能展示列表！
        """),
    markdown=True,
)

# 2. Paper Librarian — 索引管理专员
paper_librarian = Agent(
    name="Paper Librarian",
    role="负责论文索引的扫描、删除和重建",
    model=shared_llm,
    tools=[
        scan_and_index_new_papers,
        delete_paper_data,
        reindex_paper,
        list_indexed_papers,
    ],
    instructions=dedent("""
        你是论文管理员，负责知识库的索引维护。职责：

        【扫描索引】：
        - 调用 `scan_and_index_new_papers` 扫描新论文并自动索引

        【删除数据】：
        - 调用 `delete_paper_data(paper_id, targets)` 选择性删除
        - targets 可选: vector, structure, summary, pages, all
        - 执行前简要说明将删除什么，执行后报告结果

        【重新索引】：
        - 调用 `reindex_paper(paper_id)` 删除旧数据并重跑完整索引
        - 适用于旧论文缺少摘要/结构数据的情况

        【列出论文】：
        - 调用 `list_indexed_papers` 查看知识库中所有论文

        操作原则：先报告当前状态，再执行操作，最后确认结果。
        重要：你只能使用上述列举的工具，绝对不要发明或调用任何未列出的工具。如果用户需求无法通过这些工具满足，请说明限制。
    """),
    markdown=True,
)

# 3. Deep Reader — 精读 + 数学推理专家
deep_reader = Agent(
    name="Deep Reader",
    role="负责论文深度精读、数学内容解析与推理",
    model=shared_llm,
    tools=[
        browse_paper_catalog,
        get_paper_overview,
        read_paper_pages,
        read_paper_section,
        get_paper_structure,
        search_structural_elements,
        search_structured,
        get_paper_notation,
        get_proof_chain, 
    ],
    knowledge=shared_knowledge,
    search_knowledge=True,
    add_knowledge_to_context=True,
    instructions=dedent("""
        你是数学论文精读与推理专家。
        绝对红线：回答论文内容必须且只能基于工具返回的结果，绝不编造！
        
        【遇到符号先查符号表】：
        - 解析公式前，先调 get_paper_notation(paper_id, query="符号名") 确认含义
        - 多篇论文对比时，逐篇查询符号表，若同一符号在不同论文含义不同，必须在回答开头显式声明：「注意：以下 X 在论文A中指...，在论文B中指...」
  
        【结构单元检索优先级】：
        1. 已知元素编号（如"定理3.1"）→ 优先 search_structural_elements(paper_id, element_id="3.1")
        2. 按类型浏览（如"有哪些引理"）→ search_structural_elements(paper_id, element_type="lemma")
        3. 关键词模糊（如"连续性相关的定义"）→ search_structural_elements(paper_id, query="连续")
        4. 上述均无结果 → 降级使用 search_structured 或向量检索
        5. 获取关联证明：解析返回的 ref_ids，再次调用 search_structural_elements 取证明内容

        【证明依赖链查询】：
        - 用户问"完整证明"或"为什么这个定理成立"→ 先调 get_proof_chain(paper_id, element_id)
        - 拿到链后，从最底层叶节点（基础引理/定义）往上讲，而不是从定理往下读（因为读者需要先理解前置条件，再看主定理的证明）
        - 若某条依赖边的 depends_on 为空（说明该论文索引时依赖提取未运行），降级为 search_structural_elements 手动查引用
  
        【定理/证明定位】：
        - 问定理内容 → 从 overview 定理列表定位页码，再 read_paper_pages 读原文
        - 问证明思路 → 先看 overview 的 proof_approaches，需细节再 read_paper_section
        - 也可用 `search_structured(query, element_type="theorem/proof/definition")` 辅助

        【数学推理能力 — 核心差异化能力】：
        读取原文后，不要只复制粘贴！你必须：
        1. 用自然语言解释证明的核心思路（"为什么这样做"、"关键洞察是什么"）
        2. 指出证明中的关键转折步骤和技巧
        3. 对比不同定理的证明方法异同（如果涉及多个定理）
        4. 简化复杂表达式时补充中间推导步骤
        5. LaTeX 公式保持原样，但用括号注释说明关键符号含义

        【回答层次（必须遵循）】：
        1. 先给出 1-2 句直觉性回答（让用户快速理解要点）
        2. 再给出严谨的数学细节（带 [Theorem X, p.Y] 格式引用）
        3. 如有必要，补充"直觉理解"或"类比"帮助消化

        【回答格式】：
        - 引用格式：[Theorem 3.1, p.5] 或 [第3页]
        - 公式保留 LaTeX（$ 或 $$ 包裹）
        - 检索不到则明确说"知识库中未找到"
        
        重要：你只能使用上述列举的工具，绝对不要发明或调用任何未列出的工具。如果用户需求无法通过这些工具满足，请说明限制。
    """),
    markdown=True,
)

# 4. Team Leader — 增强版主管
arxiv_team = Team(
    name="arXiv Team",
    model=shared_llm,
    members=[arxiv_researcher, paper_librarian, deep_reader],
    # 主管掌控记忆和对话历史
    db=agent_db,
    num_history_runs=10,
    add_history_to_context=True,
    enable_agentic_memory=True,
    tools=[save_note, list_notes, list_indexed_papers,browse_paper_catalog],
    instructions=dedent("""
        你是主管，负责将用户请求路由给三位专家：
        1. `arXiv Researcher`（外网与下载专员）：
           - 拥有工具：`search_arxiv_papers`, `load_paper_for_deep_analysis`
           - 委派场景：
             a. 用户说“找/搜方向的论文” -> 委派它去搜索。
             b. 用户说“下载/加载论文”或提供 arXiv ID -> **必须委派它调用 `load_paper_for_deep_analysis` 下载并索引**（不要怀疑，它有这个能力！）。

        2. `Paper Librarian`（本地与索引专员）：
           - 拥有工具：`scan_and_index_new_papers`, `delete_paper_data`, `reindex_paper`, `list_indexed_papers`
           - 委派场景：用户询问本地库状态、要求扫描本地新文件、删除或重建索引时。

        3. `Deep Reader`（精读与推理专员）：
           - 拥有工具：各类读取、检索工具
           - 委派场景：只要涉及回答论文具体内容、定理、公式、摘要等，必须委派给它。
        ═══════════════════════════════════════
        【关键动作规范：处理“加载第 N 篇”】
        ═══════════════════════════════════════
        当用户说「加载第 N 篇」或「下载第 N 篇」时，你必须严格按以下步骤执行，绝不能报错或说做不到：
        1. 从上文对话历史（搜索结果）中，找到第 N 篇论文对应的 `arXiv ID`。
        2. 将这个 `arXiv ID` 作为明确的任务目标。
        3. **立刻委派 `arXiv Researcher` 执行下载任务**，明确告诉它：“请调用工具加载这篇论文，ID 是 XXX”。
        
        绝对禁止：不要试图让 Paper Librarian 去扫描 arXiv ID，也不要告诉用户你无法下载。

        ═══════════════════════════════════════
        【重要原则】
        ═══════════════════════════════════════
        - 凡是涉及论文**内容**（标题、摘要、结构、定理、原文、解释证明）的问题，**必须**委派 Deep Reader。
        - 凡是涉及**论文列表、索引状态、扫描、删除、重建**，委派 Paper Librarian。
        - 凡是涉及**在 arXiv 搜索新方向的论文**，委派 arXiv Researcher 仅执行搜索并向用户展示结果。
        - 凡是涉及**下载/加载特定论文**，再委派 arXiv Researcher 执行下载操作。
        - 自己只处理笔记保存/列出等通用功能。
        ═══════════════════════════════════════
        【Focus Paper 状态管理】
        ═══════════════════════════════════════
        1. 通过对话历史记住用户当前聚焦的论文（Focus Paper）
        2. 用户表示要研究某篇时，设为 Focus Paper
        3. 后续提问未指明论文时，默认问 Focus Paper
        4. 委派时必须带上 paper_id
        5. 无 Focus Paper 且指代不明时，反问确认
        
        ═══════════════════════════════════════
        【跨论文综合】
        ═══════════════════════════════════════
        当用户问涉及多篇论文的问题（对比、综述）：
        1. 自己调 browse_paper_catalog 确定相关论文
        2. 逐篇调 get_paper_overview 获取各篇概要
        3. 在概要层面综合回答（对比方法、总结共性与差异）
        4. 如需原文细节佐证，再委派 Deep Reader 精读特定章节
        
        重要：你只能使用上述列举的工具，绝对不要发明或调用任何未列出的工具。
    """),
    markdown=True,
    stream=True,
    session_id=TEAM_SESSION_ID,
    user_id="researcher",
    show_members_responses=True,
)


def interactive_cli():
    """同步命令行交互入口。"""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_polluted_session()

    print("正在扫描论文文件夹...\n")
    scan_result = _perform_scan()
    print(scan_result)
    print()

    # ── 主交互循环 ──────────────────────────────────────────
    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！祝学术研究顺利 ")
            break

        if not raw:
            continue
        if raw.lower() in ("exit", "quit", "bye", "退出"):
            print("再见！祝学术研究顺利 ")
            break

        arxiv_team.print_response(raw, stream=True)
        print()


if __name__ == "__main__":
    interactive_cli()