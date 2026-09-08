"""论文库：多论文索引 / 注册 / 删除（Qdrant 持久化 + SQLite 注册表）。

每篇论文额外生成一个「身份片段」（标题 + 第一页开头含摘要），
让「哪篇论文做了 X」类问题直接命中论文身份，而非散落正文关键词。
同名文件重复上传 = 稳定 paper_id 覆盖更新（幂等去重）。
"""
import hashlib
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .chunker import chunk_pages
from .config import Settings
from .embeddings import Embedder
from .pdf_parser import parse_pdf
from .vector_store import QdrantStore, point_id

REGISTRY_DB = Path("data/library.sqlite3")
# 原始 PDF 存档目录（Streamlit 静态服务 → 论文预览用）
STATIC_PAPERS_DIR = Path(__file__).resolve().parents[1] / "static" / "papers"


@dataclass
class PaperMeta:
    paper_id: str
    title: str
    filename: str
    pages: int
    chunks: int  # 正文片段数（不含身份片段）
    indexed_at: str
    md5: str = ""  # 文件内容指纹
    skipped: bool = False  # 本次调用因内容未变化而跳过向量化
    existed: bool = False  # 入库前库中已存在该论文（True=本次为覆盖更新）


# 论文标题提取时的「杂行」黑名单：页眉/版权/DOI/收稿信息/期刊名/纯数字等
JUNK_TITLE_PATTERNS = (
    r"received\s+\d{1,2}\s+\w+\s+\d{4}",
    r"revised\s+\d{1,2}\s+\w+\s+\d{4}",
    r"accepted\s+\d{1,2}\s+\w+\s+\d{4}",
    r"date of publication",
    r"digital object identifier",
    r"^\s*\d+\s*$",
    r"©|copyright|all rights reserved",
    r"personal use is permitted",
    r"ieee trans", r"acm trans", r"proceedings of",
    r"arxiv", r"preprint",
    r"creativecommons", r"creative commons",
    r"^\s*vol\.?\s*\d", r"^\s*no\.?\s*\d.*\d{4}",
)


def _is_junk_title_line(line: str) -> bool:
    low = line.lower()
    return any(re.search(p, low) for p in JUNK_TITLE_PATTERNS)


def _extract_title_from_page(page) -> str:
    """论文标题 ≈ 第一页上半页、字号最大、非杂行的文本（支持两行标题拼接）。

    行级组织（block→line→spans），行字号取 spans 最大值；
    杂行黑名单排除 IEEE/ACM 版权块、收稿日期行、DOI、期刊名、纯数字行等。
    兜底：提取不到时返回空串，由调用方回退到首个非杂行长行/文件名。
    """
    try:
        doc = page.get_text("dict")
    except Exception:
        return ""
    page_height = page.rect.height or 800
    lines = []
    for block in doc.get("blocks", []):
        for ln in block.get("lines", []):
            text = " ".join(" ".join(s.get("text", "") for s in ln.get("spans", [])).split())
            size = max((float(s.get("size", 0.0)) for s in ln.get("spans", [])), default=0.0)
            bbox = ln.get("bbox") or [0, 0, 0, 0]
            if text:
                lines.append({"text": text, "size": size, "y": float(bbox[1]), "top": float(bbox[1]) / page_height <= 0.55})
    if not lines:
        return ""
    max_size = max(l["size"] for l in lines)
    cands = [
        l for l in lines
        if l["top"] and not _is_junk_title_line(l["text"]) and l["size"] >= max_size * 0.6 and len(l["text"]) >= 8
    ]
    if not cands:
        return ""
    cands.sort(key=lambda l: (-l["size"], l["y"]))
    title = cands[0]["text"]
    # 两行标题：与首行字号相近、紧随其下的非杂行拼接
    first = cands[0]
    nxt = [
        l for l in cands
        if l is not first and abs(l["size"] - first["size"]) <= 0.8 and 0 < l["y"] - first["y"] <= 60
    ]
    if nxt:
        nxt.sort(key=lambda l: l["y"])
        title = f"{title} {nxt[0]['text']}".strip()
    return title if len(title) >= 8 else ""


class PaperLibrary:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.store = QdrantStore(self.settings)
        self.embedder = Embedder(self.settings)
        self.store.ensure_collection()
        self.db = REGISTRY_DB
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db) as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS papers (
                       paper_id TEXT PRIMARY KEY,
                       title TEXT, filename TEXT,
                       pages INTEGER, chunks INTEGER,
                       indexed_at TEXT)"""
            )
            try:  # 幂等迁移：老库补 md5 列
                con.execute("ALTER TABLE papers ADD COLUMN md5 TEXT")
            except sqlite3.OperationalError:
                pass  # 列已存在

    def _upsert_registry(self, meta: PaperMeta) -> None:
        with sqlite3.connect(self.db) as con:
            con.execute(
                "INSERT OR REPLACE INTO papers VALUES (?, ?, ?, ?, ?, ?, ?)",
                (meta.paper_id, meta.title, meta.filename, meta.pages, meta.chunks, meta.indexed_at, meta.md5),
            )

    # ---------- 查询 ----------

    def list_papers(self) -> list[PaperMeta]:
        with sqlite3.connect(self.db) as con:
            rows = con.execute(
                "SELECT paper_id, title, filename, pages, chunks, indexed_at, md5 FROM papers ORDER BY indexed_at DESC"
            ).fetchall()
        return [PaperMeta(*row) for row in rows]

    def get_paper(self, paper_id: str) -> PaperMeta | None:
        with sqlite3.connect(self.db) as con:
            row = con.execute(
                "SELECT paper_id, title, filename, pages, chunks, indexed_at, md5 FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        return PaperMeta(*row) if row else None

    def _find_by_md5(self, md5: str) -> PaperMeta | None:
        """按内容指纹查论文（标题字符串波动时仍能识别为同一篇）。"""
        with sqlite3.connect(self.db) as con:
            row = con.execute(
                "SELECT paper_id, title, filename, pages, chunks, indexed_at, md5 FROM papers WHERE md5 = ? LIMIT 1",
                (md5,),
            ).fetchone()
        return PaperMeta(*row) if row else None

    # ---------- 写入 / 删除 ----------

    @staticmethod
    def _paper_id_for(title: str) -> str:
        """论文身份锚定标题：同一篇论文跨文件名去重；更新版重传 → 同 id 覆盖。"""
        norm = title.strip().lower()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paperqa:paper:{norm}"))

    def index_paper(self, pdf_path: str, title: str | None = None, filename: str | None = None) -> PaperMeta:
        """解析 → 分块 → 身份片段 → 向量化 → 写入 Qdrant → 注册表登记。

        查重在库层完成：标题相同且 MD5 未变化的重复上传直接跳过（零向量化成本）；
        标题相同但内容更新的重传 → 同 paper_id 覆盖重建。
        """
        path = Path(pdf_path)
        filename = (filename or path.name).strip()

        pages = parse_pdf(str(path))
        chunks = chunk_pages(pages)
        file_md5 = hashlib.md5(path.read_bytes()).hexdigest()

        # 标题：优先用调用方给定；否则第一页字体+杂行过滤；再兜底首个非杂行长行 / 文件名
        if title is None or not title.strip():
            title = _extract_title_from_page(pages[0]) if pages else ""
        if not title.strip():
            for line in (pages[0]["text"].splitlines() if pages else []):
                line = line.strip()
                if len(line) >= 8 and not _is_junk_title_line(line):
                    title = line
                    break
        title = (title or path.stem).strip()

        paper_id = self._paper_id_for(title)
        old = self.get_paper(paper_id)
        # 内容去重：同 MD5 视为同一篇论文——标题字符串的微小波动不再产生重复条目
        if old is None or (old.md5 and old.md5 != file_md5):
            dup = self._find_by_md5(file_md5)
            if dup and dup.paper_id != paper_id:
                paper_id = dup.paper_id
                title = dup.title  # 沿用已有标题，避免抖动
                old = dup
        existed = old is not None
        now = datetime.now().isoformat(timespec="seconds")
        if existed and old.md5 and old.md5 == file_md5:
            # 内容未变化：跳过向量化；但确保原始 PDF 存档存在（旧版本入库的论文补档）
            STATIC_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
            archive = STATIC_PAPERS_DIR / f"{paper_id}.pdf"
            if not archive.exists():
                shutil.copyfile(path, archive)
            return PaperMeta(
                paper_id=paper_id, title=title, filename=filename,
                pages=old.pages, chunks=old.chunks, indexed_at=old.indexed_at,
                md5=file_md5, skipped=True, existed=True,
            )

        # 原始 PDF 存档（论文预览面板用）
        STATIC_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, STATIC_PAPERS_DIR / f"{paper_id}.pdf")

        # 身份片段：标题 + 第一页开头（通常含摘要）
        first_page = pages[0]["text"][:1200] if pages else ""
        identity_text = f"{title}\n\n{first_page}"

        texts = [identity_text] + [c["text"] for c in chunks]
        vectors = self.embedder.embed(texts)

        points = [
            {
                "id": point_id(paper_id, -1),
                "vector": vectors[0],
                "payload": {
                    "paper_id": paper_id, "title": title, "filename": filename,
                    "page": pages[0]["page"] if pages else 1,
                    "chunk_id": -1, "text": identity_text, "is_identity": True, "md5": file_md5,
                },
            }
        ]
        for c, vec in zip(chunks, vectors[1:]):
            points.append(
                {
                    "id": point_id(paper_id, c["id"]),
                    "vector": vec,
                    "payload": {
                        "paper_id": paper_id, "title": title, "filename": filename,
                        "page": c["page"], "chunk_id": c["id"],
                        "text": c["text"], "is_identity": False, "md5": file_md5,
                    },
                }
            )
        self.store.upsert(points)

        meta = PaperMeta(
            paper_id=paper_id, title=title, filename=filename,
            pages=len(pages), chunks=len(chunks),
            indexed_at=now, md5=file_md5, skipped=False, existed=existed,
        )
        self._upsert_registry(meta)
        return meta

    def delete_paper(self, paper_id: str) -> bool:
        """删除论文：向量点 + 注册表 + 原始 PDF 存档。返回是否 existed。"""
        existed = self.get_paper(paper_id) is not None
        self.store.delete_paper(paper_id)
        (STATIC_PAPERS_DIR / f"{paper_id}.pdf").unlink(missing_ok=True)
        with sqlite3.connect(self.db) as con:
            con.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
        return existed
