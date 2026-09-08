"""端到端编排。

两种模式：
- library=True（默认，论文知识库）：多论文持久化。Qdrant 稠密 + BM25（payload 重建）+ RRF + 重排，
  引用升级到论文级 【《标题》 第X页 / 片段N】；
- library=False（单篇会话模式）：进程内索引，评测与冒烟测试使用。
"""
from collections.abc import Iterator
from dataclasses import dataclass

from .chunker import chunk_pages
from .config import Settings
from .embeddings import Embedder
from .llm import LLM
from .library import PaperLibrary
from .library_retriever import LibraryRetriever
from .pdf_parser import parse_pdf
from .prompts import ANALYTICAL_SYSTEM, GROUNDED_QA_TEMPLATE, GROUNDED_SYSTEM
from .query_rewriter import QueryRewriter
from .reranker import Reranker
from .retriever import BM25Retriever, HybridRetriever


@dataclass
class Retrieved:
    """检索结果 + 意图路由元信息。

    library 模式下 hits 为 payload 字典（paper_id/title/page/chunk_id/text/...）；
    单篇模式下为 chunk 字典（id/page/text/...）。
    """

    hits: list[dict]
    intent: str  # "extractive"（提取型，严格摘抄） | "analytical"（推理型，允许有据推断）
    queries: list[str]  # 实际使用的查询（首个为原始问题）


class Pipeline:
    def __init__(self, settings: Settings | None = None, library: bool | None = None):
        self.settings = settings or Settings()
        self.library_mode = self.settings.library_mode if library is None else library
        self.llm = LLM(self.settings)
        self.chunks: list[dict] = []
        self.retriever: BM25Retriever | HybridRetriever | None = None
        # 查询改写 + 重排（构造不发网络请求，失败在调用时降级）
        self.query_rewriter = QueryRewriter(self.llm) if self.settings.rewrite_enabled else None
        self.reranker = Reranker(self.settings) if (self.settings.rerank_enabled and self.settings.embedding_api_key) else None
        # 知识库
        self.library: PaperLibrary | None = None
        self._library_retriever: LibraryRetriever | None = None
        if self.library_mode:
            self.library = PaperLibrary(self.settings)

    # ---------- 知识库模式 ----------

    def index_paper(self, pdf_path: str, title: str | None = None, filename: str | None = None) -> dict:
        """把一篇论文加入知识库（解析→分块→身份片段→向量化→入库）。

        filename 传原始上传文件名（否则用临时路径名，破坏查重）。
        """
        if not self.library_mode:
            raise RuntimeError("index_paper 仅在知识库模式（LIBRARY_MODE=1）可用")
        meta = self.library.index_paper(pdf_path, title=title, filename=filename)
        self._library_retriever = None  # 库变更后强制重建 BM25 语料
        return {
            "paper_id": meta.paper_id, "title": meta.title, "pages": meta.pages,
            "chunks": meta.chunks, "skipped": meta.skipped, "existed": meta.existed,
        }

    def list_papers(self) -> list:
        if not self.library_mode:
            raise RuntimeError("list_papers 仅在知识库模式可用")
        return self.library.list_papers()

    def indexed_papers(self) -> dict[str, str]:
        """{filename: md5} 查重映射：同名同 hash 跳过；同名不同 hash 覆盖更新。"""
        if not self.library_mode:
            raise RuntimeError("indexed_papers 仅在知识库模式可用")
        return {p.filename: p.md5 for p in self.list_papers()}

    def delete_paper(self, paper_id: str) -> bool:
        if not self.library_mode:
            raise RuntimeError("delete_paper 仅在知识库模式可用")
        ok = self.library.delete_paper(paper_id)
        self._library_retriever = None
        return ok

    def _get_library_retriever(self) -> LibraryRetriever:
        if self._library_retriever is None:
            self._library_retriever = LibraryRetriever(
                self.library.store,
                Embedder(self.settings),
                dense_top_k=self.settings.dense_top_k,
                rrf_k=self.settings.rrf_k,
            )
        return self._library_retriever

    # ---------- 单篇会话模式 ----------

    def index_pdf(self, pdf_path: str, max_tokens: int = 800, overlap_tokens: int = 80) -> int:
        """解析并索引一个 PDF（单篇会话模式），返回片段数量。

        检索模式由 RETRIEVAL_MODE 决定：hybrid（默认）或 bm25；
        hybrid 但未配置 EMBEDDING_API_KEY 时自动降级为 bm25。
        """
        if self.library_mode:
            raise RuntimeError("知识库模式请使用 index_paper()")
        pages = parse_pdf(pdf_path)
        self.chunks = chunk_pages(pages, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        mode = (self.settings.retrieval_mode or "hybrid").strip().lower()
        if mode == "hybrid" and self.settings.embedding_api_key:
            self.retriever = HybridRetriever(
                self.chunks,
                Embedder(self.settings),
                dense_top_k=self.settings.dense_top_k,
                rrf_k=self.settings.rrf_k,
            )
        else:
            self.retriever = BM25Retriever(self.chunks)
        return len(self.chunks)

    # ---------- 检索（两种模式共用编排）----------

    def retrieve(self, question: str, top_k: int = 4, paper_id: str | None = None) -> Retrieved:
        """检索 + 意图路由：改写/分类 → 按意图宽召回 → 重排（各环节失败自动降级）。

        library 模式下 paper_id 非空时范围限定单篇，否则全库检索。
        """
        if not self.library_mode and self.retriever is None:
            raise RuntimeError("尚未索引任何 PDF，请先调用 index_pdf")

        intent, queries = "extractive", [question]
        if self.settings.intent_routing_enabled and self.query_rewriter is not None:
            rr = self.query_rewriter.rewrite(question)
            intent, queries = rr.intent, rr.queries
        effective_top_k = self.settings.analytical_top_k if intent == "analytical" else top_k
        pool = max(effective_top_k, self.settings.rerank_pool)

        if self.library_mode:
            hits = self._get_library_retriever().retrieve_multi(queries, top_k=pool, paper_id=paper_id)
        elif isinstance(self.retriever, HybridRetriever):
            hits = self.retriever.retrieve_multi(queries, top_k=pool)
        elif self.retriever is not None:
            hits = self.retriever.retrieve(question, top_k=pool)
        else:
            raise RuntimeError("检索器未初始化")

        if self.reranker is not None and len(hits) > 1:
            hits = self.reranker.rerank(question, hits, top_n=effective_top_k)
        else:
            hits = hits[:effective_top_k]
        return Retrieved(hits=hits, intent=intent, queries=queries)

    @staticmethod
    def build_context(hits: list[dict]) -> str:
        blocks = []
        for h in hits:
            if "title" in h:  # 知识库 hit：论文级引用
                blocks.append(f"【《{h['title']}》 第{h['page']}页 / 片段{h['chunk_id']}】\n{h['text']}")
            else:  # 单篇 hit
                blocks.append(f"[片段{h['id']} / 第{h['page']}页] {h['text']}")
        return "\n\n".join(blocks)

    def answer_stream(
        self, question: str, hits: list[dict], intent: str = "extractive", temperature: float = 0.2
    ) -> Iterator[tuple[str, str]]:
        """基于检索结果流式生成，yield (kind, delta)：kind ∈ {"reasoning", "answer"}。

        intent=analytical 时使用分析模式 Prompt（允许有据推断的分级 grounding）。
        ANSWER_THINKING_LEVEL 设置时透传思考档位。
        """
        system = ANALYTICAL_SYSTEM if intent == "analytical" else GROUNDED_SYSTEM
        user = GROUNDED_QA_TEMPLATE.format(context=self.build_context(hits), question=question)
        extra_body = (
            {"thinking": {"type": "enabled", "level": self.settings.answer_thinking_level}}
            if self.settings.answer_thinking_level
            else None
        )
        yield from self.llm.chat_stream(system, user, temperature=temperature, extra_body=extra_body)

    def ask(self, question: str, top_k: int = 4, paper_id: str | None = None) -> dict:
        """一次性问答（冒烟测试等场景用），返回 {answer, sources, intent}。"""
        r = self.retrieve(question, top_k=top_k, paper_id=paper_id)
        answer = "".join(
            delta for kind, delta in self.answer_stream(question, r.hits, intent=r.intent) if kind == "answer"
        )
        return {"answer": answer, "sources": r.hits, "intent": r.intent}
