"""全库检索器：Qdrant 稠密 + BM25（从向量库 payload 重建），RRF 融合。

BM25 语料在构造时从向量库滚动加载（几千片段秒级）；
论文增删后调用 rebuild() 重建词法索引。
融合键为 paper_id:chunk_id（跨论文去重）。
"""
import re

import jieba
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .vector_store import QdrantStore


def tokenize(text: str) -> list[str]:
    """与单篇模式一致的中英混合分词。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for seg in jieba.cut(text):
        seg = seg.strip()
        if seg and re.search(r"[\u4e00-\u9fff]", seg):
            tokens.append(seg)
    return tokens


def _key(payload: dict) -> str:
    return f"{payload['paper_id']}:{payload['chunk_id']}"


class LibraryRetriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, dense_top_k: int = 8, rrf_k: int = 60):
        self.store = store
        self.embedder = embedder
        self.dense_top_k = dense_top_k
        self.rrf_k = rrf_k
        self.rebuild()

    def rebuild(self) -> None:
        """从向量库 payload 重建 BM25 语料（论文增删后调用）。"""
        points = self.store.scroll_all()
        # 身份片段参与稠密检索（论文身份命中），但不进 BM25（避免标题词噪声主导词法路）
        self.chunks = [p["payload"] for p in points if not p["payload"].get("is_identity")]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks]) if self.chunks else None

    def _rrf_fuse(self, rank_lists: list[list[str]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ranks in rank_lists:
            for pos, key in enumerate(ranks):
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + pos + 1)
        return scores

    def retrieve_multi(self, queries: list[str], top_k: int = 6, paper_id: str | None = None) -> list[dict]:
        """多路查询 × 全库混合检索。paper_id 非空时范围限定单篇。"""
        dense_ranks: list[str] = []
        dense_score: dict[str, float] = {}
        bm25_rank_lists: list[list[str]] = []
        bm25_best: dict[str, float] = {}
        payload_by_key: dict[str, dict] = {}

        for q in queries:
            vec = self.embedder.embed([q])[0]
            points = self.store.search(vec, limit=self.dense_top_k, paper_id=paper_id)
            for p in points:
                key = _key(p["payload"])
                dense_ranks.append(key)
                dense_score[key] = max(dense_score.get(key, 0.0), float(p["score"]))
                payload_by_key[key] = p["payload"]

            if self.bm25 is not None:
                scores = self.bm25.get_scores(tokenize(q))
                ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
                ranks = []
                for i in ranked[: self.dense_top_k]:
                    if scores[i] <= 0:
                        continue
                    c = self.chunks[i]
                    if paper_id and c["paper_id"] != paper_id:
                        continue
                    key = _key(c)
                    ranks.append(key)
                    bm25_best[key] = max(bm25_best.get(key, 0.0), float(scores[i]))
                bm25_rank_lists.append(ranks)

        fused = self._rrf_fuse([dense_ranks, *bm25_rank_lists])
        top = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        hits: list[dict] = []
        for key, rrf in top:
            payload = dict(payload_by_key.get(key, {}))
            if not payload:  # BM25 命中但稠密路未覆盖（极少见）：从内存语料补 payload
                pid, cid = key.split(":")
                payload = next((c for c in self.chunks if c["paper_id"] == pid and str(c["chunk_id"]) == cid), {})
            payload["score"] = float(rrf)
            payload["dense_sim"] = round(dense_score.get(key, 0.0), 4)
            payload["bm25_score"] = round(bm25_best.get(key, 0.0), 3)
            hits.append(payload)
        return hits

    def retrieve(self, query: str, top_k: int = 6, paper_id: str | None = None) -> list[dict]:
        return self.retrieve_multi([query], top_k=top_k, paper_id=paper_id)
