"""检索模块。

- BM25Retriever：纯词法基线（零 API、零显卡）。
- HybridRetriever：稠密向量 + BM25，RRF 融合（Phase 1 主力）。
  稠密路解决跨语言/语义召回，词法路兜底精确术语（公式、缩写、数字）。
"""
import re

import jieba
from rank_bm25 import BM25Okapi

from .vector_index import DenseIndex


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文按词/数字切分，中文用 jieba。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    for seg in jieba.cut(text):
        seg = seg.strip()
        if seg and re.search(r"[\u4e00-\u9fff]", seg):
            tokens.append(seg)
    return tokens


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])

        hits: list[dict] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            c = dict(self.chunks[i])
            c["score"] = float(scores[i])
            hits.append(c)

        # 若全部得分为 0，退回取分数最高的前几个，保证始终有上下文可答
        if not hits:
            for i in ranked[:top_k]:
                c = dict(self.chunks[i])
                c["score"] = float(scores[i])
                hits.append(c)
        return hits


class HybridRetriever:
    """混合检索：稠密向量 + BM25，RRF（Reciprocal Rank Fusion）融合。

    embedder 只要求鸭子类型：提供 .embed(list[str]) -> list[list[float]] 与 .batch_size。
    BM25 仅在得分 > 0 时参与融合（纯噪声排序不注入 RRF）。
    """

    def __init__(self, chunks: list[dict], embedder, dense_top_k: int = 8, rrf_k: int = 60):
        self.chunks = chunks
        self.embedder = embedder
        self.dense_top_k = dense_top_k
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

        vectors: list[list[float]] = []
        for i in range(0, len(chunks), embedder.batch_size):
            batch = [c["text"] for c in chunks[i : i + embedder.batch_size]]
            vectors.extend(embedder.embed(batch))
        self.dense = DenseIndex.from_embeddings(vectors)

    def _rrf_fuse(self, rank_lists: list[list[int]]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranks in rank_lists:
            for pos, idx in enumerate(ranks):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + pos + 1)
        return scores

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        return self.retrieve_multi([query], top_k=top_k)

    def retrieve_multi(self, queries: list[str], top_k: int = 4) -> list[dict]:
        """多路查询检索：每条查询各出稠密+词法排名，全部进 RRF 融合。

        Phase 2 核心：配合查询改写，中文原查询 + 英文变体共同召回。
        """
        dense_ranks: list[int] = []
        dense_sim: dict[int, float] = {}
        bm25_rank_lists: list[list[int]] = []
        bm25_best: dict[int, float] = {}

        for q in queries:
            vec = self.embedder.embed([q])[0]
            dhits = self.dense.search(vec, top_k=self.dense_top_k)
            dense_ranks.extend(i for i, _ in dhits)
            for i, s in dhits:
                dense_sim[i] = max(dense_sim.get(i, 0.0), s)

            scores = self.bm25.get_scores(tokenize(q))
            ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
            bm25_rank_lists.append([i for i in ranked[: self.dense_top_k] if scores[i] > 0])
            for i in ranked[: self.dense_top_k]:
                if scores[i] > bm25_best.get(i, 0.0):
                    bm25_best[i] = float(scores[i])

        fused = self._rrf_fuse([dense_ranks, *bm25_rank_lists])
        top = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

        hits: list[dict] = []
        for idx, rrf in top:
            c = dict(self.chunks[idx])
            c["score"] = float(rrf)
            c["dense_sim"] = round(float(dense_sim.get(idx, 0.0)), 4)
            c["bm25_score"] = round(float(bm25_best.get(idx, 0.0)), 3)
            hits.append(c)
        return hits
