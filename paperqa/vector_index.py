"""稠密向量索引（内存版，numpy 余弦检索）。

单篇论文场景（几百个片段）无需外部向量库，性能完全够用；
接口与 BM25Retriever 对齐，后续多文档/持久化需求出现时平滑升级 Qdrant。
"""
import numpy as np


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        """vectors: (n, dim)，需已 L2 归一化。"""
        self.vectors = vectors

    @classmethod
    def from_embeddings(cls, embeddings: list[list[float]]) -> "DenseIndex":
        mat = np.asarray(embeddings, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return cls(mat / norms)

    def search(self, query_vec: list[float], top_k: int = 8) -> list[tuple[int, float]]:
        """余弦相似度 Top-K，返回 [(chunk 下标, 相似度)]。"""
        if len(self.vectors) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        sims = self.vectors @ q
        top = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top]
