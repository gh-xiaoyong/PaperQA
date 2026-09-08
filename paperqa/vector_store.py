"""Qdrant 向量库封装（REST via httpx，零额外依赖）。

多论文持久化存储层：集合初始化 / 批量 upsert / 语义检索 / 按论文过滤与删除 / 全量滚动。
点 ID 用 UUID5(paper_id, chunk_id) 确定性生成——重复索引同一篇自动覆盖（幂等去重）。
"""
import uuid

import httpx

from .config import Settings


def point_id(paper_id: str, chunk_id: int) -> str:
    """确定性点 ID：同一篇论文重复索引时自动覆盖旧向量。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"paperqa:{paper_id}:chunk:{chunk_id}"))


class QdrantStore:
    def __init__(self, settings: Settings):
        self.base = settings.qdrant_url.rstrip("/")
        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim
        self.client = httpx.Client(timeout=60)

    # ---------- 集合管理 ----------

    def ensure_collection(self) -> None:
        r = self.client.get(f"{self.base}/collections/{self.collection}")
        if r.status_code == 200:
            return
        r = self.client.put(
            f"{self.base}/collections/{self.collection}",
            json={"vectors": {"size": self.dim, "distance": "Cosine"}},
        )
        r.raise_for_status()

    def count(self) -> int:
        r = self.client.get(f"{self.base}/collections/{self.collection}")
        if r.status_code != 200:
            return 0
        return r.json()["result"].get("points_count") or 0

    # ---------- 写入 / 删除 ----------

    def upsert(self, points: list[dict]) -> None:
        """points: [{id, vector, payload}]"""
        r = self.client.put(
            f"{self.base}/collections/{self.collection}/points?wait=true",
            json={"points": points},
        )
        r.raise_for_status()

    def delete_paper(self, paper_id: str) -> None:
        r = self.client.post(
            f"{self.base}/collections/{self.collection}/points/delete?wait=true",
            json={"filter": {"must": [{"key": "paper_id", "match": {"value": paper_id}}]}},
        )
        r.raise_for_status()

    # ---------- 检索 / 滚动 ----------

    def search(self, vector: list[float], limit: int = 8, paper_id: str | None = None) -> list[dict]:
        """语义检索，返回 [{id, score, payload}]，可选按论文过滤。"""
        body: dict = {"query": vector, "limit": limit, "with_payload": True}
        if paper_id:
            body["filter"] = {"must": [{"key": "paper_id", "match": {"value": paper_id}}]}
        r = self.client.post(f"{self.base}/collections/{self.collection}/points/query", json=body)
        r.raise_for_status()
        return r.json()["result"]["points"]

    def scroll_all(self, paper_id: str | None = None, max_points: int = 20000) -> list[dict]:
        """滚动取点（后续从 payload 重建 BM25 语料用）。with_vector 供重锚迁移复用。"""
        body: dict = {"limit": 256, "with_payload": True, "with_vector": True}
        if paper_id:
            body["filter"] = {"must": [{"key": "paper_id", "match": {"value": paper_id}}]}
        points: list[dict] = []
        offset = None
        while len(points) < max_points:
            if offset is not None:
                body["offset"] = offset
            r = self.client.post(f"{self.base}/collections/{self.collection}/points/scroll", json=body)
            r.raise_for_status()
            data = r.json()["result"]
            points.extend(data["points"])
            offset = data.get("next_page_offset")
            if offset is None:
                break
        return points
