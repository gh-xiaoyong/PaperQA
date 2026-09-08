"""重排（Cross-Encoder）：DashScope gte-rerank 原生 HTTP API。

DashScope 的 rerank 不在 OpenAI 兼容协议内，走原生 REST；
对调用失败做完全降级：直接返回融合排序的前 top_n，不阻断问答。
"""
import httpx

from .config import Settings


class Reranker:
    def __init__(self, settings: Settings):
        self.model = settings.rerank_model
        # rerank 与 embedding 同属百炼，共用 key
        self.api_key = settings.embedding_api_key
        self.base = (settings.rerank_base_url or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")

    def rerank(self, query: str, hits: list[dict], top_n: int = 4) -> list[dict]:
        """对候选片段按与 query 的相关度重排，返回前 top_n。"""
        if len(hits) <= 1:
            return hits[:top_n]
        try:
            resp = httpx.post(
                f"{self.base}/services/rerank/text-rerank/text-rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "input": {"query": query, "documents": [h["text"] for h in hits]},
                    "parameters": {"return_documents": False, "top_n": top_n},
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json()["output"]["results"]
            out: list[dict] = []
            for r in results:
                h = dict(hits[r["index"]])
                h["rerank_score"] = round(float(r["relevance_score"]), 4)
                out.append(h)
            return out[:top_n]  # 部分网关不严格执行 top_n，这里硬切片兜底
        except Exception:
            return hits[:top_n]  # 降级：保持融合排序
