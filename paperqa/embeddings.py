"""Embedding 调用（阿里云百炼，OpenAI 兼容协议）。

第二阶段混合检索时接入 pipeline；当前 tracer bullet 用 BM25，本模块先行就位。
"""
from openai import OpenAI

from .config import Settings


class Embedder:
    def __init__(self, settings: Settings, batch_size: int = 10):
        if not settings.embedding_api_key:
            raise RuntimeError("缺少 EMBEDDING_API_KEY，请在 .env 中配置")
        self.client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
        )
        self.model = settings.embedding_model
        # 百炼向量接口单次输入条数有限制，分批请求
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，返回与输入顺序一致的向量列表。"""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            data = sorted(resp.data, key=lambda d: d.index)
            out.extend([d.embedding for d in data])
        return out
