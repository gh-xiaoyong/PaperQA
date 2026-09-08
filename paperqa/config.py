"""集中读取配置与密钥。

LLM 与 Embedding 均走 OpenAI 兼容协议，可自由切换平台：
- LLM 默认 DeepSeek，可换百炼 Qwen / GLM / Kimi 等
- Embedding 默认阿里云百炼 qwen3.7-text-embedding
"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM（OpenAI 兼容）
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))

    # Embedding（阿里云百炼，OpenAI 兼容）
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding"))

    # Rerank（第二阶段启用）
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "gte-rerank-v2"))

    # 检索（Phase 1 混合检索）
    retrieval_mode: str = field(default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid"))
    dense_top_k: int = field(default_factory=lambda: int(os.getenv("DENSE_TOP_K", "8")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))

    # 查询改写与重排（Phase 2）
    rewrite_enabled: bool = field(default_factory=lambda: os.getenv("REWRITE_ENABLED", "1") == "1")
    rerank_enabled: bool = field(default_factory=lambda: os.getenv("RERANK_ENABLED", "1") == "1")
    rerank_pool: int = field(default_factory=lambda: int(os.getenv("RERANK_POOL", "8")))
    rerank_base_url: str = field(default_factory=lambda: os.getenv("RERANK_BASE_URL", ""))

    # 意图路由与分级 grounding（Phase 2.5）
    intent_routing_enabled: bool = field(default_factory=lambda: os.getenv("INTENT_ROUTING_ENABLED", "1") == "1")
    analytical_top_k: int = field(default_factory=lambda: int(os.getenv("ANALYTICAL_TOP_K", "6")))

    # 生成思考档位（GLM 网关支持 low/high/max；留空=网关默认。分析模式耗时可降档换速度）
    answer_thinking_level: str = field(default_factory=lambda: os.getenv("ANSWER_THINKING_LEVEL", ""))

    # 论文知识库（Qdrant）
    library_mode: bool = field(default_factory=lambda: os.getenv("LIBRARY_MODE", "1") == "1")
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "papers"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))


settings = Settings()
