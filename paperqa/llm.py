"""大模型调用（OpenAI 兼容协议，默认 DeepSeek，可换任意平台）。"""
from collections.abc import Iterator

from openai import OpenAI

from .config import Settings


class LLM:
    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise RuntimeError("缺少 LLM_API_KEY，请在 .env 中配置")
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model

    def chat(self, system: str, user: str, temperature: float = 0.2, extra_body: dict | None = None) -> str:
        kwargs: dict = {"extra_body": extra_body} if extra_body else {}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self, system: str, user: str, temperature: float = 0.2, extra_body: dict | None = None
    ) -> Iterator[tuple[str, str]]:
        """流式生成，yield (kind, delta)。

        kind: "reasoning" 思考内容增量 / "answer" 正式回答增量。
        兼容 reasoning_content（DeepSeek/GLM/Qwen 通用约定）与 reasoning 两种字段。
        """
        kwargs: dict = {"extra_body": extra_body} if extra_body else {}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for chunk in resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning:
                yield ("reasoning", reasoning)
            if delta.content:
                yield ("answer", delta.content)
