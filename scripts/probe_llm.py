"""LLM 连通性探测：用当前 .env 配置真实调用一次。

用法: python scripts/probe_llm.py
成功 → 打印模型回复；失败 → 打印错误类型与摘要（不显示 key）。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI

from paperqa.config import Settings


def main() -> None:
    s = Settings()
    print("base_url:", s.llm_base_url)
    print("model   :", s.llm_model)
    c = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    try:
        r = c.chat.completions.create(
            model=s.llm_model,
            messages=[{"role": "user", "content": "只回复两个字：连通"}],
            max_tokens=10,
        )
        print("✅ LLM 连通 OK, 回复:", r.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM 连通失败: {type(e).__name__}")
        print(str(e)[:400])


if __name__ == "__main__":
    main()
