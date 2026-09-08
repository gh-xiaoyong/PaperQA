"""检查 .env 结构与常见问题（只显示变量名与摘要，绝不显示值）。

用法: python scripts/env_check.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"

REQUIRED = [
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
]


def main() -> None:
    if not ENV.exists():
        print("❌ .env 不存在，先: copy .env.example .env")
        return
    raw = ENV.read_bytes()
    print(f"文件字节数: {len(raw)} | 前4字节: {raw[:4].hex()} | 含NUL: {b'\x00' in raw}")
    text = raw.decode("utf-8", errors="replace")
    vars_seen: dict[str, int] = {}
    print("--- 逐行结构（不显示值）---")
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            print(f"{i:2d} [空行]")
            continue
        if s.startswith("#"):
            print(f"{i:2d} [注释] {s[:40]}")
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            flags = []
            if not v:
                flags.append("⚠️ 空值")
            if not v.isascii():
                flags.append("⚠️ 含非ASCII(占位符?)")
            if "maas.aliyuncs.com" in v:
                flags.append("→百炼地域端点")
            if "dashscope" in v:
                flags.append("→dashscope")
            if "deepseek" in v:
                flags.append("→deepseek")
            if v.startswith("sk-") and v.isascii() and len(v) >= 30:
                flags.append("✓像真实key")
            vars_seen[k] = len(v)
            print(f"{i:2d} [变量] {k:20s} len={len(v):<4d} {' '.join(flags)}")
        else:
            print(f"{i:2d} [⚠️异常行] {s[:40]}")
    print("--- 必需变量检查 ---")
    for k in REQUIRED:
        print(f"{k:22s} {'✓' if k in vars_seen else '❌ 缺失'}")


if __name__ == "__main__":
    main()
