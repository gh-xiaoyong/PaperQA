"""会话历史持久化：token 标识 + JSON 文件存储（data/history/{token}.json）。

- token 由 secrets 生成（16 位十六进制），通过 URL 参数 ?t=... 传递，
  刷新页面/收藏链接/分享链接均可恢复会话；
- 内容为完整消息（角色/内容/引用来源/思考过程/意图）与检索范围；
- token 仅允许 16 位十六进制，防止路径穿越。
"""
import json
import re
import secrets
from datetime import datetime
from pathlib import Path

HISTORY_DIR = Path("data/history")
TOKEN_RE = re.compile(r"^[0-9a-f]{16}$")


def new_token() -> str:
    return secrets.token_hex(8)


def valid_token(token: str) -> bool:
    return bool(TOKEN_RE.fullmatch(token or ""))


def _path(token: str) -> Path:
    return HISTORY_DIR / f"{token}.json"


def save_conversation(token: str, messages: list[dict], scope: str | None = None) -> None:
    if not valid_token(token) or not messages:
        return
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    title = next((m["content"][:40] for m in messages if m["role"] == "user"), "新对话")
    data = {
        "token": token,
        "title": title,
        "scope": scope,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }
    _path(token).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_conversation(token: str) -> dict | None:
    if not valid_token(token):
        return None
    p = _path(token)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_conversation(token: str) -> None:
    if valid_token(token):
        _path(token).unlink(missing_ok=True)


def list_conversations() -> list[dict]:
    """按更新时间倒序的会话摘要列表 [{token, title, updated_at}]。"""
    if not HISTORY_DIR.exists():
        return []
    out = []
    for p in HISTORY_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"token": d["token"], "title": d.get("title", "新对话"), "updated_at": d.get("updated_at", "")})
        except Exception:
            continue
    return sorted(out, key=lambda x: x["updated_at"], reverse=True)
