"""存量修复：标题被页眉/版权行污染的论文，从身份片段文本重提取标题并重锚 paper_id。

- 新标题推导：身份文本行 → 杂行过滤 → 首个非杂行长行 + 小写词开头的续行拼接；
  并用 payload 文件名（去下划线）做包含校验；
- paper_id 重锚到标题基（UUID5），点 ID 同步重键；
- 向量复用，不重新嵌入。

用法: python scripts/repair_junk_titles.py
"""
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperqa.config import Settings
from paperqa.vector_store import QdrantStore
from paperqa.library import _is_junk_title_line, REGISTRY_DB
import sqlite3


def _norm(text: str) -> str:
    t = text.lower().replace("_", " ").replace(":", " ")
    return " ".join(t.split())


def _derive_title_from_identity(identity_text: str, filename: str) -> str:
    lines = [ln.strip() for ln in identity_text.splitlines() if ln.strip()]
    fname_norm = _norm(Path(filename).stem if filename.endswith(".pdf") else _norm(filename))
    # 首个非杂行候选 + 可选续行
    for i, ln in enumerate(lines):
        if _is_junk_title_line(ln) or len(ln) < 12:
            continue
        title_parts = [ln]
        # 续行：小写词开头（and/of/for/via/on/in/to...）且非杂行
        for nxt in lines[i + 1 : i + 3]:
            if _is_junk_title_line(nxt) or len(nxt) > 90:
                break
            if re.match(r"^(and|of|for|via|on|in|to|with)\b", nxt.lower()):
                title_parts.append(nxt)
            else:
                break
        candidate = " ".join(title_parts)
        if fname_norm and fname_norm in _norm(candidate):
            return candidate
        # 文件名校验失败也接受候选（文件名可能截断/含版本号）
        if len(candidate) >= 20:
            return candidate
    return ""


def main() -> None:
    store = QdrantStore(Settings())
    points = store.scroll_all()

    identities = [p for p in points if p["payload"].get("is_identity")]
    print(f"全库 {len(points)} 点 / {len(identities)} 篇\n")

    repairs = []
    for ident in identities:
        old_title = ident["payload"].get("title", "")
        if not _is_junk_title_line(old_title):
            continue
        new_title = _derive_title_from_identity(ident["payload"].get("text", ""), ident["payload"].get("filename", ""))
        if not new_title or _norm(new_title) == _norm(old_title):
            print(f"  ⚠️ 无法自动推导新标题: {old_title[:60]}")
            continue
        repairs.append((ident["payload"]["paper_id"], old_title, new_title))

    print(f"待修复: {len(repairs)} 篇")
    for old_pid, old_t, new_t in repairs:
        print(f"  {old_pid[:8]} 《{old_t[:50]}》\n         → 《{new_t[:70]}》")

    for old_pid, old_title, new_title in repairs:
        pts = [p for p in points if p["payload"]["paper_id"] == old_pid]
        norm_pid = new_title.strip().lower()
        new_pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"paperqa:paper:{norm_pid}"))

        new_points, old_ids = [], []
        for pt in pts:
            pay = dict(pt["payload"])
            pay["paper_id"] = new_pid
            pay["title"] = new_title
            if pay.get("is_identity"):
                # 身份片段文本里的旧标题前缀替换为新标题
                rest = pay["text"][len(old_title):].lstrip("\n")
                pay["text"] = f"{new_title}\n\n{rest}"
            cid = pay["chunk_id"]
            new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"paperqa:{new_pid}:chunk:{cid}"))
            new_points.append({"id": new_id, "vector": pt["vector"], "payload": pay})
            old_ids.append(pt["id"])

        for i in range(0, len(new_points), 100):
            store.upsert(new_points[i : i + 100])
        store.client.post(
            f"{store.base}/collections/{store.collection}/points/delete?wait=true",
            json={"points": old_ids},
        )

        con = sqlite3.connect(REGISTRY_DB)
        with con:
            con.execute(
                "UPDATE papers SET paper_id = ?, title = ? WHERE paper_id = ?",
                (new_pid, new_title, old_pid),
            )
        print(f"  ✅ 已修复: 《{new_title[:60]}》 ({len(new_points)} 点重锚)")


if __name__ == "__main__":
    main()
