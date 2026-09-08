"""一次性修复：把「临时文件名时代」入库的论文重锚到标题基 paper_id。

复用已有向量（不重新嵌入）：滚动全库 → 按旧 paper_id 分组 → 以 payload 标题
计算新 paper_id → 重写点 ID 与 payload → upsert 新点 → 删除旧点 → 重建注册表。
同名标题的多组旧点视为同一篇论文，仅保留第一组（后续被同标题重传覆盖自愈）。

用法: python scripts/repair_library_ids.py
"""
import sys
import uuid
from datetime import datetime
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperqa.config import Settings
from paperqa.vector_store import QdrantStore
from paperqa.library import REGISTRY_DB


def main() -> None:
    store = QdrantStore(Settings())
    points = store.scroll_all()
    print(f"滚动到 {len(points)} 个点")

    groups: dict[str, list[dict]] = {}
    for p in points:
        groups.setdefault(p["payload"]["paper_id"], []).append(p)
    print(f"旧 paper_id 分组: {len(groups)} 篇")

    new_points: list[dict] = []
    old_ids: list[str] = []
    rows: list[tuple] = []
    seen_new_ids: set[str] = set()
    merged = 0

    for old_pid, pts in groups.items():
        title = (pts[0]["payload"].get("title") or "").strip()
        if not title:
            print(f"  ⚠️ 跳过无标题组 {old_pid[:8]}")
            continue
        norm = title.lower()
        new_pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"paperqa:paper:{norm}"))
        if new_pid in seen_new_ids:
            merged += 1  # 同标题重复组：保留先到组
            continue
        seen_new_ids.add(new_pid)

        pages = max(pt["payload"]["page"] for pt in pts)
        n_chunks = sum(1 for pt in pts if not pt["payload"].get("is_identity"))
        md5 = pts[0]["payload"].get("md5") or ""

        for pt in pts:
            pay = dict(pt["payload"])
            pay["paper_id"] = new_pid
            cid = pay["chunk_id"]
            new_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"paperqa:{new_pid}:chunk:{cid}"))
            new_points.append({"id": new_id, "vector": pt["vector"], "payload": pay})
            old_ids.append(pt["id"])
        rows.append(
            (new_pid, title, pts[0]["payload"].get("filename", ""), pages, n_chunks,
             datetime.now().isoformat(timespec="seconds"), md5)
        )

    for i in range(0, len(new_points), 100):
        store.upsert(new_points[i : i + 100])
    if old_ids:
        store.client.post(
            f"{store.base}/collections/{store.collection}/points/delete?wait=true",
            json={"points": old_ids},
        )

    con = sqlite3.connect(REGISTRY_DB)
    with con:
        con.execute("DELETE FROM papers")
        con.executemany(
            "INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?)", rows
        )
    print(f"修复完成: {len(rows)} 篇注册 / {len(new_points)} 点重锚 / {merged} 组同标题合并 / {len(old_ids)} 旧点删除")


if __name__ == "__main__":
    main()
