"""合并重复条目：同 MD5（同内容）只保留一条，其余删除（含向量点与存档）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperqa.config import Settings
from paperqa.library import STATIC_PAPERS_DIR
from paperqa.pipeline import Pipeline

p = Pipeline(Settings(), library=True)
papers = p.list_papers()

by_md5: dict[str, list] = {}
for pp in papers:
    if pp.md5:
        by_md5.setdefault(pp.md5, []).append(pp)

for md5, grp in by_md5.items():
    if len(grp) <= 1:
        continue
    # 保留有存档的那条；都有则保留标题更完整（更长）的
    grp.sort(key=lambda pp: (not (STATIC_PAPERS_DIR / f"{pp.paper_id}.pdf").exists(), -len(pp.title)))
    keep, dups = grp[0], grp[1:]
    print(f"同内容 {len(grp)} 条 → 保留 [{keep.paper_id[:8]}] 《{keep.title[:50]}》")
    for d in dups:
        print(f"  删除重复 [{d.paper_id[:8]}] 《{d.title[:50]}》")
        p.delete_paper(d.paper_id)

print(f"\n合并后: {len(p.list_papers())} 篇")
