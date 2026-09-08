"""论文库存储层验证：索引示例论文 → 列表 → 语义检索（含跨论文式提问）。

用法: python scripts/library_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperqa.config import Settings
from paperqa.library import PaperLibrary


def main() -> None:
    lib = PaperLibrary(Settings())

    meta = lib.index_paper("data/sample/paperqa_demo.pdf", title="PaperQA Demo: RAG for Academic QA")
    print(f"索引完成: {meta.title} ({meta.pages}页 / {meta.chunks}片段)")

    print("\n=== 论文库列表 ===")
    for p in lib.list_papers():
        print(f"  [{p.paper_id[:8]}] {p.title} ({p.pages}页 / {p.chunks}片段, {p.indexed_at})")
    print(f"库内向量总数: {lib.store.count()}")

    print("\n=== 语义检索测试 ===")
    queries = [
        "分块大小是多少？",
        "哪篇论文提出了三层诊断框架？",  # 跨论文式提问：应命中身份片段
    ]
    for q in queries:
        vec = lib.embedder.embed([q])[0]
        points = lib.store.search(vec, limit=3)
        print(f"\nQ: {q}")
        for p in points:
            pay = p["payload"]
            tag = "身份片段" if pay.get("is_identity") else f"第{pay['page']}页/片段{pay['chunk_id']}"
            print(f"  score={p['score']:.3f} 《{pay['title'][:44]}》 [{tag}]")
            print(f"    {pay['text'][:76]}…")


if __name__ == "__main__":
    main()
