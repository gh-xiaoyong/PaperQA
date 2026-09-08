"""命令行冒烟测试：对示例论文跑一遍全流程。"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from paperqa.config import Settings
from paperqa.pipeline import Pipeline

QUESTIONS = [
    "这篇论文的核心贡献是什么？",
    "What is the chunk size used in this paper?",  # 英文问英文文档，BM25 应命中
    "How does the retrieval method work?",
    "准确率提升到了多少？",
    "这篇论文用了 Adam 优化器吗？",  # 无答案型（中文）
    "Does the paper use the Adam optimizer?",  # 无答案型（英文）
    "这篇论文有哪些局限性？",  # 推理型 → 应路由 analytical
    "这篇论文的实验设计有什么不足？",  # 推理型 → 应路由 analytical
]


def retrieval_only(pdf: Path) -> None:
    """BM25 基线 vs 混合检索对比（重点观察中文问英文文档的召回差异）。"""
    from paperqa.chunker import chunk_pages
    from paperqa.pdf_parser import parse_pdf
    from paperqa.retriever import BM25Retriever

    pages = parse_pdf(str(pdf))
    chunks = chunk_pages(pages)
    print(f"解析 {len(pages)} 页 → {len(chunks)} 个片段\n" + "=" * 70)
    bm25 = BM25Retriever(chunks)

    hybrid = None
    try:
        p = Pipeline(Settings(), library=False)
        p.index_pdf(str(pdf))
        hybrid = p.retriever
        print(f"混合检索已就绪: {type(hybrid).__name__}")
    except Exception as e:
        print(f"混合检索不可用（{type(e).__name__}: {e}），仅显示 BM25 基线")

    for q in QUESTIONS:
        print(f"\nQ: {q}")
        b = bm25.retrieve(q, top_k=2)
        print("  BM25  :", [(h["page"], round(h["score"], 2)) for h in b])
        if hybrid is not None:
            if p.query_rewriter is not None:
                rr = p.query_rewriter.rewrite(q)
                print("  改写  :", rr.queries, f"(intent={rr.intent})")
            r = p.retrieve(q, top_k=2)  # 完整链路：意图路由 → 多路 RRF → 重排
            h = r.hits
            extra = f"rerank={h[0]['rerank_score']:.3f}" if h and "rerank_score" in h[0] else "rerank未生效"
            print("  Hybrid:", [(x["page"], f"dense={x['dense_sim']:.3f}") for x in h], f"({extra})")


def main() -> None:
    pdf = Path("data/sample/paperqa_demo.pdf")
    if not pdf.exists():
        print("先运行: python scripts/make_sample_pdf.py")
        return
    if "--retrieval-only" in sys.argv:
        retrieval_only(pdf)
        return
    p = Pipeline(Settings(), library=False)
    n = p.index_pdf(str(pdf))
    print(f"已索引 {n} 个片段\n" + "=" * 60)
    for q in QUESTIONS:
        r = p.ask(q)
        print(f"\nQ: {q} [intent={r.get('intent', '?')}]")
        print(f"A: {r['answer']}")
        print("来源:", [(c["page"], round(c["score"], 3)) for c in r["sources"]])


if __name__ == "__main__":
    if "--retrieval-only" in sys.argv:
        retrieval_only(Path("data/sample/paperqa_demo.pdf"))
    else:
        main()
