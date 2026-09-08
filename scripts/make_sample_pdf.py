"""生成一篇假论文 PDF，用于本地冒烟测试（无需真实论文即可跑通全流程）。"""
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF（新版统一入口名）

SECTIONS = [
    (
        "Retrieval-Augmented Generation for Academic Paper Question Answering",
        "Abstract. We present PaperQA, a retrieval-augmented generation system for academic papers. "
        "It parses PDF documents into semantic chunks of 800 tokens with 10% overlap, retrieves via a hybrid of "
        "dense vectors and BM25 full-text search, reranks candidates with a cross-encoder, and generates "
        "citation-grounded answers. On 30 papers and 240 questions, answer accuracy improves from 62% to 85% "
        "after adding a hallucination constraint to the system prompt.",
    ),
    (
        "1 Introduction",
        "Reading a single academic paper takes about two hours, and more than half of that time is spent locating "
        "key information. We target the single-paper question answering scenario and leave multi-paper comparison "
        "for future work. The core contribution is a three-layer diagnosis framework that attributes retrieval errors "
        "to the retrieval layer, the prompt layer, or the model layer. Experiments show 45% of errors come from the "
        "retrieval layer.",
    ),
    (
        "2 Method",
        "We split each paper by section, then into chunks of at most 800 tokens with a 10% overlap between consecutive "
        "chunks. Chunk sizes of 400, 800 and 1200 tokens were compared; 800 gave the best retrieval recall. For "
        "retrieval we combine a bge-m3 dense vector index with BM25 full-text search, and apply query rewriting plus "
        "top-K reranking with bge-reranker-v2-m3. Every answer must cite the source page.",
    ),
    (
        "3 Results",
        "After rewriting the system prompt to require answers grounded only in the provided context and to state "
        "explicitly when information is missing, answer accuracy rises from 62% to 85%. In a trial with 15 students, "
        "the average time to locate key information drops from 40 minutes to 10 minutes.",
    ),
    (
        "4 Conclusion",
        "PaperQA demonstrates that citation-grounded generation and a strong hallucination constraint are the two "
        "most important factors for trustworthy academic question answering.",
    ),
]


def build(path: Path) -> None:
    doc = fitz.open()
    for heading, body in SECTIONS:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), f"{heading}\n\n{body}", fontsize=11)
    doc.save(str(path))
    doc.close()
    print(f"已生成示例论文: {path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/paperqa_demo.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out)
