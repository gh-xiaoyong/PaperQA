"""生成第二篇假论文（KT 知识追踪主题），用于验证跨论文检索。

用法: python scripts/make_kt_paper.py [输出路径]
"""
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF（新版统一入口名）

SECTIONS = [
    (
        "Improving Knowledge Tracing with Contrastive Learning on Interaction Sequences",
        "Abstract. Knowledge tracing (KT) models predict students' future interactions from their answering "
        "histories. We propose CLKT, an improved KT model that enhances the standard self-attention knowledge "
        "tracer with a supervised contrastive objective over interaction sequences. On three public datasets "
        "(ASSISTments, EdNet, XES3G5M), CLKT improves AUC by 2.1 points on average over the self-attention "
        "baseline while using the same input features.",
    ),
    (
        "1 Introduction",
        "Knowledge tracing is the task of modeling a student's evolving knowledge state. The self-attention "
        "knowledge tracer treats interactions as a sequence but ignores the similarity structure between "
        "students' practice patterns. We argue that contrasting augmented views of the same interaction "
        "sequence yields better representations. Unlike prior KT work that adds external side information, "
        "our improvement requires no extra features and applies to any attention-based KT backbone.",
    ),
    (
        "2 Method",
        "CLKT builds on the self-attention knowledge tracer. We construct two augmented views of each student's "
        "interaction sequence by masking and cropping, encode both views, and pull their representations "
        "together while pushing apart representations of different students (supervised contrastive loss). "
        "The contrastive loss is combined with the standard prediction loss with weight 0.5. No additional "
        "input features such as question difficulty or student metadata are required.",
    ),
    (
        "3 Results",
        "On ASSISTments, CLKT reaches AUC 0.79 versus 0.77 for the self-attention baseline; on EdNet 0.74 "
        "versus 0.72; on XES3G5M 0.80 versus 0.78. The improvement is consistent across all three datasets. "
        "Ablation shows removing the supervised contrastive loss drops AUC back to baseline, confirming the "
        "gain comes from the contrastive objective. Training cost increases by only 15%.",
    ),
    (
        "4 Conclusion",
        "A supervised contrastive objective over interaction sequences is a simple and effective way to improve "
        "attention-based knowledge tracing. Future work includes extending the contrastive scheme to graph-based "
        "KT models and studying cold-start students.",
    ),
]


def build(path: Path) -> None:
    doc = fitz.open()
    for heading, body in SECTIONS:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), f"{heading}\n\n{body}", fontsize=11)
    doc.save(str(path))
    doc.close()
    print(f"已生成 KT 主题论文: {path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/kt_contrastive_paper.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out)
