"""PDF 解析。

Tracer 阶段用 PyMuPDF（快、零配置）；Day 1-3 升级为 MinerU/marker
以正确处理学术论文的双栏排版与公式。
"""
import pymupdf as fitz  # PyMuPDF（新版统一入口名，避免 fitz 弃用警告）


def parse_pdf(path: str) -> list[dict]:
    """解析 PDF，返回 [{page: int, text: str}]。"""
    pages: list[dict] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": i + 1, "text": text})
    return pages
