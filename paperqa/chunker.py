"""语义分块：按段落聚合，控制 token 上限与重叠。

简历对标：800 Token 分块 + 10% 重叠（经 400/800/1200 对比实验确定）。
"""
import re


def _approx_tokens(text: str) -> int:
    """粗估 token 数：英文约 4 字符/token，中文约 1.5 字符/token。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_pages(pages: list[dict], max_tokens: int = 800, overlap_tokens: int = 80) -> list[dict]:
    """把每页文本切成带页码元数据的片段。返回 [{text, page, id}]。"""
    chunks: list[dict] = []
    chunk_id = 0
    for page in pages:
        paragraphs = _split_paragraphs(page["text"])
        cur: list[str] = []
        cur_tokens = 0
        for para in paragraphs:
            t = _approx_tokens(para)
            if cur and cur_tokens + t > max_tokens:
                chunks.append(_emit(cur, page["page"], chunk_id))
                chunk_id += 1
                cur, cur_tokens = _carry_overlap(cur, overlap_tokens)
            cur.append(para)
            cur_tokens += t
        if cur:
            chunks.append(_emit(cur, page["page"], chunk_id))
            chunk_id += 1
    return chunks


def _carry_overlap(cur: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    overlap: list[str] = []
    ot = 0
    for p in reversed(cur):
        pt = _approx_tokens(p)
        if ot + pt > overlap_tokens:
            break
        overlap.insert(0, p)
        ot += pt
    return overlap, ot


def _emit(paragraphs: list[str], page: int, chunk_id: int) -> dict:
    return {"text": "\n\n".join(paragraphs), "page": page, "id": chunk_id}
