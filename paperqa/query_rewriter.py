"""查询改写 + 意图分类：一次 LLM 调用完成两件事。

1. intent 分类：extractive（提取型，答案在原文某处）| analytical（推理型，需跨片段归纳）；
2. queries 改写：口语化 → 检索友好；中文问题追加英文变体；
   推理型追加「信号扩展」变体（论文实际会出现的表述：limitation/future work/scope 等）。

任何失败都优雅降级：intent=extractive + 原始查询，不阻断问答。
"""
import json
import re
from dataclasses import dataclass

from .llm import LLM

REWRITE_SYSTEM = (
    "你是学术论文问答系统的检索前置模块。对用户提问完成两件事：\n"
    "\n"
    "一、判断问题类型（intent）：\n"
    '- "extractive"（提取型）：答案可在原文某处直接找到——事实、数字、方法、结果等；\n'
    '- "analytical"（推理型）：需要跨片段综合归纳或评价——如局限性、优点缺点、意义价值、'
    "为什么有效、改进建议。\n"
    '- 无法确定时一律判 "extractive"。\n'
    "\n"
    "二、改写出 1-3 条更适合检索的查询（queries）：\n"
    "1. 第一条保留原意，把口语化表达补全为书面检索式；\n"
    "2. 若问题是中文而论文很可能是英文，追加一条地道的英文版本；\n"
    "3. 若为 analytical，追加 1-2 条「信号扩展」变体：用论文里实际会出现的表述来检索，\n"
    "   如 limitation / future work / scope / sample size / only / evaluation setup 等；\n"
    "4. 不要回答问题本身，只做改写。\n"
    "\n"
    '只输出 JSON 对象，格式：{"intent": "extractive 或 analytical", "queries": ["查询1", "查询2"]}\n'
    "不要输出任何其他文字。"
)

# 改写/分类是机械任务：GLM 网关要求显式低档思考，避免每次改写都深度思考拖慢响应
REWRITE_EXTRA_BODY = {"thinking": {"type": "enabled", "level": "low"}}


@dataclass
class RewriteResult:
    intent: str  # "extractive" | "analytical"
    queries: list[str]  # 首个元素恒为原始查询


class QueryRewriter:
    def __init__(self, llm: LLM):
        self.llm = llm

    def rewrite(self, query: str, max_variants: int = 4) -> RewriteResult:
        """返回意图分类 + 改写查询列表（首个元素恒为原始查询）。"""
        queries = [query]
        intent = "extractive"
        try:
            raw = self.llm.chat(REWRITE_SYSTEM, query, temperature=0.0, extra_body=REWRITE_EXTRA_BODY)
            intent, parsed = self._parse(raw)
            for q in parsed:
                q = q.strip()
                if q and q not in queries:
                    queries.append(q)
        except Exception:
            pass  # 改写/分类失败 → 降级为提取型 + 原始查询
        return RewriteResult(intent=intent, queries=queries[:max_variants])

    @staticmethod
    def _parse(raw: str) -> tuple[str, list[str]]:
        """解析 {"intent", "queries"} 对象；兼容旧版裸数组输出。"""
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    intent = obj.get("intent", "extractive")
                    intent = intent if intent in ("extractive", "analytical") else "extractive"
                    queries = [q for q in obj.get("queries", []) if isinstance(q, str)]
                    return intent, queries
            except Exception:
                pass
        m = re.search(r"\[.*\]", raw, re.S)
        if m:
            try:
                arr = json.loads(m.group(0))
                if isinstance(arr, list):
                    return "extractive", [q for q in arr if isinstance(q, str)]
            except Exception:
                pass
        return "extractive", []
