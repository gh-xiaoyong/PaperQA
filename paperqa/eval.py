"""评测体系（Phase 3）：测试集驱动 → 三维指标 → Bad Case 三层归因。

指标：
- 检索召回率 hit@k：gold 页码是否出现在检索结果中（无答案型不适用）
- 回答准确率：LLM-as-judge 判 correct / partial / wrong；无答案型以「正确拒答且不编造」为正确
- 引用正确率：机械校验（引用的片段必须在检索结果中）+ judge 判「片段是否支撑论断」
- 推断有据率（分析型）：「推断」论断是否有片段依据且能从片段合理推出

Bad Case 三层归因（仅 verdict ≠ correct 时）：
- retrieval：gold 页码不在检索结果中，生成无从谈起
- prompt：证据已检索到，但回答未利用 / 错误拒答
- model：证据在手仍答错或编造

注意：judge 复用同源 GLM（low 档思考提速），存在一定乐观偏差；
正式对外报告建议换用异源 judge 复核。
"""
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .config import Settings
from .llm import LLM
from .pipeline import Pipeline

# judge 用低档思考：结构化判定任务无需深思考，且评测量大、提速收益明显
JUDGE_EXTRA_BODY = {"thinking": {"type": "enabled", "level": "low"}}

# 回答中的引用标记（容忍「来源:」「来源：」与页/片段顺序、顿号分隔多引用）
CITATION_RE = re.compile(r"来源\s*[:：]\s*([^\]\n]*)")
PAGE_RE = re.compile(r"第?\s*(\d+)\s*页")
CHUNK_RE = re.compile(r"片段\s*(\d+)")

REFUSAL_MARKERS = (
    "未提供", "未提及", "没有提及", "未说明", "无法判断",
    "无法回答", "无法从", "未报告", "无法确认", "没有描述",
)

VERDICT_SYSTEM = (
    "你是 RAG 问答系统的严格评测员。给定：用户问题、标准答案（gold）、检索到的片段页码、系统回答。\n"
    "判定 verdict：\n"
    '- correct：回答核心内容与 gold 一致，且无编造；\n'
    "- partial：部分正确或遗漏 gold 的要点；\n"
    "- wrong：内容错误、编造事实，或 gold 表示应拒答却强行回答。\n"
    "若 gold 表示「应拒答」：系统正确拒答（明确说明未提供，且没有编造）→ correct；否则 → wrong。\n"
    "\n"
    "fault_layer（仅 verdict ≠ correct 时填，否则为 null）：\n"
    "- retrieval：gold 页码不在检索页码列表中，生成侧无从谈起；\n"
    "- prompt：证据已在检索页码中，但回答未利用、答非所问或错误拒答；\n"
    "- model：证据在手、任务清晰，模型仍答错或编造。\n"
    "\n"
    '只输出 JSON：{"verdict": "correct|partial|wrong", "fault_layer": "retrieval|prompt|model|null", "reason": "一句话依据"}'
)

SUPPORT_SYSTEM = (
    "你是引用审核员。下面给出若干条目：每条包含「论断」（系统回答中的一段话）与它引用的「论文片段」。\n"
    "逐条判断：该片段是否足以支持该论断（允许基于片段综合归纳，但片段中没有的信息不能算支持）。\n"
    '只输出 JSON 数组：[{"i": 条目序号, "supported": true|false, "reason": "一句话"}]'
)

INFERENCE_SYSTEM = (
    "你是推理审核员。下面给出分析型回答中的「推断」论断列表，以及检索到的全部论文片段。\n"
    "逐条判断：该推断能否从片段内容合理推出（允许归纳与评价，但依据必须真的在片段里）。\n"
    '只输出 JSON 数组：[{"i": 条目序号, "grounded": true|false, "reason": "一句话"}]'
)


@dataclass
class CitationCheck:
    claim: str
    raw: str
    page: int | None
    chunk_id: int | None
    valid: bool  # (page, chunk) 能对应到检索结果
    supported: bool | None  # judge 判定；None = 未能配对送审


@dataclass
class QuestionResult:
    qid: str
    qtype: str
    question: str
    intent: str
    queries: list[str]
    gold_pages: list[int]
    retrieved_pages: list[int]
    recall_hit: bool | None  # 无答案型为 None（不适用）
    answer: str
    refusal: bool
    verdict: str
    fault_layer: str | None
    judge_reason: str
    citations: list[CitationCheck]
    inference_total: int
    inference_grounded: int
    stage_seconds: dict = None  # {retrieve, answer, judge} 分阶段耗时

    def __post_init__(self):
        if self.stage_seconds is None:
            self.stage_seconds = {}

    def citation_valid_rate(self) -> float | None:
        if not self.citations:
            return None
        return sum(1 for c in self.citations if c.valid) / len(self.citations)

    def citation_support_rate(self) -> float | None:
        judged = [c for c in self.citations if c.supported is not None]
        if not judged:
            return None
        return sum(1 for c in judged if c.supported) / len(judged)

    def inference_grounded_rate(self) -> float | None:
        if not self.inference_total:
            return None
        return self.inference_grounded / self.inference_total

    @classmethod
    def from_dict(cls, d: dict) -> "QuestionResult":
        """从检查点 JSON 反序列化（断点续跑用）。"""
        d = dict(d)
        d["citations"] = [CitationCheck(**c) for c in d.get("citations", [])]
        return cls(**d)


def _parse_json_obj(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_json_arr(raw: str) -> list | None:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


class EvalRunner:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.pipeline = Pipeline(self.settings, library=False)
        self.judge = LLM(self.settings)

    # ---------- 单题评测 ----------

    def evaluate_question(self, q: dict, top_k: int = 4) -> QuestionResult:
        t_retrieve = time.time()
        r = self.pipeline.retrieve(q["question"], top_k=top_k)
        t_answer = time.time()
        answer = "".join(
            delta for kind, delta in self.pipeline.answer_stream(q["question"], r.hits, intent=r.intent)
            if kind == "answer"
        )
        t_judge = time.time()
        retrieved_pages = sorted({h["page"] for h in r.hits})
        gold_pages = q.get("gold_pages", [])

        recall_hit = None if q["type"] == "no_answer" else any(g in retrieved_pages for g in gold_pages)
        # 拒答判定：分析型回答的「无法判断」小节属合法结构，需叠加长度条件避免误判
        refusal = len(answer) < 300 and any(m in answer for m in REFUSAL_MARKERS)

        verdict, fault_layer, reason = self._judge_verdict(q, answer, retrieved_pages, refusal)
        # judge 漏填归因时按定义机械兜底：证据没召回→retrieval；在召回了却拒答→prompt；在手仍错→model
        if verdict != "correct" and fault_layer is None:
            fault_layer = "retrieval" if recall_hit is False else ("prompt" if refusal else "model")

        citations = self._extract_citations(answer, r.hits)
        self._judge_citation_support(answer, citations, r.hits)

        inf_total, inf_grounded = 0, 0
        if q["type"] == "analytical":
            inf_total, inf_grounded = self._judge_inferences(answer, r.hits)

        return QuestionResult(
            qid=q["id"], qtype=q["type"], question=q["question"],
            intent=r.intent, queries=r.queries,
            gold_pages=gold_pages, retrieved_pages=retrieved_pages,
            recall_hit=recall_hit, answer=answer, refusal=refusal,
            verdict=verdict, fault_layer=fault_layer, judge_reason=reason,
            citations=citations,
            inference_total=inf_total, inference_grounded=inf_grounded,
            stage_seconds={
                "retrieve": round(t_answer - t_retrieve, 1),
                "answer": round(t_judge - t_answer, 1),
                "judge": round(time.time() - t_judge, 1),
            },
        )

    def _judge_verdict(self, q: dict, answer: str, retrieved_pages: list[int], refusal: bool):
        user = (
            f"【用户问题】{q['question']}\n"
            f"【标准答案 gold】{q['gold_answer']}\n"
            f"【gold 页码】{q.get('gold_pages', []) or '无（应为拒答）'}\n"
            f"【检索到的页码】{retrieved_pages}\n"
            f"【系统回答（是否检测到拒答措辞：{refusal}）】\n{answer[:2500]}"
        )
        raw = self.judge.chat(VERDICT_SYSTEM, user, temperature=0.0, extra_body=JUDGE_EXTRA_BODY)
        obj = _parse_json_obj(raw) or {}
        verdict = obj.get("verdict", "wrong")
        verdict = verdict if verdict in ("correct", "partial", "wrong") else "wrong"
        fault = obj.get("fault_layer")
        fault = fault if fault in ("retrieval", "prompt", "model") else None
        return verdict, fault, str(obj.get("reason", ""))[:200]

    def _extract_citations(self, answer: str, hits: list[dict]) -> list[CitationCheck]:
        hits_by_chunk = {h["id"]: h for h in hits}
        parts = CITATION_RE.split(answer)
        checks: list[CitationCheck] = []
        for i in range(1, len(parts), 2):
            args, claim = parts[i], parts[i - 1]
            claim = claim.strip()[-300:]  # 引用标记前最近的一段论述
            pages = [int(x) for x in PAGE_RE.findall(args)]
            chunks = [int(x) for x in CHUNK_RE.findall(args)]
            combos: list[tuple[int | None, int | None, bool]] = []
            if chunks:
                for ch in chunks:
                    pg = pages[0] if pages else None
                    h = hits_by_chunk.get(ch)
                    valid = h is not None and (pg is None or h["page"] == pg)
                    combos.append((pg, ch, valid))
            elif pages:
                for pg in pages:
                    valid = any(h["page"] == pg for h in hits)
                    combos.append((pg, None, valid))
            for pg, ch, valid in combos:
                checks.append(CitationCheck(claim=claim, raw=args[:60], page=pg, chunk_id=ch, valid=valid, supported=None))
        return checks

    def _judge_citation_support(self, answer: str, citations: list[CitationCheck], hits: list[dict]):
        text_by_chunk = {h["id"]: h["text"] for h in hits}
        items = [c for c in citations if c.chunk_id is not None and c.chunk_id in text_by_chunk][:12]
        if not items:
            return
        lines = []
        for i, c in enumerate(items, 1):
            chunk_text = text_by_chunk[c.chunk_id][:600]
            lines.append(f"[条目{i}]\n论断：{c.claim or '（整段回答）'}\n引用片段：{chunk_text}")
        raw = self.judge.chat(SUPPORT_SYSTEM, "\n\n".join(lines), temperature=0.0, extra_body=JUDGE_EXTRA_BODY)
        arr = _parse_json_arr(raw) or []
        by_idx = {item.get("i"): item for item in arr if isinstance(item, dict)}
        for i, c in enumerate(items, 1):
            item = by_idx.get(i)
            if item is not None:
                c.supported = bool(item.get("supported"))

    def _judge_inferences(self, answer: str, hits: list[dict]) -> tuple[int, int]:
        # 抽取带「推断」标记的论断行
        statements = []
        for line in answer.splitlines():
            s = line.strip().lstrip("-*·0123456789. 、")
            if "推断" in s and len(s) > 12:
                statements.append(s[:300])
        statements = statements[:10]
        if not statements:
            return 0, 0
        chunks_text = "\n\n".join(f"[片段{h['id']} / 第{h['page']}页] {h['text'][:400]}" for h in hits)
        user = "【推断论断列表】\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1)) + "\n\n【检索到的论文片段】\n" + chunks_text
        raw = self.judge.chat(INFERENCE_SYSTEM, user, temperature=0.0, extra_body=JUDGE_EXTRA_BODY)
        arr = _parse_json_arr(raw) or []
        grounded = sum(1 for item in arr if isinstance(item, dict) and item.get("grounded"))
        return len(statements), grounded

    # ---------- 整体运行 ----------

    def run(self, testset_path: Path, top_k: int = 4, limit: int | None = None,
            resume: bool = False, only: list[str] | None = None) -> dict:
        """运行评测。resume=True 时从检查点续跑；only 指定只跑某些 qid。

        健壮性：单题失败记为 verdict="error" 不中断全场；每题完成后立即写检查点，
        网关断流/重试风暴不会摧毁整场评测。
        """
        ts = json.loads(Path(testset_path).read_text(encoding="utf-8"))
        questions = ts["questions"][:limit] if limit else ts["questions"]
        if only:
            questions = [q for q in questions if q["id"] in only]

        reports_dir = Path("eval/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts_name = ts.get("name", "testset")
        partial_path = reports_dir / f"partial_{ts_name}.jsonl"

        done: dict[str, QuestionResult] = {}
        if resume and partial_path.exists():
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = QuestionResult.from_dict(json.loads(line))
                        done[r.qid] = r  # 同 qid 后写覆盖先写
                    except Exception:
                        pass

        # 按论文分组索引（只为待跑题目建索引；当前种子集一篇，30 篇扩充时同 schema 生效）
        pending = [q for q in questions if q["id"] not in done]
        by_pdf: dict[str, list[dict]] = {}
        for q in pending:
            by_pdf.setdefault(ts["pdf"], []).append(q)
        for pdf, qs in by_pdf.items():
            n = self.pipeline.index_pdf(pdf)
            print(f"[索引] {pdf} → {n} 片段（{len(qs)} 题待跑）")

        results: list[QuestionResult] = []
        for q in questions:
            if q["id"] in done:
                results.append(done[q["id"]])
                print(f"[↩] {q['id']} 命中检查点，跳过")
                continue
            try:
                t0 = time.time()
                res = self.evaluate_question(q, top_k=top_k)
                flag = "✅" if res.verdict == "correct" else ("🟡" if res.verdict == "partial" else "❌")
                print(f"[{flag}] {res.qid} {res.qtype:11s} intent={res.intent:10s} "
                      f"recall={res.recall_hit} verdict={res.verdict:8s} fault={res.fault_layer} "
                      f"({time.time() - t0:.0f}s)")
            except Exception as e:
                res = QuestionResult(
                    qid=q["id"], qtype=q["type"], question=q["question"], intent="error",
                    queries=[], gold_pages=q.get("gold_pages", []), retrieved_pages=[],
                    recall_hit=None, answer="", refusal=False,
                    verdict="error", fault_layer=None,
                    judge_reason=f"{type(e).__name__}: {str(e)[:150]}",
                    citations=[],
                )
                print(f"[⚠️] {q['id']} 评测异常：{type(e).__name__}: {str(e)[:120]}")
            results.append(res)
            with partial_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")

        report = self._aggregate(ts, results)
        return report

    def _aggregate(self, ts: dict, results: list[QuestionResult]) -> dict:
        ts_name = ts.get("name", "testset")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        applicable_recall = [r for r in results if r.recall_hit is not None]
        recall_rate = sum(1 for r in applicable_recall if r.recall_hit) / len(applicable_recall) if applicable_recall else None
        n = len(results)
        acc = sum(1 for r in results if r.verdict == "correct") / n if n else 0.0
        partial = sum(1 for r in results if r.verdict == "partial") / n if n else 0.0
        wrong = sum(1 for r in results if r.verdict == "wrong") / n if n else 0.0

        valid_rates = [r.citation_valid_rate() for r in results if r.citation_valid_rate() is not None]
        support_rates = [r.citation_support_rate() for r in results if r.citation_support_rate() is not None]
        inf_rates = [r.inference_grounded_rate() for r in results if r.inference_grounded_rate() is not None]
        faults = Counter(r.fault_layer for r in results if r.verdict != "correct" and r.fault_layer)

        summary = {
            "testset": ts_name,
            "total": n,
            "recall_hit_k": round(recall_rate, 3) if recall_rate is not None else None,
            "answer_accuracy": round(acc, 3),
            "partial_rate": round(partial, 3),
            "wrong_rate": round(wrong, 3),
            "citation_valid_rate": round(sum(valid_rates) / len(valid_rates), 3) if valid_rates else None,
            "citation_support_rate": round(sum(support_rates) / len(support_rates), 3) if support_rates else None,
            "inference_grounded_rate": round(sum(inf_rates) / len(inf_rates), 3) if inf_rates else None,
            "fault_distribution": dict(faults),
            "judge_model": self.settings.llm_model,
        }

        # ---------- Markdown 报告 ----------
        lines = [
            f"# PaperQA 效果评估报告 · {ts_name}",
            f"",
            f"- 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　|　测试集：`{ts_name}`（{n} 题）　|　judge：`{self.settings.llm_model}`（同源自评，存在乐观偏差）",
            f"- 检索链路：查询改写 + 意图路由 → 稠密+BM25（RRF）→ gte-rerank 重排",
            "",
            "## 总览",
            "",
            "| 指标 | 数值 |",
            "|---|---|",
            f"| 检索召回率 hit@k | {summary['recall_hit_k']} |",
            f"| 回答准确率（correct） | {summary['answer_accuracy']} |",
            f"| partial / wrong | {summary['partial_rate']} / {summary['wrong_rate']} |",
            f"| 引用有效率（机械） | {summary['citation_valid_rate']} |",
            f"| 引用支撑率（judge） | {summary['citation_support_rate']} |",
            f"| 推断有据率（分析型） | {summary['inference_grounded_rate']} |",
            f"| Bad Case 归因分布 | {summary['fault_distribution']} |",
            "",
            "## 分题结果",
            "",
            "| ID | 类型 | intent | recall | verdict | fault | 引用(有效/支撑) | 推断有据 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in results:
            cv = r.citation_valid_rate()
            cs = r.citation_support_rate()
            ig = r.inference_grounded_rate()
            lines.append(
                f"| {r.qid} | {r.qtype} | {r.intent} | {r.recall_hit} | {r.verdict} | {r.fault_layer or '-'} "
                f"| {f'{cv:.2f}' if cv is not None else '-'} / {f'{cs:.2f}' if cs is not None else '-'} "
                f"| {f'{r.inference_grounded}/{r.inference_total}' if r.inference_total else '-'} |"
            )
        lines += ["", "## Bad Case 明细", ""]
        bad = [r for r in results if r.verdict != "correct"]
        if not bad:
            lines.append("（无）")
        for r in bad:
            lines.append(f"### {r.qid} [{r.verdict}/{r.fault_layer}] {r.question}")
            lines.append(f"- judge：{r.judge_reason}")
            lines.append(f"- 回答摘要：{r.answer[:300]}…")
            lines.append("")

        reports_dir = Path("eval/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"report_{ts_name}_{stamp}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        json_path = reports_dir / f"results_{ts_name}_{stamp}.json"
        json_path.write_text(json.dumps(
            {"summary": summary, "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2
        ), encoding="utf-8")

        return {"summary": summary, "report_path": str(md_path), "results_path": str(json_path), "results": results}
