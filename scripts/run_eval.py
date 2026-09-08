"""运行评测：python scripts/run_eval.py [--testset eval/testset.sample.json] [--limit N] [--top-k 4]

输出：console 进度 + eval/reports/ 下的 Markdown 报告与原始 JSON。
"""
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from paperqa.config import Settings
from paperqa.eval import EvalRunner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval/testset.sample.json")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    ap.add_argument("--resume", action="store_true", help="从检查点续跑（跳过已完成题目）")
    ap.add_argument("--only", default=None, help="只跑指定 qid，逗号分隔，如 --only A3,N2")
    args = ap.parse_args()

    runner = EvalRunner(Settings())
    report = runner.run(
        Path(args.testset), top_k=args.top_k, limit=args.limit,
        resume=args.resume, only=[x.strip() for x in args.only.split(",")] if args.only else None,
    )
    s = report["summary"]
    print("\n" + "=" * 60)
    print(f"总题数: {s['total']}")
    print(f"检索召回 hit@k      : {s['recall_hit_k']}")
    print(f"回答准确率 (correct): {s['answer_accuracy']}  (partial {s['partial_rate']} / wrong {s['wrong_rate']})")
    print(f"引用有效率 (机械)   : {s['citation_valid_rate']}")
    print(f"引用支撑率 (judge)  : {s['citation_support_rate']}")
    print(f"推断有据率 (分析型) : {s['inference_grounded_rate']}")
    print(f"Bad Case 归因       : {s['fault_distribution']}")
    print(f"报告: {report['report_path']}")
    print(f"明细: {report['results_path']}")


if __name__ == "__main__":
    main()
