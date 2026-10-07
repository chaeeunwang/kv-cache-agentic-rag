"""저장된 State(result/state.json)로 보고서 품질 평가 노드만 단독 실행한다.

그래프 전체를 다시 돌리지 않고 Judge 호출만으로 판정·임계값을 점검한다 (OpenAI API 비용 발생).
    uv run python scripts/eval_check.py --state result/state.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from service.agent.evaluation import make_report_evaluator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="보고서 품질 평가 단독 실행")
    parser.add_argument("--state", type=Path, default=Path("result/state.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/eval_check"))
    args = parser.parse_args()

    if not settings.openai_api_key:
        parser.error("OPENAI_API_KEY가 없습니다. .env를 확인하세요.")
    if not args.state.is_file():
        parser.error(f"{args.state}가 없습니다. 먼저 uv run run_agent.py로 State를 생성하세요.")
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if not state.get("report_markdown"):
        parser.error(f"{args.state}에 report_markdown이 없습니다. 보고서가 생성된 State로 실행하세요.")
    # 그래프 재작업 횟수와 무관하게 단독 점검은 항상 1회차로 기록한다.
    state["eval_count"] = 0

    print(f"Judge 모델: {settings.judge_model} / 인용 비율 기준 {settings.groundedness_min_ratio:.0%} / "
          f"Judge 통과 {settings.judge_pass_score}점", flush=True)
    verdict = make_report_evaluator(output_dir=args.output_dir)(state)["eval_result"]

    print(f"\n종합: {'PASS' if verdict['passed'] else 'FAIL'}")
    for name, result in verdict["criteria"].items():
        score = "-" if result["score"] is None else f"{result['score']:g}"
        print(f"- {name:<21} {result['method']:<9} {'PASS' if result['passed'] else 'FAIL'}  score={score}")
        for reason in result["reasons"]:
            print(f"    · {reason}")
    if verdict["retry_targets"]:
        print("\n재작업 권고 대상:", ", ".join(verdict["retry_targets"]))
    for item in verdict["feedback"]:
        print(f"  - {item}")
    print(f"\n상세: {verdict['detail_path']}")


if __name__ == "__main__":
    main()
