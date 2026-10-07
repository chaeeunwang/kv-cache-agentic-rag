"""보고서 품질 평가 노드. 1안(룰베이스)과 2안(LLM Judge)을 합친 하이브리드 판정을 State에 기록한다."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from service.agent.evaluation.judge import judge_report
from service.agent.evaluation.rules import RuleOutcome, check_groundedness, check_perspective_coverage
from service.agent.report.pdf import RESULT_DIR
from service.schema.state import EvalVerdict, GraphState

logger = logging.getLogger(__name__)

# 판정·보고 순서: 1안 Groundedness, 2안 중립성·편향 통제, 1안 관점 커버리지
CRITERIA_ORDER = ("groundedness", "neutrality", "bias_control", "perspective_coverage")


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def make_report_evaluator(judge_model=None, output_dir: Path = RESULT_DIR):
    """평가 노드를 만든다. judge_model을 생략하면 첫 호출 시 설정된 Judge 모델을 생성한다."""
    model_holder = {"model": judge_model}

    def evaluate_report(state: GraphState) -> dict:
        attempt = state.get("eval_count", 0) + 1
        report = state.get("report_markdown") or ""

        outcomes: dict[str, RuleOutcome] = {
            "groundedness": check_groundedness(state, report, settings.groundedness_min_ratio),
            "perspective_coverage": check_perspective_coverage(state, report),
        }
        if model_holder["model"] is None:
            from config.model import get_judge_model

            model_holder["model"] = get_judge_model()
        outcomes |= judge_report(model_holder["model"], state, report, settings.judge_pass_score)

        ordered = {name: outcomes[name] for name in CRITERIA_ORDER}
        passed = all(outcome.result["passed"] for outcome in ordered.values())

        # 위반 문장·미인용 주장 등 상세 진단은 State가 아닌 외부 파일에 둔다(체크포인트 크기 제한).
        detail_path = output_dir / f"eval_{attempt}.json"
        verdict: EvalVerdict = {
            "passed": passed,
            "attempt": attempt,
            "criteria": {name: outcome.result for name, outcome in ordered.items()},
            "feedback": _unique([item for outcome in ordered.values() for item in outcome.feedback]),
            "retry_targets": _unique([t for outcome in ordered.values() for t in outcome.retry_targets]),
            "detail_path": str(detail_path),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps({
            # trace_id가 State에 있으면 함께 기록해 LangSmith 트레이스와 연결한다.
            "trace_id": state.get("trace_id"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "judge_model": settings.judge_model,
            "verdict": verdict,
            "details": {name: outcome.details for name, outcome in ordered.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "report_evaluation attempt=%s passed=%s failed=%s retry_targets=%s",
            attempt, passed,
            [name for name, outcome in ordered.items() if not outcome.result["passed"]],
            verdict["retry_targets"],
        )
        return {"eval_result": verdict, "eval_count": attempt}

    return evaluate_report
