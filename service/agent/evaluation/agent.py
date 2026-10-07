"""보고서 품질 평가 노드. 1안(룰베이스)과 2안(LLM Judge)을 합친 하이브리드 판정을 State에 기록한다.

그래프 위치: report_agent → report_evaluator → supervisor
재시도 여부·대상·횟수 상한은 Supervisor가 판단한다(policy.MAX_EVAL_RETRY). 이 노드는 판정만 한다.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from service.agent.evaluation.judge import judge_report
from service.agent.evaluation.rules import (
    PERSPECTIVES,
    RuleOutcome,
    check_groundedness,
    check_perspective_coverage,
)
from service.agent.report.pdf import RESULT_DIR
from service.schema.control import EvalResult
from service.schema.state import GraphState

logger = logging.getLogger(__name__)

# 내부 판정 이름 → EvalResult 키. 순서는 1안 Groundedness, 2안 중립성·편향 통제, 1안 관점 커버리지.
CRITERIA_KEYS = {
    "groundedness": "groundedness",
    "neutrality": "neutrality",
    "bias_control": "bias_control",
    "perspective_coverage": "coverage",
}
CRITERIA_LABELS = {
    "groundedness": "Groundedness",
    "neutrality": "중립성",
    "bias_control": "편향 통제",
    "perspective_coverage": "관점 커버리지",
}


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def to_eval_result(outcomes: dict[str, RuleOutcome]) -> EvalResult:
    """항목별 판정을 Supervisor가 읽는 EvalResult로 줄인다."""
    failed = {name: outcome for name, outcome in outcomes.items() if not outcome.result["passed"]}
    perspectives = set(PERSPECTIVES.values())
    return {
        "passed": not failed,
        **{key: outcomes[name].result["passed"] for name, key in CRITERIA_KEYS.items()},
        # 재작업 담당이 바로 고칠 수 있는 수정 지시
        "issues": _unique([item for outcome in failed.values() for item in outcome.feedback]),
        # 어떤 항목이 왜 미달인지 한 줄 요약
        "retry_instruction": " / ".join(
            f"{CRITERIA_LABELS[name]} 미달: {'; '.join(outcome.result['reasons'])}" for name, outcome in failed.items()
        ),
        # 근거 보강이 필요한 관점. 보고서 서술 문제(report)는 넣지 않는다.
        "perspectives": _unique([t for outcome in failed.values() for t in outcome.retry_targets if t in perspectives]),
    }


def make_report_evaluator(judge_model=None, output_dir: Path = RESULT_DIR):
    """평가 노드를 만든다. judge_model을 생략하면 첫 호출 시 설정된 Judge 모델을 생성한다."""
    model_holder = {"model": judge_model}

    def report_evaluator(state: GraphState) -> dict:
        # Supervisor가 FAIL 재시도를 보낼 때 eval_retry_count를 올리므로 이 값으로 평가 회차를 정한다.
        attempt = state.get("eval_retry_count", 0) + 1
        report = state.get("report_markdown") or ""

        outcomes: dict[str, RuleOutcome] = {
            "groundedness": check_groundedness(state, report, settings.groundedness_min_ratio),
            "perspective_coverage": check_perspective_coverage(state, report),
        }
        if model_holder["model"] is None:
            from config.model import get_judge_model

            model_holder["model"] = get_judge_model()
        outcomes |= judge_report(model_holder["model"], state, report, settings.judge_pass_score)
        outcomes = {name: outcomes[name] for name in CRITERIA_KEYS}
        eval_result = to_eval_result(outcomes)

        # 점수·사유·위반 문장 등 상세 진단은 State가 아닌 외부 파일에 둔다(체크포인트 크기 제한).
        output_dir.mkdir(parents=True, exist_ok=True)
        detail_path = output_dir / f"eval_{attempt}.json"
        detail_path.write_text(json.dumps({
            # trace_id로 Supervisor 결정 로그·LangSmith 트레이스와 연결한다.
            "trace_id": state.get("trace_id"),
            "attempt": attempt,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "judge_model": settings.judge_model,
            "eval_result": eval_result,
            "criteria": {name: outcome.result for name, outcome in outcomes.items()},
            "details": {name: outcome.details for name, outcome in outcomes.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(json.dumps({
            "trace_id": state.get("trace_id"), "node": "report_evaluator", "attempt": attempt,
            "passed": eval_result["passed"],
            "failed": [name for name, outcome in outcomes.items() if not outcome.result["passed"]],
            "perspectives": eval_result["perspectives"], "detail": str(detail_path),
        }, ensure_ascii=False))
        return {"eval_result": eval_result}

    return report_evaluator


def report_evaluator(state: GraphState) -> dict:
    """run_agent.py의 기본 평가 노드(service.agent.evaluation:report_evaluator). Judge 모델은 첫 호출 때 만든다."""
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = make_report_evaluator()
    return _default_evaluator(state)


_default_evaluator = None
