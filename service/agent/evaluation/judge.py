"""2안 LLM Judge 평가: 중립성과 편향 통제를 루브릭으로 채점한다."""

import json
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from service.agent.evaluation.prompt import JUDGE_SYSTEM_PROMPT
from service.agent.evaluation.rules import EVIDENCE_FIELDS, RuleOutcome, extract_citations, split_reference
from service.schema.state import GraphState

RetryTarget = Literal["report", "technical", "market", "stakeholder", "domain"]


class JudgeCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # strict 구조화 출력 호환을 위해 범위 제약은 코드에서 보정한다.
    score: int = Field(description="1~5점")
    violations: list[str] = Field(description="위반 근거가 되는 보고서 원문 문장, 최대 5개")
    rationale: str
    fix_directions: list[str] = Field(description="수정 방향, 최대 3개")
    retry_targets: list[RetryTarget]


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    neutrality: JudgeCriterion
    bias_control: JudgeCriterion


def build_judge_signals(state: GraphState, report: str) -> dict:
    """[확장 지점] Judge 입력에 덧붙일 룰 기반 보조 신호.

    팀 합의에 따라 중립성·편향 통제는 LLM Judge만으로 판정하므로 현재는 빈 dict를 반환한다.
    확장 시 출처 도메인 분포(단일 출처 비율), 관점별 finding stance 분포(positive/negative),
    기술별 근거 수 비대칭 등을 계산해 넣으면 Judge 판정을 수치 근거로 고정할 수 있다.
    """
    return {}


def judge_context(state: GraphState, report: str) -> dict:
    """보고서와 인용 근거의 출처 정보를 Judge 입력으로 만든다.

    출처 다양성을 판단할 수 있도록 인용된 근거의 출처(제목·도메인)만 원자료로 전달하고,
    분포 통계는 계산하지 않는다(build_judge_signals 확장 지점 참고).
    """
    body, _ = split_reference(report)
    cited = set(extract_citations(body))
    catalog = {}
    for name in EVIDENCE_FIELDS:
        for evidence in (state.get(name) or {}).get("evidence", []):
            if evidence["id"] in cited and evidence["id"] not in catalog:
                catalog[evidence["id"]] = {
                    "source_type": evidence.get("source_type"),
                    "title": evidence.get("title"),
                    "domain": urlparse(evidence.get("url") or "").netloc or None,
                    "published_at": evidence.get("published_at"),
                }
    context = {
        "target_domain": state.get("target_domain"),
        "technologies": [{"id": t["id"], "name": t["name"], "approach": t["approach"]}
                         for t in state.get("technologies", [])],
        "cited_evidence_sources": catalog,
        "report_markdown": report,
    }
    if signals := build_judge_signals(state, report):
        context["signals"] = signals
    return context


def to_outcome(criterion: JudgeCriterion, pass_score: int) -> RuleOutcome:
    score = min(max(criterion.score, 1), 5)
    passed = score >= pass_score
    reasons = [f"Judge {score}/5: {criterion.rationale}"]
    return RuleOutcome(
        result={"passed": passed, "method": "llm_judge", "score": float(score), "reasons": reasons},
        feedback=[] if passed else criterion.fix_directions[:3],
        retry_targets=[] if passed else (list(criterion.retry_targets) or ["report"]),
        details={"violations": criterion.violations[:5], "rationale": criterion.rationale,
                 "fix_directions": criterion.fix_directions[:3]},
    )


def judge_report(model, state: GraphState, report: str, pass_score: int) -> dict[str, RuleOutcome]:
    """Judge를 호출해 neutrality·bias_control 결과를 반환한다. 호출 실패는 미달로 기록한다."""
    try:
        output: JudgeOutput = model.with_structured_output(JudgeOutput).invoke([
            ("system", JUDGE_SYSTEM_PROMPT),
            ("human", json.dumps(judge_context(state, report), ensure_ascii=False)),
        ])
    except Exception as error:  # 모델·네트워크 오류로 평가가 멈추지 않게 하고 사유를 남긴다.
        reason = f"Judge 호출 실패({type(error).__name__})"
        failed = RuleOutcome(
            result={"passed": False, "method": "llm_judge", "score": None, "reasons": [reason]},
            feedback=[reason + ": 재평가가 필요합니다."],
            retry_targets=["report"],
            details={"error": str(error)[:500]},
        )
        return {"neutrality": failed, "bias_control": failed}
    return {
        "neutrality": to_outcome(output.neutrality, pass_score),
        "bias_control": to_outcome(output.bias_control, pass_score),
    }
