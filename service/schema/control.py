"""Supervisor가 쓰는 제어 필드와 보고서 품질 평가 판정 형식."""

from typing import Literal, TypedDict


class DecisionRecord(TypedDict):
    next: str
    reason: str
    # rule: 코드 규칙, llm: LLM 판정, guard: LLM 답을 쓸 수 없어 코드 기본값으로 대체
    by: Literal["rule", "llm", "guard"]


class EvalResult(TypedDict):
    passed: bool
    groundedness: bool
    neutrality: bool
    bias_control: bool
    coverage: bool
    issues: list[str]
    retry_instruction: str
    # 근거 보강이 필요한 관점: technical, market, stakeholder, domain
    perspectives: list[str]


class SupervisorControl(TypedDict, total=False):
    """GraphState가 상속하는 제어 필드. 모두 선택 필드다."""

    next: str
    step_count: int
    rework_counts: dict[str, int]
    eval_retry_count: int
    decision: DecisionRecord
    warning: str | None
    trace_id: str
    eval_result: EvalResult | None
