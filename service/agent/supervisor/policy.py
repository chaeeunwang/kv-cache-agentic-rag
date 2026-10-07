"""Supervisor의 결정 규칙. State만 읽는 순수 함수이며 LLM을 호출하지 않는다."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

# 한도는 프롬프트가 아니라 코드 상수로 강제한다.
MAX_REWORK_PER_AGENT = 1
MAX_EVAL_RETRY = 1
MAX_STEPS = 20

FINISH = "FINISH"
END_WARNING = "END_WARNING"
SYNTHESIS = "synthesis_agent"
REPORT = "report_agent"

PERSPECTIVES = ("technical", "market", "stakeholder", "domain")
NODE_OF = {"technical": "technical_agent", "market": "market_node",
           "stakeholder": "stakeholder_node", "domain": "domain_agent"}
PERSPECTIVE_OF = {node: perspective for perspective, node in NODE_OF.items()}
RESULT_OF = {perspective: f"{perspective}_result" for perspective in PERSPECTIVES}
# 시장·이해관계자 에이전트는 이 이름이 든 지시만 읽는다(tavily/evaluation.py FEEDBACK_KEYWORDS).
LABEL_OF = {"technical": "기술 조사", "market": "시장", "stakeholder": "이해관계자", "domain": "도메인"}
EVIDENCE_CRITERIA = ("groundedness", "bias_control", "coverage")

State = Mapping[str, Any]
Kind = Literal["forced", "sufficiency", "fail_analysis"]


@dataclass(frozen=True)
class Choice:
    """한 스텝에서 고를 수 있는 후보와 코드 기본값."""

    candidates: tuple[str, ...]
    default: str
    kind: Kind
    reason: str


@dataclass(frozen=True)
class Verdict:
    passed: bool
    # Groundedness·편향 통제·관점 커버리지 중 하나라도 미달이면 True
    evidence_related: bool
    perspectives: tuple[str, ...]
    instructions: tuple[str, ...]


def read_verdict(raw: Mapping[str, Any]) -> Verdict:
    """평가 판정의 형태를 아는 유일한 곳. 빠진 키는 미달이 아닌 것으로 읽는다."""
    evidence_related = not all(raw.get(name, True) for name in EVIDENCE_CRITERIA)
    perspectives = tuple(p for p in raw.get("perspectives") or [] if p in PERSPECTIVES)
    instructions = [*(raw.get("issues") or [])]
    if raw.get("retry_instruction"):
        instructions.append(raw["retry_instruction"])
    return Verdict(bool(raw.get("passed")), evidence_related, perspectives, tuple(instructions))


def current_verdict(state: State) -> Verdict | None:
    """보고서와 판정이 모두 있을 때만 현재 보고서에 대한 판정으로 본다."""
    raw = state.get("eval_result")
    if not state.get("report_markdown") or not raw:
        return None
    return read_verdict(raw)


def pending_report_instructions(state: State) -> list[str]:
    """FAIL 판정이 남아 있으면 다음 보고서 생성에 넘길 수정 지시를 돌려준다."""
    raw = state.get("eval_result")
    if not raw or raw.get("passed"):
        return []
    return list(read_verdict(raw).instructions)


def status_of(state: State, perspective: str) -> str | None:
    result = state.get(RESULT_OF[perspective])
    return result["status"] if result else None


def has_rework_budget(state: State, perspective: str) -> bool:
    return (state.get("rework_counts") or {}).get(perspective, 0) < MAX_REWORK_PER_AGENT


def forced(target: str, reason: str) -> Choice:
    return Choice((target,), target, "forced", reason)


def decide(state: State) -> Choice:
    """설계 4절의 규칙을 위에서부터 적용해 처음 맞는 결정을 돌려준다."""
    verdict = current_verdict(state)
    report = state.get("report_markdown")
    synthesis = state.get("synthesis_result")

    if verdict and verdict.passed:                                  # 규칙 1
        return forced(FINISH, "보고서 품질 평가를 통과했다")
    if state.get("step_count", 0) >= MAX_STEPS:                     # 규칙 2
        if report:
            return forced(END_WARNING, f"스텝 상한({MAX_STEPS})에 도달했다")
        return forced(REPORT if synthesis else SYNTHESIS,
                      f"스텝 상한({MAX_STEPS})에 도달해 재작업 없이 보고서로 직행한다")
    if verdict:
        if state.get("eval_retry_count", 0) >= MAX_EVAL_RETRY:      # 규칙 3
            return forced(END_WARNING, "품질 평가가 FAIL이고 재시도를 이미 썼다")
        return fail_analysis(verdict)                               # 규칙 4

    missing = [p for p in PERSPECTIVES if not state.get(RESULT_OF[p])]
    if missing:                                                     # 규칙 5
        # 시장·이해관계자·도메인은 기술 조사 근거를 입력으로 쓰므로 기술 조사가 먼저다.
        target = "technical" if "technical" in missing else missing[0]
        return forced(NODE_OF[target], f"{LABEL_OF[target]} 결과가 아직 없다")
    for perspective in PERSPECTIVES:                                # 규칙 6
        if status_of(state, perspective) == "error" and has_rework_budget(state, perspective):
            return forced(NODE_OF[perspective], f"{LABEL_OF[perspective]} 결과가 오류로 끝나 다시 호출한다")
    if not synthesis:                                               # 규칙 7
        return sufficiency(state)
    if not report:                                                  # 규칙 8
        return forced(REPORT, "종합이 끝나 보고서를 생성한다")
    return forced(END_WARNING, "보고서는 있으나 품질 평가 판정이 없다")    # 규칙 9


def sufficiency(state: State) -> Choice:
    if state.get("eval_retry_count", 0) > 0:
        return forced(SYNTHESIS, "FAIL 재시도 중이므로 추가 재작업 없이 종합한다")
    partial = [NODE_OF[p] for p in PERSPECTIVES if status_of(state, p) == "partial" and has_rework_budget(state, p)]
    if not partial:
        return forced(SYNTHESIS, "재작업할 관점이 없어 종합으로 진행한다")
    return Choice((SYNTHESIS, *partial), partial[0], "sufficiency", "근거가 부분적인 관점이 있어 충분성을 판정한다")


def fail_analysis(verdict: Verdict) -> Choice:
    if not verdict.evidence_related:
        return forced(REPORT, "근거가 아닌 표현 문제만 미달이어서 보고서를 다시 쓴다")
    named = [NODE_OF[p] for p in verdict.perspectives]
    agents = named or [NODE_OF[p] for p in PERSPECTIVES]
    return Choice((REPORT, *agents), named[0] if named else REPORT, "fail_analysis",
                  "근거 관련 항목이 미달이어서 재작업 대상을 고른다")


def default_instructions(state: State, choice: Choice, perspective: str) -> list[str]:
    """LLM 지시가 없을 때 관점 에이전트에 줄 재작업 지시."""
    if choice.kind == "fail_analysis":
        return pending_report_instructions(state)
    result = state.get(RESULT_OF[perspective]) or {}
    if result.get("status") == "error":
        return ["직전 실행이 오류로 끝났다. 같은 작업을 다시 수행한다."]
    return list(result.get("limitations") or []) or ["근거가 부족한 항목을 다시 조사한다."]


def label_instructions(perspective: str, items: list[str]) -> list[str]:
    return [f"[{LABEL_OF[perspective]}] {item}" for item in items]


def dispatch_update(state: State, choice: Choice, target: str, instructions: list[str] | None) -> dict:
    """호출 대상에 따라 quality_feedback, 무효화, 카운터를 정한다(설계 4.3)."""
    if target in (FINISH, END_WARNING):
        return {}
    # 현재 보고서에 대한 FAIL 판정을 보고 내린 결정이면 FAIL 재시도 한 번을 쓴 것이다.
    retry = {"eval_retry_count": state.get("eval_retry_count", 0) + 1} if current_verdict(state) else {}
    if target == SYNTHESIS:
        return {"quality_feedback": []}
    if target == REPORT:
        return {"quality_feedback": pending_report_instructions(state), "eval_result": None, **retry}
    perspective = PERSPECTIVE_OF[target]
    if not state.get(RESULT_OF[perspective]):
        return {"quality_feedback": []}
    return {**_perspective_rework_update(state, choice, perspective, instructions), **retry}


def _perspective_rework_update(state: State, choice: Choice, perspective: str,
                              instructions: list[str] | None) -> dict:
    """관점 재호출에 필요한 기존 State 갱신을 구성한다."""
    counts = dict(state.get("rework_counts") or {})
    counts[perspective] = counts.get(perspective, 0) + 1
    items = instructions or default_instructions(state, choice, perspective)
    # 상류 근거가 바뀌므로 하류 산출물을 비운다. 평가 판정은 보고서를 다시 보낼 때 지시로 쓰려고 남긴다.
    return {"quality_feedback": label_instructions(perspective, items), "rework_counts": counts,
            "synthesis_result": None, "report_markdown": None}
