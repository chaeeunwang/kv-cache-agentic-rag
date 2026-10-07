"""Supervisor 노드, 라우팅 함수, 경고 종료 노드."""

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from service.agent.supervisor.policy import Choice, decide, dispatch_update

logger = logging.getLogger("supervisor")

# judge(choice, state)는 next·reason·instructions 속성을 가진 답 또는 None을 돌려준다.
Judge = Callable[[Choice, Mapping[str, Any]], Any]
CRITERION_NAMES = ("groundedness", "neutrality", "bias_control", "coverage")


def make_supervisor_node(judge: Judge | None = None):
    """판정 함수를 주입받아 supervisor 노드를 만든다. judge가 없으면 코드 기본값만 쓴다."""

    def supervisor_node(state: Mapping[str, Any]) -> dict:
        choice = decide(state)
        target, reason, by, instructions = choice.default, choice.reason, "rule", None
        # 선택지가 둘 이상일 때만 LLM에 묻고, 답은 코드가 계산한 후보 안에서만 받아들인다.
        if judge is None and len(choice.candidates) > 1:
            reason = f"{choice.reason} (판정 함수가 없어 기본값을 적용했다)"
        elif len(choice.candidates) > 1:
            try:
                answer = judge(choice, state)
            except Exception:
                answer = None
            if answer is not None and answer.next in choice.candidates:
                target, reason, by = answer.next, answer.reason, "llm"
                instructions = list(answer.instructions)
            else:
                reason, by = f"{choice.reason} (LLM 판정을 쓸 수 없어 기본값을 적용했다)", "guard"
        step = state.get("step_count", 0) + 1
        update = dispatch_update(state, choice, target, instructions)
        update.update({"next": target, "step_count": step, "decision": {"next": target, "reason": reason, "by": by}})
        # 결정 이력은 State에 쌓지 않고 trace_id를 붙여 로그로 내보낸다.
        logger.info(json.dumps({"trace_id": state.get("trace_id"), "step": step, "next": target,
                                "reason": reason, "by": by}, ensure_ascii=False))
        return update

    return supervisor_node


def route_from_supervisor(state: Mapping[str, Any]) -> str:
    """조건부 엣지 함수. supervisor 노드가 State에 쓴 결정을 읽기만 한다."""
    return state["next"]


def end_with_warning_node(state: Mapping[str, Any]) -> dict:
    """한도에 걸려 끝날 때 사유와 마지막 판정의 미달 항목을 남긴다."""
    reason = (state.get("decision") or {}).get("reason", "사유 없음")
    raw = state.get("eval_result") or {}
    failed = [name for name in CRITERION_NAMES if raw.get(name) is False]
    detail = f" 미달 항목: {', '.join(failed)}." if failed else ""
    return {"warning": f"{reason}.{detail}"}
