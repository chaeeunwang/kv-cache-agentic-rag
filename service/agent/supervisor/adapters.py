"""quality_feedback을 읽지 않는 에이전트에 Supervisor 지시를 전달하는 래퍼."""

from collections.abc import Callable, Mapping
from typing import Any

INSTRUCTION_HEADER = "[Supervisor 지시: 아래 내용은 본문에 옮겨 쓰지 말고 이번 작업의 수정에만 반영한다]"


def with_request_instructions(node: Callable[[Mapping[str, Any]], dict]) -> Callable[[Mapping[str, Any]], dict]:
    """지시가 있으면 입력 State 사본의 request 끝에 덧붙여 넘긴다. 부모 State는 바꾸지 않는다."""

    def run(state: Mapping[str, Any]) -> dict:
        instructions = state.get("quality_feedback") or []
        if not instructions:
            return node(state)
        lines = "\n".join(f"- {item}" for item in instructions)
        return node({**state, "request": f"{state['request']}\n\n{INSTRUCTION_HEADER}\n{lines}"})

    return run
