"""지시 전달 래퍼 테스트."""
import pytest

from service.agent.supervisor.adapters import INSTRUCTION_HEADER, with_request_instructions


def test_adapter_appends_instructions_without_touching_the_parent_state():
    seen = []
    node = with_request_instructions(lambda state: seen.append(state) or {})
    parent = {"request": "원래 요청", "quality_feedback": ["SUMMARY의 추천 표현을 지운다"]}
    node(parent)
    assert parent["request"] == "원래 요청"
    assert seen[0]["request"] == f"원래 요청\n\n{INSTRUCTION_HEADER}\n- SUMMARY의 추천 표현을 지운다"


@pytest.mark.parametrize("parent", [
    {"request": "요청"},
    {"request": "요청", "quality_feedback": []},
    {"request": "요청", "quality_feedback": None},
])
def test_adapter_passes_the_state_through_when_there_are_no_instructions(parent):
    seen = []
    with_request_instructions(lambda state: seen.append(state) or {})(parent)
    assert seen[0] is parent
