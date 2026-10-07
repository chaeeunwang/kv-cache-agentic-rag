"""Supervisor 노드의 판정 경계 테스트. 미리 정한 답을 돌려주는 판정 대역만 쓴다."""
import json
from dataclasses import dataclass, field

import pytest

from service.agent.supervisor.node import end_with_warning_node, make_supervisor_node, route_from_supervisor
from service.agent.supervisor.policy import REPORT, SYNTHESIS
from tests.supervisor_fixtures import collected, result


@dataclass
class Answer:
    """판정 함수가 돌려주는 답의 모양(next, reason, instructions)."""

    next: str
    reason: str
    instructions: list[str] = field(default_factory=list)


def partial_market(**overrides):
    return collected(market_result=result("partial", ["hw_01 상용화 근거 없음"]), **overrides)


class ScriptedJudge:
    """미리 정한 답을 돌려주고 호출을 기록하는 판정 대역."""

    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def __call__(self, choice, state):
        self.calls.append(choice)
        if self.error:
            raise self.error
        return self.answer


def test_answer_inside_candidates_is_adopted():
    judge = ScriptedJudge(Answer(next="market_node", reason="파일럿 사례를 더 찾을 수 있다",
                                      instructions=["CXL-PNM 파일럿 사례를 찾는다"]))
    update = make_supervisor_node(judge)(partial_market())
    assert update["next"] == "market_node"
    assert update["decision"] == {"next": "market_node", "reason": "파일럿 사례를 더 찾을 수 있다", "by": "llm"}
    assert update["quality_feedback"] == ["[시장] CXL-PNM 파일럿 사례를 찾는다"]
    assert update["rework_counts"] == {"market": 1} and update["step_count"] == 1
    assert judge.calls[0].candidates == (SYNTHESIS, "market_node")


def test_judge_can_decide_the_evidence_is_sufficient():
    judge = ScriptedJudge(Answer(next=SYNTHESIS, reason="공개 정보의 한계다", instructions=[]))
    update = make_supervisor_node(judge)(partial_market())
    assert update["next"] == SYNTHESIS and update["decision"]["by"] == "llm"
    assert update["quality_feedback"] == [] and "rework_counts" not in update


@pytest.mark.parametrize("answer", [
    Answer(next=REPORT, reason="후보 밖", instructions=[]),          # 이 지점의 후보가 아닌 노드
    Answer(next="Market_Node", reason="대소문자 오타", instructions=[]),
    Answer(next="", reason="빈 답", instructions=[]),
    None,
])
def test_unusable_answer_falls_back_to_the_code_default(answer):
    update = make_supervisor_node(ScriptedJudge(answer))(partial_market())
    assert update["next"] == "market_node" and update["decision"]["by"] == "guard"
    assert update["quality_feedback"] == ["[시장] hw_01 상용화 근거 없음"]


def test_judge_exception_falls_back_to_the_code_default():
    update = make_supervisor_node(ScriptedJudge(error=RuntimeError("API 오류")))(partial_market())
    assert update["next"] == "market_node" and update["decision"]["by"] == "guard"
    assert "API 오류" not in json.dumps(update, ensure_ascii=False)


def test_judge_is_not_called_when_there_is_one_candidate():
    judge = ScriptedJudge(Answer(next="market_node", reason="x", instructions=[]))
    update = make_supervisor_node(judge)(collected())
    assert judge.calls == [] and update["next"] == SYNTHESIS and update["decision"]["by"] == "rule"


def test_without_a_judge_the_default_is_a_rule_decision():
    update = make_supervisor_node()(partial_market(step_count=4))
    assert update["next"] == "market_node" and update["decision"]["by"] == "rule"
    assert update["step_count"] == 5


def test_route_reads_the_decision_from_state():
    assert route_from_supervisor({"next": "domain_agent"}) == "domain_agent"


def test_end_with_warning_records_reason_and_failed_criteria():
    state = {"decision": {"next": "END_WARNING", "reason": "품질 평가가 FAIL이고 재시도를 이미 썼다", "by": "rule"},
             "eval_result": {"passed": False, "groundedness": True, "neutrality": False, "coverage": False}}
    warning = end_with_warning_node(state)["warning"]
    assert warning == "품질 평가가 FAIL이고 재시도를 이미 썼다. 미달 항목: neutrality, coverage."
    assert end_with_warning_node({})["warning"] == "사유 없음."
