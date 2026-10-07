"""LLM 판정 함수 테스트. with_structured_output과 invoke만 흉내 내는 가짜 모델을 쓴다."""
import json

from service.agent.supervisor.judge import JudgeOutput, build_digest, make_judge
from service.agent.supervisor.policy import SYNTHESIS, decide
from tests.supervisor_fixtures import collected, result


class FakeModel:
    """with_structured_output과 invoke만 흉내 내고 받은 메시지를 기록한다."""

    def __init__(self, answer):
        self.answer, self.schema, self.messages = answer, None, None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.answer


def test_make_judge_sends_candidates_and_a_bounded_digest():
    model = FakeModel(JudgeOutput(next=SYNTHESIS, reason="충분하다", instructions=[]))
    state = collected(market_result=result("partial", ["a", "b", "c", "d", "e", "f"]), rework_counts={"market": 0})
    answer = make_judge(model)(decide(state), state)
    assert answer.next == SYNTHESIS and model.schema is JudgeOutput
    payload = json.loads(model.messages[-1][1])
    assert payload["kind"] == "sufficiency" and payload["candidates"] == [SYNTHESIS, "market_node"]
    assert payload["perspectives"]["market"]["status"] == "partial"
    assert payload["perspectives"]["market"]["limitations"] == ["a", "b", "c", "d", "e"]
    assert "eval_result" not in payload


def test_digest_includes_the_verdict_only_when_present():
    verdict = {"passed": False, "coverage": False, "issues": ["이해관계자 관점 누락"]}
    digest = build_digest(collected(eval_result=verdict, rework_counts={"domain": 1}))
    assert digest["eval_result"] == verdict
    assert digest["perspectives"]["domain"] == {"node": "domain_agent", "label": "도메인", "status": "complete",
                                                "limitations": [], "findings": 0, "evidence": 0, "rework_count": 1}
    assert digest["perspectives"]["technical"]["rework_count"] == 0
