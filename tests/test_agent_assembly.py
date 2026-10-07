"""실제 기술 조사 서브그래프와 실제 에이전트 조립을 검증한다. 모델과 검색은 대역을 쓴다."""
import pytest

from service.agent.graph import build_agent_graph
from service.agent.graph.technical import build_technical_research_graph
from service.agent.supervisor.graph import build_supervisor_graph
from tests.fixtures import ModelFixture, PaperFixture, sample_state
from tests.supervisor_fixtures import verdict
from tests.test_agent_graph import Harness


def test_real_technical_subgraph_runs_and_receives_rework_instructions():
    model = ModelFixture()
    harness = Harness(verdicts=[verdict(groundedness=False, perspectives=["technical"]), verdict(passed=True)])
    nodes = harness.nodes()
    nodes["technical_agent"] = build_technical_research_graph(PaperFixture(), model, 2, rules="공통 규칙. ")
    graph = build_supervisor_graph(nodes, evaluator=harness.evaluator)
    initial = {key: value for key, value in sample_state().items() if key != "technical_retry_count"}
    out = graph.invoke({**initial, "quality_feedback": [], "revision_count": 0}, config={"recursion_limit": 60})
    assert out["technical_result"]["status"] == "complete" and out["next"] == "FINISH"
    assert out["rework_counts"] == {"technical": 1}
    # 서브그래프 내부 필드는 부모 State로 새지 않는다.
    assert "technical_retry_count" not in out and "technical_evidence" not in out
    # 첫 실행에는 지시가 없고, FAIL 뒤 재작업에는 Supervisor 지시가 모델 입력에 들어간다.
    assert model.contexts[0]["quality_feedback"] == []
    assert model.contexts[-1]["quality_feedback"] == ["[기술 조사] 문제", "[기술 조사] 고친다"]


def test_build_agent_graph_assembles_the_real_agents_around_the_supervisor():
    graph = build_agent_graph(PaperFixture(), ModelFixture(), 2, rules="공통 규칙. ",
                              evaluator=lambda state: {"eval_result": {"passed": True}})
    assert set(graph.get_graph().nodes) == {
        "__start__", "__end__", "supervisor", "technical_agent", "market_node", "stakeholder_node",
        "domain_agent", "synthesis_agent", "report_agent", "report_evaluator", "end_with_warning"}


def test_build_agent_graph_requires_an_evaluator():
    with pytest.raises(TypeError, match="evaluator"):
        build_agent_graph(PaperFixture(), ModelFixture())
    with pytest.raises(ValueError, match="evaluator"):
        build_agent_graph(PaperFixture(), ModelFixture(), evaluator=None)
