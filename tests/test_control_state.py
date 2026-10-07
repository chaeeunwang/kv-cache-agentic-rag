"""GraphState에 얹은 Supervisor 제어 필드 테스트."""
from typing import get_type_hints

from langgraph.graph import END, START, StateGraph

from service.schema.control import SupervisorControl
from service.schema.state import GraphState

CONTROL_FIELDS = {"next", "step_count", "rework_counts", "eval_retry_count", "decision", "warning", "trace_id",
                  "eval_result"}


def test_graph_state_inherits_optional_control_fields():
    assert set(SupervisorControl.__annotations__) == CONTROL_FIELDS
    assert CONTROL_FIELDS <= set(get_type_hints(GraphState))
    assert CONTROL_FIELDS <= GraphState.__optional_keys__
    # 기존 필수 필드는 그대로여서 기존 코드가 만드는 State가 계속 유효하다.
    assert GraphState.__required_keys__ == {"request", "target_domain", "evaluation_criteria", "technologies",
                                            "quality_feedback", "revision_count"}


def test_langgraph_carries_control_fields_and_cleared_values():
    def node(state):
        return {"next": "FINISH", "step_count": state.get("step_count", 0) + 1,
                "synthesis_result": None, "report_markdown": None, "eval_result": None}

    workflow = StateGraph(GraphState)
    workflow.add_node("node", node)
    workflow.add_edge(START, "node")
    workflow.add_edge("node", END)
    out = workflow.compile().invoke({"request": "평가", "report_markdown": "이전 보고서"})
    assert out["next"] == "FINISH" and out["step_count"] == 1
    assert out["report_markdown"] is None and out["synthesis_result"] is None and out["eval_result"] is None
