"""Supervisor 그래프 배선. 노드 callable을 주입받아 연결만 한다."""

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from service.agent.supervisor.node import end_with_warning_node, make_supervisor_node, route_from_supervisor
from service.agent.supervisor.policy import END_WARNING, FINISH, NODE_OF, REPORT, SYNTHESIS
from service.schema.state import GraphState

SUPERVISOR = "supervisor"
EVALUATOR = "report_evaluator"
WARNING_NODE = "end_with_warning"
# 작업을 마치면 supervisor로 돌아오는 하위 에이전트
WORKERS = (*NODE_OF.values(), SYNTHESIS)


def build_supervisor_graph(nodes: Mapping[str, Any], *, evaluator: Callable[..., dict] | None, judge=None,
                           checkpointer=None):
    """하위 에이전트와 평가 노드를 받아 Supervisor 그래프를 컴파일한다."""
    if evaluator is None:
        raise ValueError("보고서 품질 평가 노드(evaluator)가 필요하다. 자동 통과로 대체하지 않는다.")
    missing = [name for name in (*WORKERS, REPORT) if name not in nodes]
    if missing:
        raise ValueError(f"하위 에이전트 노드가 없다: {missing}")

    workflow = StateGraph(GraphState)
    workflow.add_node(SUPERVISOR, make_supervisor_node(judge))
    for name in WORKERS:
        workflow.add_node(name, nodes[name])
        # 하위 에이전트는 서로를 직접 호출하지 않고 항상 supervisor로 돌아온다.
        workflow.add_edge(name, SUPERVISOR)
    workflow.add_node(REPORT, nodes[REPORT])
    workflow.add_node(EVALUATOR, evaluator)
    workflow.add_node(WARNING_NODE, end_with_warning_node)

    workflow.add_edge(START, SUPERVISOR)
    workflow.add_conditional_edges(
        SUPERVISOR,
        route_from_supervisor,
        {**{name: name for name in (*WORKERS, REPORT)}, FINISH: END, END_WARNING: WARNING_NODE},
    )
    # 보고서가 생성되면 반드시 품질 평가를 거쳐 supervisor로 돌아온다.
    workflow.add_edge(REPORT, EVALUATOR)
    workflow.add_edge(EVALUATOR, SUPERVISOR)
    workflow.add_edge(WARNING_NODE, END)
    return workflow.compile(checkpointer=checkpointer)
