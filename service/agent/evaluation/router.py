"""평가 결과에 따른 분기. add_conditional_edges에 연결한다."""

from langgraph.graph import END, StateGraph

from config import settings
from service.schema.state import GraphState

EVALUATE_NODE = "evaluate_report"
FINALIZE_NODE = "finalize_eval"
# Supervisor 패턴의 조정 노드 이름. 미달 시 재작업 판단은 Supervisor에게 넘긴다.
SUPERVISOR_NODE = "supervisor"


def make_eval_router(retry_node: str = SUPERVISOR_NODE, max_attempts: int | None = None):
    """통과면 종료, 미달이면 한도 안에서 retry_node로 재작업, 한도 도달 시 한계 기록 후 종료한다."""

    def route_after_evaluation(state: GraphState) -> str:
        limit = max_attempts or settings.eval_max_attempts
        verdict = state.get("eval_result")
        if verdict and verdict["passed"]:
            return END
        # eval_count는 평가 노드만 증가시키므로 재작업 루프가 limit회를 넘지 않는다.
        if state.get("eval_count", 0) < limit:
            return retry_node
        return FINALIZE_NODE

    return route_after_evaluation


route_after_evaluation = make_eval_router()


def attach_report_evaluation(
    workflow: StateGraph,
    *,
    source: str = "report_agent",
    retry_node: str = SUPERVISOR_NODE,
    judge_model=None,
    max_attempts: int | None = None,
    evaluator=None,
    finalizer=None,
) -> StateGraph:
    """보고서 생성 노드 뒤에 품질 평가 루프를 붙인다.

    source → evaluate_report ─ PASS → END
                             ├ FAIL & 한도 미만 → retry_node
                             └ FAIL & 한도 도달 → finalize_eval → END
    source 노드에서 END로 가는 기존 edge는 호출 측에서 제거해야 한다.
    evaluator·finalizer는 테스트에서 노드를 대체할 때만 지정한다.
    """
    from service.agent.evaluation.agent import make_report_evaluator
    from service.agent.evaluation.finalize import make_finalize_eval

    workflow.add_node(EVALUATE_NODE, evaluator or make_report_evaluator(judge_model))
    workflow.add_node(FINALIZE_NODE, finalizer or make_finalize_eval())
    workflow.add_edge(source, EVALUATE_NODE)
    workflow.add_conditional_edges(
        EVALUATE_NODE,
        make_eval_router(retry_node, max_attempts),
        [END, retry_node, FINALIZE_NODE],
    )
    workflow.add_edge(FINALIZE_NODE, END)
    return workflow
