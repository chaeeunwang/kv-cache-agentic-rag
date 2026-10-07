"""KV cache 평가 에이전트의 최종 LangGraph 정의."""

from service.agent.graph.technical import build_technical_research_graph
from service.agent.node.domain import domain_node
from service.agent.node.market import market_node
from service.agent.node.stakeholder import stakeholder_node
from service.agent.report import report_agent
from service.agent.supervisor import build_supervisor_graph, with_request_instructions
from service.agent.synthesis import synthesis_agent


def build_agent_graph(index, technical_model=None, max_technical_retries: int = 2, *, rules: str = "",
                      evaluator, judge=None, checkpointer=None):
    """실제 하위 에이전트를 조립해 Supervisor 그래프를 만들고 컴파일한다.

    실행 순서는 여기서 정하지 않는다. supervisor 노드가 State를 보고 다음 에이전트를 고르며,
    하위 에이전트는 작업을 마치면 supervisor로 돌아온다. evaluator는 보고서 품질 평가 노드로 필수다.
    """
    technical_graph = build_technical_research_graph(
        index,
        model=technical_model,
        max_retries=max_technical_retries,
        rules=rules,
    )
    nodes = {
        "technical_agent": technical_graph,
        "market_node": market_node,
        "stakeholder_node": stakeholder_node,
        # 도메인·보고서 에이전트는 quality_feedback을 읽지 않으므로 지시를 request에 덧붙여 넘긴다.
        "domain_agent": with_request_instructions(domain_node),
        "synthesis_agent": synthesis_agent,
        "report_agent": with_request_instructions(report_agent),
    }
    return build_supervisor_graph(nodes, evaluator=evaluator, judge=judge, checkpointer=checkpointer)
