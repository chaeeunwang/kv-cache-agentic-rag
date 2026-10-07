"""가짜 하위 에이전트로 Supervisor 그래프의 동적 경로를 검증한다. 외부 API를 쓰지 않는다."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from service.agent.supervisor import policy
from service.agent.supervisor.adapters import INSTRUCTION_HEADER, with_request_instructions
from service.agent.supervisor.graph import build_supervisor_graph
from tests.supervisor_fixtures import result, verdict

FIELD_OF = {"technical_agent": "technical_result", "market_node": "market_result",
            "stakeholder_node": "stakeholder_result", "domain_agent": "domain_result"}
NORMAL = ["technical_agent", "market_node", "stakeholder_node", "domain_agent",
          "synthesis_agent", "report_agent", "report_evaluator"]


def scripted(script):
    """목록을 앞에서부터 하나씩 돌려주고 마지막 값은 계속 되풀이한다."""
    return script.pop(0) if len(script) > 1 else script[0]


class Harness:
    """호출 순서와 각 노드가 받은 입력을 기록하는 가짜 그래프 부품."""

    def __init__(self, statuses=None, verdicts=None):
        self.statuses = {name: list(values) for name, values in (statuses or {}).items()}
        self.verdicts = list(verdicts or [verdict(passed=True)])
        self.calls: list[str] = []
        self.seen: dict[str, list[dict]] = {}

    def record(self, name, state):
        self.calls.append(name)
        self.seen.setdefault(name, []).append(
            {"request": state["request"], "quality_feedback": list(state.get("quality_feedback") or [])})

    def worker(self, name):
        def run(state):
            self.record(name, state)
            status = scripted(self.statuses.setdefault(name, ["complete"]))
            return {FIELD_OF[name]: result(status, [] if status == "complete" else [f"{name} 근거 부족"])}
        return run

    def synthesis(self, state):
        self.record("synthesis_agent", state)
        # 실제 종합 에이전트처럼 quality_feedback과 revision_count도 함께 쓴다.
        return {"synthesis_result": result(), "quality_feedback": ["종합이 남긴 피드백"],
                "revision_count": state.get("revision_count", 0) + 1}

    def report(self, state):
        self.record("report_agent", state)
        return {"report_markdown": f"# SUMMARY v{self.calls.count('report_agent')}", "report_evidence_ids": []}

    def evaluator(self, state):
        self.record("report_evaluator", state)
        return {"eval_result": scripted(self.verdicts)}

    def nodes(self):
        return {**{name: self.worker(name) for name in FIELD_OF}, "synthesis_agent": self.synthesis,
                "report_agent": with_request_instructions(self.report)}

    def run(self, initial=None, **kwargs):
        graph = build_supervisor_graph(self.nodes(), evaluator=self.evaluator, **kwargs)
        state = {"request": "두 기술을 비교한다", "quality_feedback": [], "revision_count": 0, **(initial or {})}
        return graph.invoke(state, config={"recursion_limit": 60, "configurable": {"thread_id": "t1"}})


def test_normal_path_finishes_after_one_pass():
    harness = Harness()
    out = harness.run()
    assert harness.calls == NORMAL
    assert out["next"] == "FINISH" and out["step_count"] == 7 and "warning" not in out
    assert out["report_markdown"] == "# SUMMARY v1"
    # 종합이 남긴 quality_feedback은 Supervisor가 덮어쓰므로 보고서 지시로 넘어가지 않는다.
    assert harness.seen["report_agent"][0]["request"] == "두 기술을 비교한다"


def test_start_state_with_existing_results_skips_them():
    harness = Harness()
    harness.run(initial={"technical_result": result(), "market_result": result()})
    assert harness.calls == ["stakeholder_node", "domain_agent", "synthesis_agent", "report_agent", "report_evaluator"]


def test_only_the_partial_perspective_is_reworked():
    harness = Harness(statuses={"market_node": ["partial", "complete"]})
    out = harness.run()
    assert harness.calls == ["technical_agent", "market_node", "stakeholder_node", "domain_agent",
                             "market_node", "synthesis_agent", "report_agent", "report_evaluator"]
    assert harness.seen["market_node"][0]["quality_feedback"] == []
    assert harness.seen["market_node"][1]["quality_feedback"] == ["[시장] market_node 근거 부족"]
    assert out["rework_counts"] == {"market": 1}


def test_perspective_that_stays_partial_is_reworked_only_once():
    harness = Harness(statuses={"domain_agent": ["partial"]})
    out = harness.run()
    assert harness.calls.count("domain_agent") == 2 and out["next"] == "FINISH"


def test_perspective_that_keeps_failing_does_not_block_the_report():
    harness = Harness(statuses={"technical_agent": ["error"]})
    out = harness.run()
    assert harness.calls.count("technical_agent") == 2
    assert out["technical_result"]["status"] == "error" and out["report_markdown"] == "# SUMMARY v1"


def test_fail_with_named_perspective_reworks_it_and_regenerates_downstream():
    harness = Harness(verdicts=[verdict(coverage=False, perspectives=["stakeholder"]), verdict(passed=True)])
    out = harness.run()
    assert harness.calls == NORMAL + ["stakeholder_node", "synthesis_agent", "report_agent", "report_evaluator"]
    assert harness.seen["stakeholder_node"][1]["quality_feedback"] == ["[이해관계자] 문제", "[이해관계자] 고친다"]
    second_report = harness.seen["report_agent"][1]["request"]
    assert INSTRUCTION_HEADER in second_report and second_report.endswith("- 문제\n- 고친다")
    assert out["next"] == "FINISH" and out["eval_retry_count"] == 1
    assert out["report_markdown"] == "# SUMMARY v2"


def test_expression_failure_rewrites_only_the_report():
    harness = Harness(verdicts=[verdict(neutrality=False), verdict(passed=True)])
    out = harness.run()
    assert harness.calls == NORMAL + ["report_agent", "report_evaluator"]
    assert INSTRUCTION_HEADER in harness.seen["report_agent"][1]["request"]
    assert out["next"] == "FINISH"


def test_second_fail_ends_with_warning_and_keeps_the_report():
    # 평가가 매번 FAIL이고 매번 다른 관점을 지목해도 재시도는 한 번뿐이다.
    harness = Harness(verdicts=[verdict(coverage=False, perspectives=["market"]),
                                verdict(bias_control=False, perspectives=["domain"])])
    out = harness.run()
    assert harness.calls.count("report_evaluator") == 2 and harness.calls.count("domain_agent") == 1
    assert out["next"] == "END_WARNING" and "bias_control" in out["warning"]
    assert out["report_markdown"] == "# SUMMARY v2"


def test_step_cap_skips_rework_and_still_produces_a_report(monkeypatch):
    monkeypatch.setattr(policy, "MAX_STEPS", 3)
    harness = Harness(verdicts=[verdict(coverage=False)])
    out = harness.run()
    assert harness.calls == ["technical_agent", "market_node", "stakeholder_node",
                             "synthesis_agent", "report_agent", "report_evaluator"]
    assert out["next"] == "END_WARNING" and out["report_markdown"] == "# SUMMARY v1"


def test_judge_choice_changes_the_route():
    class Answer:
        next, reason, instructions = "report_agent", "인용 표기 문제다", []

    harness = Harness(verdicts=[verdict(coverage=False, perspectives=["market"]), verdict(passed=True)])
    harness.run(judge=lambda choice, state: Answer())
    # 코드 기본값은 market_node 재작업이지만 판정이 report_agent를 골랐다.
    assert harness.calls == NORMAL + ["report_agent", "report_evaluator"]


def test_runs_with_a_checkpointer():
    harness = Harness(verdicts=[verdict(coverage=False, perspectives=["market"]), verdict(passed=True)])
    out = harness.run(checkpointer=InMemorySaver())
    assert out["next"] == "FINISH" and out["synthesis_result"]["status"] == "complete"


def test_workers_only_talk_to_the_supervisor():
    harness = Harness()
    drawn = build_supervisor_graph(harness.nodes(), evaluator=harness.evaluator).get_graph()
    targets: dict[str, set[str]] = {}
    for edge in drawn.edges:
        targets.setdefault(edge.source, set()).add(edge.target)
    for name in (*FIELD_OF, "synthesis_agent", "report_evaluator"):
        assert targets[name] == {"supervisor"}
    assert targets["report_agent"] == {"report_evaluator"}
    assert targets["supervisor"] == {*FIELD_OF, "synthesis_agent", "report_agent", "end_with_warning", "__end__"}
    assert targets["end_with_warning"] == {"__end__"}


def test_builder_rejects_a_missing_evaluator_or_node():
    harness = Harness()
    with pytest.raises(ValueError, match="evaluator"):
        build_supervisor_graph(harness.nodes(), evaluator=None)
    nodes = harness.nodes()
    del nodes["domain_agent"]
    with pytest.raises(ValueError, match="domain_agent"):
        build_supervisor_graph(nodes, evaluator=harness.evaluator)
