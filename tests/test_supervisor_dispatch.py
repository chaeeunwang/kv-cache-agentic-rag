"""Supervisor가 에이전트를 호출할 때 State에 쓰는 값(설계 4.3) 테스트."""
from service.agent.supervisor.policy import FINISH, REPORT, SYNTHESIS, decide, dispatch_update
from tests.supervisor_fixtures import collected, reported, result, verdict


def test_first_call_carries_no_instructions():
    state = {"request": "평가", "technical_result": result()}
    assert dispatch_update(state, decide(state), "market_node", None) == {"quality_feedback": []}


def test_rework_labels_instructions_counts_and_invalidates_downstream():
    state = collected(market_result=result("partial", ["hw_01 상용화 근거 없음"]), synthesis_result=result())
    update = dispatch_update(state, decide(state), "market_node", None)
    assert update == {"quality_feedback": ["[시장] hw_01 상용화 근거 없음"], "rework_counts": {"market": 1},
                      "synthesis_result": None, "report_markdown": None}


def test_rework_prefers_judge_instructions_and_keeps_other_counts():
    state = collected(stakeholder_result=result("partial", ["한계"]), rework_counts={"market": 1})
    update = dispatch_update(state, decide(state), "stakeholder_node", ["투자 업계 반응을 다시 찾는다"])
    assert update["quality_feedback"] == ["[이해관계자] 투자 업계 반응을 다시 찾는다"]
    assert update["rework_counts"] == {"market": 1, "stakeholder": 1}


def test_error_rework_gets_a_retry_instruction():
    state = collected(domain_result=result("error", ["domain_result: RuntimeError"]))
    update = dispatch_update(state, decide(state), "domain_agent", None)
    assert update["quality_feedback"] == ["[도메인] 직전 실행이 오류로 끝났다. 같은 작업을 다시 수행한다."]


def test_fail_to_report_spends_the_retry_and_clears_the_verdict():
    state = reported(eval_result=verdict(neutrality=False))
    update = dispatch_update(state, decide(state), REPORT, None)
    assert update == {"quality_feedback": ["문제", "고친다"], "eval_result": None, "eval_retry_count": 1}


def test_fail_to_perspective_keeps_the_verdict_for_the_next_report():
    state = reported(eval_result=verdict(coverage=False, perspectives=["market"]))
    update = dispatch_update(state, decide(state), "market_node", None)
    assert update["eval_retry_count"] == 1 and update["report_markdown"] is None
    assert update["quality_feedback"] == ["[시장] 문제", "[시장] 고친다"]
    assert "eval_result" not in update

    # 관점 재작업과 종합이 끝난 뒤 보고서를 다시 보낼 때 남겨 둔 판정을 지시로 넘기고 비운다.
    later = collected(synthesis_result=result(), report_markdown=None, eval_retry_count=1,
                      eval_result=verdict(coverage=False, perspectives=["market"]))
    update = dispatch_update(later, decide(later), REPORT, None)
    assert update == {"quality_feedback": ["문제", "고친다"], "eval_result": None}


def test_first_report_and_terminal_decisions_write_nothing_extra():
    state = collected(synthesis_result=result())
    assert dispatch_update(state, decide(state), REPORT, None) == {"quality_feedback": [], "eval_result": None}
    done = reported(eval_result=verdict(passed=True))
    assert dispatch_update(done, decide(done), FINISH, None) == {}
    assert dispatch_update(collected(), decide(collected()), SYNTHESIS, None) == {"quality_feedback": []}
