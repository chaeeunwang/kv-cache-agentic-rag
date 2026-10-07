"""Supervisor 결정 규칙 테스트. dict State만 쓰고 LLM을 호출하지 않는다."""
import pytest

from service.agent.supervisor import policy
from service.agent.supervisor.policy import END_WARNING, FINISH, REPORT, SYNTHESIS, decide, read_verdict
from tests.supervisor_fixtures import collected, reported, result, verdict


# 규칙 5: 결과가 없는 관점부터 수집한다.
def test_starts_with_technical_when_nothing_is_collected():
    choice = decide({"request": "평가"})
    assert choice.candidates == ("technical_agent",) and choice.kind == "forced"


@pytest.mark.parametrize("present, expected", [
    (["technical"], "market_node"),
    (["technical", "market"], "stakeholder_node"),
    (["technical", "market", "stakeholder"], "domain_agent"),
    # 시작 State에 이미 있는 관점은 건너뛰므로 State가 다르면 경로가 달라진다.
    (["technical", "domain"], "market_node"),
    (["market", "stakeholder", "domain"], "technical_agent"),
])
def test_collects_only_missing_perspectives(present, expected):
    state = {"request": "평가", **{f"{name}_result": result() for name in present}}
    assert decide(state).default == expected


# 규칙 6: 오류로 끝난 관점은 한 번만 다시 호출한다.
def test_error_result_is_retried_while_budget_remains():
    state = collected(market_result=result("error"))
    assert decide(state).candidates == ("market_node",)


def test_error_result_is_accepted_after_budget_is_spent():
    state = collected(market_result=result("error"), rework_counts={"market": 1})
    assert decide(state).candidates == (SYNTHESIS,)


# 규칙 7: 네 관점이 모이면 근거 충분성을 판정한다.
def test_all_complete_goes_straight_to_synthesis():
    choice = decide(collected())
    assert choice.candidates == (SYNTHESIS,) and choice.kind == "forced"


def test_partial_perspective_opens_sufficiency_choice():
    state = collected(market_result=result("partial"), domain_result=result("partial"))
    choice = decide(state)
    assert choice.kind == "sufficiency"
    assert choice.candidates == (SYNTHESIS, "market_node", "domain_agent")
    assert choice.default == "market_node"


def test_partial_perspective_without_budget_is_not_a_candidate():
    state = collected(market_result=result("partial"), rework_counts={"market": 1})
    assert decide(state).candidates == (SYNTHESIS,)


def test_no_extra_rework_during_fail_retry():
    state = collected(market_result=result("partial"), eval_retry_count=1)
    assert decide(state).candidates == (SYNTHESIS,)


# 규칙 8
def test_report_follows_synthesis():
    assert decide(collected(synthesis_result=result())).candidates == (REPORT,)


# 규칙 1, 3, 4, 9: 보고서 뒤의 결정
def test_pass_verdict_finishes():
    assert decide(reported(eval_result=verdict(passed=True))).candidates == (FINISH,)


def test_pass_verdict_finishes_even_at_step_cap():
    state = reported(eval_result=verdict(passed=True), step_count=policy.MAX_STEPS)
    assert decide(state).candidates == (FINISH,)


def test_neutrality_only_failure_rewrites_report():
    choice = decide(reported(eval_result=verdict(neutrality=False)))
    assert choice.candidates == (REPORT,) and choice.kind == "forced"


def test_evidence_failure_with_named_perspective_offers_that_agent():
    choice = decide(reported(eval_result=verdict(coverage=False, perspectives=["stakeholder"])))
    assert choice.kind == "fail_analysis"
    assert choice.candidates == (REPORT, "stakeholder_node")
    assert choice.default == "stakeholder_node"


def test_evidence_failure_without_named_perspective_offers_every_agent():
    choice = decide(reported(eval_result=verdict(bias_control=False)))
    assert choice.candidates == (REPORT, "technical_agent", "market_node", "stakeholder_node", "domain_agent")
    assert choice.default == REPORT


def test_fail_after_retry_ends_with_warning():
    state = reported(eval_result=verdict(coverage=False), eval_retry_count=1)
    assert decide(state).candidates == (END_WARNING,)


def test_report_without_verdict_ends_with_warning():
    assert decide(reported()).candidates == (END_WARNING,)


# 규칙 2: 스텝 상한
def test_step_cap_with_report_ends_with_warning():
    state = reported(eval_result=verdict(coverage=False), step_count=policy.MAX_STEPS)
    assert decide(state).candidates == (END_WARNING,)


def test_step_cap_without_report_goes_straight_to_report():
    capped = {"step_count": policy.MAX_STEPS}
    assert decide(collected(market_result=result("partial"), **capped)).candidates == (SYNTHESIS,)
    assert decide(collected(synthesis_result=result(), **capped)).candidates == (REPORT,)
    # 수집이 덜 끝났어도 더 조사하지 않고 있는 결과로 보고서까지 간다.
    assert decide({"request": "평가", "technical_result": result(), **capped}).candidates == (SYNTHESIS,)


# 평가 판정 읽기: 형태가 어긋나도 예외 없이 읽는다.
def test_read_verdict_tolerates_missing_and_unknown_fields():
    parsed = read_verdict({"passed": False, "coverage": False, "perspectives": ["market", "report", "시장"]})
    assert parsed.passed is False and parsed.evidence_related is True
    assert parsed.perspectives == ("market",)
    assert parsed.instructions == ()
    assert read_verdict({}).passed is False and read_verdict({}).evidence_related is False
    assert read_verdict({"passed": True, "perspectives": None, "issues": None}).passed is True


def test_verdict_with_only_passed_flag_still_routes():
    assert decide(reported(eval_result={"passed": True})).candidates == (FINISH,)
    assert decide(reported(eval_result={"passed": False})).candidates == (REPORT,)
