"""실행 진입점의 보조 함수 테스트. 그래프를 실행하지 않는다."""
import pytest

from run_agent import DEFAULT_DOMAIN, initial_state, load_evaluator, outcome_lines


def sample_evaluator(state):
    return {"eval_result": {"passed": True}}


def test_load_evaluator_resolves_a_module_function_path():
    assert load_evaluator("tests.test_run_agent:sample_evaluator") is sample_evaluator


@pytest.mark.parametrize("path", ["service.agent.evaluation", ":report_evaluator", ""])
def test_load_evaluator_rejects_a_malformed_path(path):
    with pytest.raises(ValueError, match="모듈:함수"):
        load_evaluator(path)


def test_load_evaluator_reports_a_missing_module_or_function():
    with pytest.raises(ImportError):
        load_evaluator("service.agent.no_such_module:report_evaluator")
    with pytest.raises(AttributeError):
        load_evaluator("tests.test_run_agent:no_such_function")


def test_initial_state_starts_the_control_fields():
    state = initial_state(DEFAULT_DOMAIN, "abc123")
    assert (state["step_count"], state["rework_counts"], state["eval_retry_count"]) == (0, {}, 0)
    assert state["trace_id"] == "abc123" and state["quality_feedback"] == []
    assert "next" not in state and "report_markdown" not in state


def test_outcome_lines_distinguish_pass_warning_and_missing_report():
    passed = outcome_lines({"report_markdown": "# SUMMARY", "eval_result": {"passed": True}, "step_count": 7})
    assert passed == ["보고서: result/report.md, result/report.pdf", "품질 평가 통과",
                      "Supervisor 결정 7회, 재작업 0회, FAIL 재시도 0회"]
    warned = outcome_lines({"report_markdown": "# SUMMARY", "warning": "스텝 상한(20)에 도달했다.",
                            "rework_counts": {"market": 1, "domain": 1}, "eval_retry_count": 1, "step_count": 20})
    assert warned[1] == "경고 종료: 스텝 상한(20)에 도달했다."
    assert warned[2] == "Supervisor 결정 20회, 재작업 2회, FAIL 재시도 1회"
    assert outcome_lines({})[0] == "보고서 미생성"
