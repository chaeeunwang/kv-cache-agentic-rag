"""Supervisor 테스트가 함께 쓰는 State 조각. 외부 API와 모델을 쓰지 않는다."""


def result(status="complete", limitations=()):
    return {"status": status, "summary": "요약", "findings": [], "evidence": [], "limitations": list(limitations)}


def verdict(passed=False, groundedness=True, neutrality=True, bias_control=True, coverage=True, perspectives=()):
    return {"passed": passed, "groundedness": groundedness, "neutrality": neutrality, "bias_control": bias_control,
            "coverage": coverage, "issues": ["문제"], "retry_instruction": "고친다", "perspectives": list(perspectives)}


def collected(**overrides):
    """네 관점이 모두 complete인 State. 인자로 일부 필드를 바꾼다."""
    state = {"request": "평가", "technical_result": result(), "market_result": result(),
             "stakeholder_result": result(), "domain_result": result()}
    state.update(overrides)
    return state


def reported(**overrides):
    """종합과 보고서까지 끝난 State."""
    return collected(synthesis_result=result(), report_markdown="# SUMMARY", **overrides)
