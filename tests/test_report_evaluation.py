"""보고서 품질 평가(1안 룰베이스 + 2안 LLM Judge) 회귀 테스트. 외부 API를 호출하지 않는다."""

import json

import pytest
from service.agent.evaluation import make_report_evaluator
from service.agent.supervisor.adapters import with_request_instructions
from service.agent.supervisor.graph import build_supervisor_graph
from service.agent.evaluation.judge import JudgeCriterion, JudgeOutput, judge_context
from service.agent.evaluation.rules import (
    check_groundedness,
    check_perspective_coverage,
    claim_units,
    extract_citations,
)
from tests.fixtures import sample_technologies


def evidence(eid: str, url: str = "https://arxiv.org/abs/1") -> dict:
    return {"id": eid, "source_type": "paper", "title": f"title {eid}", "url": url,
            "page": 1, "published_at": None, "excerpt": "excerpt"}


def result(prefix: str, status: str = "complete", technologies=("sw_01", "hw_01")) -> dict:
    return {
        "status": status,
        "summary": f"{prefix} 요약",
        "findings": [{"technology_ids": [tid], "claim": f"{prefix} {tid} 주장",
                      "evidence_ids": [f"{prefix}_{tid}_e1"], "is_inference": False} for tid in technologies],
        "evidence": [evidence(f"{prefix}_{tid}_e1") for tid in ("sw_01", "hw_01")],
        "limitations": [],
    }


GOOD_REPORT = """# SUMMARY
두 기술은 조건별로 다른 특성을 보이며 우열을 판정하지 않는다 [tech_sw_01_e1] [tech_hw_01_e1].

# 1. 평가 목적·대상·범위
평가 대상 기술과 도메인을 정리한다.

# 3. 관점별 비교 평가
## 3.1 기술 성숙도와 TRL
MLA는 KV cache 저장량을 줄이는 결과가 보고되었다 [tech_sw_01_e1].
CXL-PNM은 프로토타입 수준의 실험실 평가가 보고되었다 [tech_hw_01_e1].
## 3.2 시장성
| 기술 | 판단 |
|---|---|
| MLA | 오픈소스 추론 엔진 지원이 확인되었으나 직접 시장 근거는 부족하다 [market_sw_01_e1] |
| CXL-PNM | 인접 시장 성장 전망만 확인되어 직접 판단은 유보한다 [market_hw_01_e1] |
## 3.3 이해관계자 반응
- 개발자 커뮤니티는 KV cache 압축 수단으로 긍정적으로 평가한다 [stake_sw_01_e1].
- 메모리 반도체 기업들이 관련 제품을 개발 중인 것으로 보고된다 [stake_hw_01_e1].
## 3.4 도메인 적용성
장문맥 서빙에서는 추가 하드웨어 없이 적용 가능하나 미세 조정이 필요하다 [domain_sw_01_e1].
CXL 호환 장치와 런타임 등 인프라 투자가 전제된다 [domain_hw_01_e1].

# 4. 관점 간 일치·상충과 적용 조건
모델 전환 가능 여부와 신규 인프라 도입 가능 여부에 따라 적합성이 달라진다 [domain_sw_01_e1] [domain_hw_01_e1].

# REFERENCE
- 논문 A [tech_sw_01_e1] [tech_hw_01_e1]
- 웹 B [market_sw_01_e1] [market_hw_01_e1] [stake_sw_01_e1] [stake_hw_01_e1]
- 논문 C [domain_sw_01_e1] [domain_hw_01_e1]
"""


def make_state(report: str = GOOD_REPORT, **overrides) -> dict:
    state = {
        "request": "비교 평가", "target_domain": "데이터센터 LLM 서빙", "evaluation_criteria": {},
        "technologies": sample_technologies(),
        "technical_result": result("tech"), "market_result": result("market"),
        "stakeholder_result": result("stake"), "domain_result": result("domain"),
        "quality_feedback": [], "revision_count": 1,
        "report_markdown": report, "report_evidence_ids": [],
    }
    state.update(overrides)
    return state


def criterion(score: int, targets=()) -> JudgeCriterion:
    return JudgeCriterion(score=score, violations=["MLA가 더 우수하다."] if score < 3 else [],
                          rationale="채점 사유", fix_directions=["우열 표현을 조건부 비교로 바꾸세요."] if score < 3 else [],
                          retry_targets=list(targets))


class JudgeFixture:
    """호출 순서대로 (neutrality, bias_control) 점수를 돌려주는 Judge 대역."""

    def __init__(self, *scores, fail=False):
        self.scores = list(scores) or [(5, 5)]
        self.fail = fail
        self.messages = []

    def with_structured_output(self, schema):
        assert schema is JudgeOutput
        return self

    def invoke(self, messages):
        self.messages.append(messages)
        if self.fail:
            raise RuntimeError("judge down")
        neutral, bias = self.scores[min(len(self.messages), len(self.scores)) - 1]
        return JudgeOutput(neutrality=criterion(neutral, ["report"]), bias_control=criterion(bias, ["market"]))


# ── 1안 룰베이스 ─────────────────────────────────────────

def test_extract_citations_ignores_links_and_numeric_refs():
    text = "주장 [web_abc123] [sw_p1_c1, hw_p2_c3] 링크 [문서](https://x.org) 각주 [1]"
    assert extract_citations(text) == ["web_abc123", "sw_p1_c1", "hw_p2_c3"]


def test_claim_units_cover_sections_3_and_4_only():
    units = claim_units(GOOD_REPORT.split("# REFERENCE")[0])
    assert all("평가 대상 기술과 도메인" not in unit for unit in units)
    assert not any(unit.startswith("| 기술") for unit in units)  # 표 머리행 제외
    assert len(units) == 9


def test_groundedness_passes_for_grounded_report():
    outcome = check_groundedness(make_state(), GOOD_REPORT, min_ratio=0.7)
    assert outcome.result["passed"], outcome.result["reasons"]
    assert outcome.result["score"] == 1.0


def test_groundedness_fails_on_hallucinated_id():
    report = GOOD_REPORT.replace("[tech_sw_01_e1].\nCXL", "[fake_id_999].\nCXL")
    outcome = check_groundedness(make_state(report), report, min_ratio=0.7)
    assert not outcome.result["passed"]
    assert outcome.details["unknown_ids"] == ["fake_id_999"]
    assert "report" in outcome.retry_targets


def test_groundedness_fails_when_claims_are_uncited():
    report = GOOD_REPORT.replace(" [domain_sw_01_e1].\n", ".\n").replace(" [domain_hw_01_e1].\n", ".\n") \
        .replace(" [stake_sw_01_e1].", ".").replace(" [stake_hw_01_e1].", ".")
    outcome = check_groundedness(make_state(report), report, min_ratio=0.7)
    assert not outcome.result["passed"]
    assert outcome.result["score"] < 0.7
    assert any("인용 비율" in reason for reason in outcome.result["reasons"])


def test_groundedness_fails_when_reference_misses_cited_id():
    report = GOOD_REPORT.replace("- 논문 C [domain_sw_01_e1] [domain_hw_01_e1]", "- 논문 C")
    outcome = check_groundedness(make_state(report), report, min_ratio=0.7)
    assert outcome.details["missing_in_reference"] == ["domain_hw_01_e1", "domain_sw_01_e1"]


def test_groundedness_accepts_bare_ids_in_reference():
    report = GOOD_REPORT.replace("- 논문 C [domain_sw_01_e1] [domain_hw_01_e1]", "- domain_sw_01_e1: 논문 C\n- domain_hw_01_e1: ibid.")
    outcome = check_groundedness(make_state(report), report, min_ratio=0.7)
    assert outcome.details["missing_in_reference"] == []


def test_groundedness_flags_dangling_finding_evidence():
    market = result("market")
    market["findings"][0]["evidence_ids"] = ["ghost_e1"]
    outcome = check_groundedness(make_state(market_result=market), GOOD_REPORT, min_ratio=0.7)
    assert not outcome.result["passed"]
    assert outcome.retry_targets == ["market"]


def test_coverage_passes_when_all_perspectives_present():
    outcome = check_perspective_coverage(make_state(), GOOD_REPORT)
    assert outcome.result["passed"] and outcome.result["score"] == 1.0


@pytest.mark.parametrize("override, target", [
    ({"market_result": None}, "market"),
    ({"stakeholder_result": result("stake", status="error")}, "stakeholder"),
    ({"domain_result": result("domain", technologies=("sw_01",))}, "domain"),
])
def test_coverage_fails_for_missing_or_broken_perspective(override, target):
    outcome = check_perspective_coverage(make_state(**override), GOOD_REPORT)
    assert not outcome.result["passed"]
    assert outcome.retry_targets == [target]


def test_coverage_fails_for_missing_report_section():
    report = GOOD_REPORT.replace("## 3.2 시장성", "## 시장")
    outcome = check_perspective_coverage(make_state(report), report)
    assert outcome.details["missing_sections"] == ["3.2 시장성"]
    assert outcome.retry_targets == ["report"]


# ── 2안 LLM Judge ───────────────────────────────────────

def test_judge_context_lists_only_cited_sources():
    state = make_state()
    state["market_result"]["evidence"][0]["url"] = "https://news.example.com/a"
    context = judge_context(state, GOOD_REPORT)
    assert context["cited_evidence_sources"]["market_sw_01_e1"]["domain"] == "news.example.com"
    assert "signals" not in context  # 확장 지점은 현재 비활성
    assert set(context["cited_evidence_sources"]) == set(extract_citations(GOOD_REPORT.split("# REFERENCE")[0]))


# ── 평가 노드 (Supervisor EvalResult 형식) ───────────────────

EVAL_RESULT_KEYS = {"passed", "groundedness", "neutrality", "bias_control", "coverage",
                    "issues", "retry_instruction", "perspectives"}


def test_evaluator_passes_and_writes_detail_file(tmp_path):
    node = make_report_evaluator(JudgeFixture((5, 4)), output_dir=tmp_path)
    update = node(make_state())
    assert set(update) == {"eval_result"}
    eval_result = update["eval_result"]
    assert set(eval_result) == EVAL_RESULT_KEYS
    assert eval_result["passed"] and all(eval_result[k] for k in ("groundedness", "neutrality", "bias_control", "coverage"))
    assert eval_result["issues"] == [] and eval_result["retry_instruction"] == "" and eval_result["perspectives"] == []
    detail = json.loads((tmp_path / "eval_1.json").read_text(encoding="utf-8"))
    assert detail["criteria"]["neutrality"] == {"passed": True, "method": "llm_judge", "score": 5.0,
                                                "reasons": ["Judge 5/5: 채점 사유"]}
    assert detail["criteria"]["groundedness"]["method"] == "rule"


def test_evaluator_attempt_follows_supervisor_retry_count(tmp_path):
    make_report_evaluator(JudgeFixture(), output_dir=tmp_path)(make_state(eval_retry_count=1))
    assert (tmp_path / "eval_2.json").is_file()


def test_evaluator_fails_on_judge_score_and_keeps_violations_out_of_state(tmp_path):
    eval_result = make_report_evaluator(JudgeFixture((2, 5)), output_dir=tmp_path)(make_state())["eval_result"]
    assert not eval_result["passed"] and not eval_result["neutrality"] and eval_result["bias_control"]
    # 표현 문제는 관점 재작업이 아니라 보고서 재작성 대상이다.
    assert eval_result["perspectives"] == []
    assert eval_result["issues"] == ["우열 표현을 조건부 비교로 바꾸세요."]
    assert eval_result["retry_instruction"].startswith("중립성 미달: Judge 2/5")
    assert "MLA가 더 우수하다." not in json.dumps(eval_result, ensure_ascii=False)
    detail = json.loads((tmp_path / "eval_1.json").read_text(encoding="utf-8"))
    assert detail["details"]["neutrality"]["violations"] == ["MLA가 더 우수하다."]


def test_evaluator_names_perspectives_for_evidence_failures(tmp_path):
    state = make_state(market_result=None)
    eval_result = make_report_evaluator(JudgeFixture((5, 2)), output_dir=tmp_path)(state)["eval_result"]
    assert not eval_result["coverage"] and not eval_result["bias_control"]
    # 커버리지 누락(market)과 Judge가 지목한 편향 관점(market)을 중복 없이 넘긴다.
    assert eval_result["perspectives"] == ["market"]


def test_evaluator_records_judge_failure_as_unmet(tmp_path):
    eval_result = make_report_evaluator(JudgeFixture(fail=True), output_dir=tmp_path)(make_state())["eval_result"]
    assert not eval_result["passed"] and not eval_result["neutrality"]
    assert "Judge 호출 실패(RuntimeError)" in eval_result["retry_instruction"]


# ── 실제 Supervisor 그래프와의 연결 ──────────────────────

class FakeAgents:
    """네 관점·종합·보고서 대역. 보고서 버전을 바꿔 가며 반환한다."""

    def __init__(self, reports):
        self.reports = list(reports)
        self.calls = []

    def worker(self, field, prefix):
        def run(state):
            self.calls.append(field)
            return {field: result(prefix)}
        return run

    def nodes(self):
        def synthesis(state):
            self.calls.append("synthesis")
            return {"synthesis_result": result("tech"), "quality_feedback": []}

        def report(state):
            self.calls.append("report")
            text = self.reports.pop(0) if len(self.reports) > 1 else self.reports[0]
            return {"report_markdown": text, "report_evidence_ids": []}

        return {"technical_agent": self.worker("technical_result", "tech"),
                "market_node": self.worker("market_result", "market"),
                "stakeholder_node": self.worker("stakeholder_result", "stake"),
                "domain_agent": self.worker("domain_result", "domain"),
                "synthesis_agent": synthesis, "report_agent": with_request_instructions(report)}


def run_graph(agents, judge, tmp_path):
    graph = build_supervisor_graph(agents.nodes(), evaluator=make_report_evaluator(judge, output_dir=tmp_path))
    initial = {**{k: v for k, v in make_state().items() if not k.endswith("_result") and k != "report_markdown"},
               "eval_retry_count": 0}
    return graph.invoke(initial, config={"recursion_limit": 60})


def test_supervisor_graph_retries_report_once_then_finishes(tmp_path):
    uncited = GOOD_REPORT.replace("[market_sw_01_e1]", "").replace("[market_hw_01_e1]", "") \
        .replace(" [stake_sw_01_e1]", "").replace(" [stake_hw_01_e1]", "") \
        .replace(" [domain_sw_01_e1].\n장", ".\n장").replace(" [domain_hw_01_e1].\n\n", ".\n\n")
    agents = FakeAgents([uncited, GOOD_REPORT])
    out = run_graph(agents, JudgeFixture(), tmp_path)
    assert out["next"] == "FINISH" and out["eval_result"]["passed"]
    assert out["eval_retry_count"] == 1 and agents.calls.count("report") == 2
    # 표현 문제(인용 누락)는 관점 재작업 없이 보고서만 다시 쓴다.
    assert agents.calls.count("technical_result") == 1
    first = json.loads((tmp_path / "eval_1.json").read_text(encoding="utf-8"))
    assert first["eval_result"]["groundedness"] is False


def test_supervisor_graph_ends_with_warning_after_retry_budget(tmp_path):
    agents = FakeAgents([GOOD_REPORT])
    out = run_graph(agents, JudgeFixture((1, 5)), tmp_path)
    assert out["next"] == "END_WARNING" and out["eval_retry_count"] == 1
    assert agents.calls.count("report") == 2
    assert "neutrality" in out["warning"]
