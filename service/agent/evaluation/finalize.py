"""재시도 후에도 평가 미달이면 보고서를 유지하고 한계·Fail 분석 절을 덧붙인다."""

import logging
import re

from service.agent.evaluation.rules import HEADING, REQUIRED_SECTIONS
from service.agent.report.pdf import save_report_pdf
from service.schema.state import GraphState

logger = logging.getLogger(__name__)

SECTION_TITLE = "품질 평가 결과 및 한계"
CRITERIA_LABELS = {
    "groundedness": ("Groundedness", "1안 룰베이스"),
    "neutrality": ("중립성", "2안 LLM Judge"),
    "bias_control": ("편향 통제", "2안 LLM Judge"),
    "perspective_coverage": ("관점 커버리지", "1안 룰베이스"),
}
TARGET_LABELS = {
    "report": "보고서 생성",
    "technical": "기술 조사",
    "market": "시장성 평가",
    "stakeholder": "이해관계자 평가",
    "domain": "도메인 평가",
}


def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("|", "/")).strip()


def failure_section(state: GraphState) -> str:
    verdict = state["eval_result"]
    rows = []
    for name, result in verdict["criteria"].items():
        label, method = CRITERIA_LABELS[name]
        score = "-" if result["score"] is None else f"{result['score']:g}"
        status = "충족" if result["passed"] else "미달"
        rows.append(f"| {label} | {method} | {status} | {score} | {_cell('; '.join(result['reasons'])) or '-'} |")
    targets = ", ".join(TARGET_LABELS.get(t, t) for t in verdict["retry_targets"]) or "-"
    lines = [
        f"# {SECTION_TITLE}",
        "",
        f"이 보고서는 품질 평가를 {verdict['attempt']}회 수행하고 재작업 한도에 도달했으나 일부 항목이 기준에 미달한 상태로 생성되었습니다. "
        "아래 미달 항목에 해당하는 내용은 제한적으로 해석해야 합니다.",
        "",
        "| 평가 항목 | 방식 | 판정 | 점수 | 사유 |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "## Fail 분석",
        f"- 재작업 권고 단계: {targets}",
        *[f"- {_cell(item)}" for item in verdict["feedback"]],
        "- 권고 단계만 선택적으로 재실행하는 특정 노드 재시도는 후속 확장 과제입니다.",
    ]
    return "\n".join(lines)


def insert_before_reference(report: str, section: str) -> str:
    """REFERENCE 앞에 절을 넣어 PDF의 참고문헌 서식과 섞이지 않게 한다."""
    lines = report.splitlines()
    for i, line in enumerate(lines):
        heading = HEADING.match(line)
        if heading and REQUIRED_SECTIONS[-1][1].match(heading.group(2)):
            return "\n".join([*lines[:i], section, "", *lines[i:]])
    return f"{report.rstrip()}\n\n{section}\n"


def make_finalize_eval(writer=save_report_pdf):
    """최종 미달 처리 노드를 만든다. writer는 Markdown·PDF 저장 함수다."""

    def finalize_eval(state: GraphState) -> dict:
        report = insert_before_reference(state.get("report_markdown") or "", failure_section(state))
        try:
            writer(report, state.get("report_evidence_ids", []))
        except FileNotFoundError as error:
            # 한국어 폰트가 없어 PDF만 실패한 경우 Markdown은 이미 저장되었으므로 종료를 막지 않는다.
            logger.warning("PDF 저장 생략: %s", error)
        logger.info("report_evaluation finalized with limitations attempt=%s", state["eval_result"]["attempt"])
        return {"report_markdown": report}

    return finalize_eval
