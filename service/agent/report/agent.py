"""검증된 에이전트 결과로 Markdown과 PDF 보고서를 생성한다."""

import json
import logging
import re

from config import settings
from config.model import get_chat_model
from service.agent.report.pdf import ReportLengthError, save_report_pdf
from service.agent.report.prompt import REPORT_LENGTH_PROMPT, REPORT_SYSTEM_PROMPT, shortening_instruction
from service.schema.state import GraphState

logger = logging.getLogger(__name__)
# PDF 분량 초과 시에만 적용하며 Supervisor의 품질 재시도 예산과는 별개다.
MAX_LENGTH_RETRIES = 2

RESULT_FIELDS = (
    "technical_result",
    "market_result",
    "stakeholder_result",
    "domain_result",
    "synthesis_result",
)

def _build_report_context(state: GraphState) -> dict:
    """보고서 모델에 전달할 기존 입력과 허용 근거 목록을 구성한다."""
    results = {field: state.get(field) for field in RESULT_FIELDS}
    evidence_ids = {
        evidence["id"]
        for result in results.values()
        if result
        for evidence in result["evidence"]
    }
    return {
        "request": state["request"],
        "target_domain": state["target_domain"],
        "technologies": state["technologies"],
        "results": results,
        "allowed_evidence_ids": sorted(evidence_ids),
    }


def _used_evidence_ids(markdown: str, allowed_ids: list[str]) -> list[str]:
    """기존 단일 ID 인용 형식과 정렬 순서를 유지한다."""
    return sorted(
        evidence_id
        for evidence_id in allowed_ids
        if re.search(rf"\[{re.escape(evidence_id)}\]", markdown)
    )


def report_agent(state: GraphState) -> dict:
    """최종 보고서를 작성하고 실제 사용된 근거 ID만 반환한다."""
    model = get_chat_model(max_tokens=settings.report_max_tokens)
    context = _build_report_context(state)

    source_message = (
        "system",
        f"{REPORT_SYSTEM_PROMPT}\n\n{REPORT_LENGTH_PROMPT}\n\n보고서 작성 자료:\n"
        f"{json.dumps(context, ensure_ascii=False)}",
    )
    messages = [source_message]
    for attempt in range(MAX_LENGTH_RETRIES + 1):
        response = model.invoke(messages)
        report_markdown = str(response.content).strip()
        report_evidence_ids = _used_evidence_ids(report_markdown, context["allowed_evidence_ids"])
        try:
            save_report_pdf(report_markdown, report_evidence_ids)
        except ReportLengthError as error:
            logger.warning("보고서 분량 초과: pages=%s, 축약 재시도=%s/%s",
                           error.pages, attempt, MAX_LENGTH_RETRIES)
            if attempt == MAX_LENGTH_RETRIES:
                raise
            # 초안 이력을 누적하지 않고 원자료와 직전 초안, 실측 분량만 전달한다.
            messages = [source_message, ("assistant", report_markdown),
                        ("human", shortening_instruction(error.pages))]
        else:
            return {
                "report_markdown": report_markdown,
                "report_evidence_ids": report_evidence_ids,
            }
