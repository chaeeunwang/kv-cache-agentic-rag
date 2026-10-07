"""1안 룰베이스 평가: Groundedness와 관점 커버리지를 LLM 없이 결정적으로 판정한다.

형식 검사만 수행한다. 인용 ID가 실제 수집 근거인지, 주장 단위에 인용이 붙었는지,
네 관점 결과가 State에 존재하는지를 보며 인용 내용이 주장을 뒷받침하는지는 판정하지 않는다.
"""

import re
from dataclasses import dataclass, field

from typing import Literal, TypedDict

from service.schema.state import GraphState


class CriterionResult(TypedDict):
    """항목별 상세 판정. State에는 통과 여부(bool)만 넣고, 점수·사유는 상세 파일에 저장한다."""

    passed: bool
    method: Literal["rule", "llm_judge"]
    score: float | None
    reasons: list[str]

# State의 관점 결과 필드 → 재작업 권고 대상 이름
PERSPECTIVES: dict[str, str] = {
    "technical_result": "technical",
    "market_result": "market",
    "stakeholder_result": "stakeholder",
    "domain_result": "domain",
}
PERSPECTIVE_LABELS = {
    "technical": "기술 성숙도",
    "market": "시장성",
    "stakeholder": "이해관계자",
    "domain": "도메인 적용",
}
# 평가 종합 결과는 앞선 관점의 근거를 그대로 보존하므로 허용 근거 집합에만 포함한다.
EVIDENCE_FIELDS = (*PERSPECTIVES, "synthesis_result")

# 보고서 필수 목차: (표시 이름, 제목 판정 정규식)
REQUIRED_SECTIONS = (
    ("SUMMARY", re.compile(r"^SUMMARY\b", re.I)),
    ("3.1 기술 성숙도", re.compile(r"^3\.1\b")),
    ("3.2 시장성", re.compile(r"^3\.2\b")),
    ("3.3 이해관계자", re.compile(r"^3\.3\b")),
    ("3.4 도메인 적용", re.compile(r"^3\.4\b")),
    ("REFERENCE", re.compile(r"^REFERENCES?\b", re.I)),
)

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# [ID] 또는 [ID1, ID2]. Markdown 링크 [text](url)는 제외한다.
CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_\-]*(?:\s*[,;]\s*[A-Za-z0-9][A-Za-z0-9_\-]*)*)\](?!\()")
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}")
# 주장 단위로 보기 위한 최소 길이(Markdown 기호 제거 후). 짧은 표 칸·라벨은 제외한다.
MIN_CLAIM_CHARS = 20
# 근거 비율 계산 대상: 관점별 비교(3장)와 관점 간 종합(4장)
CLAIM_SECTION = re.compile(r"^[34](\.|\s|$)")


@dataclass
class RuleOutcome:
    result: CriterionResult
    feedback: list[str] = field(default_factory=list)
    retry_targets: list[str] = field(default_factory=list)
    # 외부 상세 파일에만 저장하는 진단 정보
    details: dict = field(default_factory=dict)


def extract_citations(text: str) -> list[str]:
    """본문에 표기된 근거 ID를 등장 순서대로 반환한다. 밑줄이 없는 [1] 같은 표기는 근거 ID로 보지 않는다."""
    ids = []
    for match in CITATION.finditer(text):
        for token in re.split(r"\s*[,;]\s*", match.group(1)):
            if "_" in token:
                ids.append(token)
    return ids


def split_reference(markdown: str) -> tuple[str, str]:
    """REFERENCE 제목을 기준으로 본문과 참고문헌을 나눈다."""
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        heading = HEADING.match(line)
        if heading and REQUIRED_SECTIONS[-1][1].match(heading.group(2)):
            return "\n".join(lines[:i]), "\n".join(lines[i + 1:])
    return markdown, ""


def headings(markdown: str) -> list[str]:
    return [m.group(2) for line in markdown.splitlines() if (m := HEADING.match(line))]


def claim_units(body: str) -> list[str]:
    """3~4장에서 주장 단위(문단 줄, 목록 항목, 표 데이터 행)를 추출한다."""
    units, in_section = [], False
    lines = body.splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        heading = HEADING.match(line)
        if heading:
            # 번호가 있는 제목은 장 번호로 구간을 정하고, 번호 없는 최상위 제목(SUMMARY 등)은 구간을 닫는다.
            # 번호 없는 하위 제목은 상위 장을 따른다.
            title = heading.group(2)
            if title[:1].isdigit():
                in_section = bool(CLAIM_SECTION.match(title))
            elif len(heading.group(1)) == 1:
                in_section = False
            continue
        if not in_section or not line or TABLE_SEPARATOR.match(line):
            continue
        # 표 머리행(다음 줄이 구분선)은 주장이 아니다.
        if line.startswith("|") and i + 1 < len(lines) and TABLE_SEPARATOR.match(lines[i + 1].strip()):
            continue
        plain = re.sub(r"[|*_`>#\-]", "", CITATION.sub("", line)).strip()
        if len(plain) >= MIN_CLAIM_CHARS:
            units.append(line)
    return units


def allowed_evidence_ids(state: GraphState) -> set[str]:
    return {
        evidence["id"]
        for name in EVIDENCE_FIELDS
        if (result := state.get(name))
        for evidence in result.get("evidence", [])
    }


def check_groundedness(state: GraphState, report: str, min_ratio: float) -> RuleOutcome:
    """주장이 실제 검색 출처로 추적되는지 형식 검사한다."""
    body, reference = split_reference(report)
    allowed = allowed_evidence_ids(state)
    cited = extract_citations(body)
    cited_set = set(cited)

    reasons, feedback, targets = [], [], []
    unknown = sorted(cited_set - allowed)
    if unknown:
        reasons.append(f"수집되지 않은 근거 ID 인용 {len(unknown)}건")
        feedback.append("보고서가 수집 근거에 없는 ID를 인용했습니다. 허용 근거 ID만 사용하고 해당 주장을 삭제하거나 실제 근거로 교체하세요: "
                        + ", ".join(unknown[:5]))
        targets.append("report")
    if not cited_set:
        reasons.append("본문에 근거 ID 인용이 없음")
        feedback.append("본문 주장에 [근거ID] 인용을 붙이세요.")
        targets.append("report")

    units = claim_units(body)
    uncited = [unit for unit in units if not extract_citations(unit)]
    ratio = 1.0 if not units else (len(units) - len(uncited)) / len(units)
    if ratio < min_ratio:
        reasons.append(f"3~4장 주장 인용 비율 {ratio:.0%} < 기준 {min_ratio:.0%}")
        feedback.append(f"3~4장 주장 {len(uncited)}건에 근거 인용이 없습니다. 근거를 붙이거나 근거 부족을 명시하세요.")
        targets.append("report")

    # REFERENCE는 'ID: 서지' 목록처럼 대괄호 없이 쓰는 경우가 있어 ID 문자열 포함 여부로 판정한다.
    missing_in_reference = sorted(eid for eid in cited_set & allowed if eid not in reference)
    if missing_in_reference:
        reasons.append(f"REFERENCE에 없는 본문 인용 {len(missing_in_reference)}건")
        feedback.append("본문에서 인용한 근거 ID를 REFERENCE에 모두 수록하세요: " + ", ".join(missing_in_reference[:5]))
        targets.append("report")

    # 관점 결과의 finding도 자기 결과의 evidence만 참조해야 보고서까지 출처가 이어진다.
    dangling: dict[str, list[str]] = {}
    for name, perspective in PERSPECTIVES.items():
        result = state.get(name)
        if not result:
            continue
        own = {evidence["id"] for evidence in result.get("evidence", [])}
        bad = sorted({eid for f in result.get("findings", []) for eid in f.get("evidence_ids", []) if eid not in own})
        if bad:
            dangling[perspective] = bad
            reasons.append(f"{PERSPECTIVE_LABELS[perspective]} finding의 미확인 근거 ID {len(bad)}건")
            feedback.append(f"{PERSPECTIVE_LABELS[perspective]} 평가의 finding이 자기 evidence에 없는 ID를 참조합니다.")
            targets.append(perspective)

    return RuleOutcome(
        result={"passed": not reasons, "method": "rule", "score": round(ratio, 3), "reasons": reasons},
        feedback=feedback,
        retry_targets=targets,
        details={
            "cited_ids": sorted(cited_set),
            "unknown_ids": unknown,
            "missing_in_reference": missing_in_reference,
            "claim_units": len(units),
            "uncited_units": uncited,
            "dangling_finding_ids": dangling,
        },
    )


def check_perspective_coverage(state: GraphState, report: str) -> RuleOutcome:
    """4개 관점 결과가 State에 있고 보고서 목차에 반영되었는지 형식 검사한다."""
    technology_ids = {technology["id"] for technology in state.get("technologies", [])}
    reasons, feedback, targets, details = [], [], [], {}
    perspective_failures = 0

    for name, perspective in PERSPECTIVES.items():
        label = PERSPECTIVE_LABELS[perspective]
        result = state.get(name)
        if not result:
            problem = "State에 결과 없음"
        elif result.get("status") == "error":
            problem = "결과 status=error"
        elif not result.get("findings"):
            problem = "finding 없음"
        else:
            covered = {tid for f in result["findings"] for tid in f.get("technology_ids", [])}
            missing = sorted(technology_ids - covered)
            problem = f"기술 미포함({', '.join(missing)})" if missing else None
        details[perspective] = problem or "ok"
        if problem:
            perspective_failures += 1
            reasons.append(f"{label}: {problem}")
            feedback.append(f"{label} 관점 평가를 보완하세요({problem}). 두 기술을 모두 다뤄야 합니다.")
            targets.append(perspective)

    titles = headings(report)
    missing_sections = [label for label, pattern in REQUIRED_SECTIONS if not any(pattern.match(t) for t in titles)]
    if missing_sections:
        reasons.append("보고서 필수 목차 누락: " + ", ".join(missing_sections))
        feedback.append("보고서에 필수 목차를 추가하세요: " + ", ".join(missing_sections))
        targets.append("report")
    details["missing_sections"] = missing_sections

    # 점수 = 통과한 검사 비율 (관점 4개 + 필수 목차 6개)
    checks = len(PERSPECTIVES) + len(REQUIRED_SECTIONS)
    failed = perspective_failures + len(missing_sections)
    return RuleOutcome(
        result={"passed": not reasons, "method": "rule", "score": round(1 - failed / checks, 3), "reasons": reasons},
        feedback=feedback,
        retry_targets=targets,
        details=details,
    )
