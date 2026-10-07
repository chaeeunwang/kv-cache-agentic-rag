"""KV cache 다관점 평가 그래프 실행 진입점."""

import argparse
import importlib
import json
import logging
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from config import settings
from service.agent.graph import build_agent_graph
from service.agent.supervisor import make_judge
from service.agent.tavily.query_templates import CRITERIA as TAVILY_CRITERIA
from service.retrieval.paper_index import get_paper_index
from service.schema.state import GraphState


DEFAULT_DOMAIN = "데이터센터·클라우드 장문맥 LLM 서빙"
DEFAULT_REQUEST = """장문맥 LLM 추론에서 발생하는 KV-cache 메모리·대역폭·데이터 이동 병목을 평가한다.
Human 기반으로 선정된 SW 기술 Multi-head Latent Attention(MLA)와 HW 기술 CXL 기반
Processing-Near-Memory(PNM)를 데이터센터·클라우드 장문맥 LLM 서빙 환경에서 비교한다.
대표 검증 시나리오는 기업 내부의 긴 보고서와 기술 문서 묶음을 활용하는 질의응답 서비스다.

허용된 논문·문서 RAG로 두 기술의 작동 원리, 성능 검증 조건, 도입 제약과 한계를 조사하고,
시장성과 이해관계자는 Tavily 웹 검색으로 평가한다. 논문 사실, 에이전트 해석, 설계상 가설을 구분하며
서로 다른 논문의 배수 수치로 기술 우열을 단정하지 않는다. 공개 정보 기반 TRL은 공인 인증이 아니며
평가 기준일은 2026-09-21이다. 자료에서 확인되지 않는 가격·공급사·채택 현황은 판단을 유보한다."""


# 3번 담당이 제공하는 보고서 품질 평가 노드의 기본 위치. 달라지면 --evaluator로 바꾼다.
DEFAULT_EVALUATOR = "service.agent.evaluation:report_evaluator"
# Supervisor의 스텝 상한(policy.MAX_STEPS) 바깥에서 그래프 전체를 멈추는 최종 안전장치다.
RECURSION_LIMIT = 60


def load_evaluator(path: str):
    """'모듈:함수' 형식의 경로에서 보고서 품질 평가 노드를 불러온다."""
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"평가 노드 경로는 '모듈:함수' 형식이어야 합니다: {path}")
    return getattr(importlib.import_module(module_name), attribute)


def outcome_lines(result: dict) -> list[str]:
    """최종 State에서 종료 방식을 읽어 출력할 문장을 만든다."""
    lines = ["보고서: result/report.md, result/report.pdf" if result.get("report_markdown") else "보고서 미생성"]
    if result.get("warning"):
        lines.append(f"경고 종료: {result['warning']}")
    elif (result.get("eval_result") or {}).get("passed"):
        lines.append("품질 평가 통과")
    rework = sum((result.get("rework_counts") or {}).values())
    lines.append(f"Supervisor 결정 {result.get('step_count', 0)}회, 재작업 {rework}회, "
                 f"FAIL 재시도 {result.get('eval_retry_count', 0)}회")
    return lines


def initial_state(domain: str, trace_id: str = "") -> GraphState:
    """사용자 입력과 고정 평가 기준으로 최초 GraphState를 만든다."""
    return {
        "request": DEFAULT_REQUEST,
        "target_domain": domain,
        "evaluation_criteria": {
            "technical": ["원리", "성능", "실험 조건", "도입 조건", "한계", "TRL"],
            "market": list(TAVILY_CRITERIA["market"]),
            "stakeholder": list(TAVILY_CRITERIA["stakeholder"]),
            "domain": ["문맥 수용", "응답 성능·품질", "비용·전력·운영"],
            "synthesis": ["관점별 일치", "관점별 충돌", "적용 조건", "불확실성", "후속 검증"],
        },
        "technologies": [
            {
                "id": "sw_01",
                "name": "Multi-head Latent Attention (MLA)",
                "approach": "SW",
                "selection_reason": (
                    "DeepSeek-V2에 적용된 어텐션 구조로 Key와 Value를 저차원 잠재 표현으로 압축해 "
                    "KV-cache 저장량을 줄인다. 정량 근거와 함께 기존 모델의 구조 변경, 추가 학습, "
                    "추론 엔진 호환성과 도입 비용을 평가할 수 있어 선정했다."
                ),
            },
            {
                "id": "hw_01",
                "name": "CXL 기반 Processing-Near-Memory (PNM)",
                "approach": "HW",
                "selection_reason": (
                    "CXL 메모리 가까이에서 토큰 페이지 선택과 KV-cache 관련 연산을 수행해 데이터 이동과 "
                    "GPU 병목을 줄인다. 128K~1M 토큰 범위의 평가와 비용·전력 지표, 장치 비율과 대역폭에 "
                    "따른 불리한 조건을 함께 검토할 수 있어 선정했다."
                ),
            },
        ],
        "quality_feedback": [],
        "revision_count": 0,
        # Supervisor 제어 필드
        "step_count": 0,
        "rework_counts": {},
        "eval_retry_count": 0,
        "trace_id": trace_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KV cache 다관점 평가 에이전트 실행")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="평가 대상 도메인")
    parser.add_argument("--max-technical-retries", type=int, choices=range(6), default=2)
    parser.add_argument("--state-output", type=Path, default=Path("result/state.json"))
    parser.add_argument("--evaluator", default=DEFAULT_EVALUATOR, help="보고서 품질 평가 노드의 '모듈:함수' 경로")
    args = parser.parse_args()

    # 외부 호출 전에 필수 키를 확인해 그래프 중간에서 실패하는 것을 막는다.
    missing = [
        name
        for name, value in {
            "OPENAI_API_KEY": settings.openai_api_key,
            "TAVILY_API_KEY": settings.tavily_api_key,
        }.items()
        if not value
    ]
    if missing:
        parser.error("필수 환경변수가 없습니다: " + ", ".join(missing))

    # 평가 노드가 없으면 자동 통과로 대체하지 않고 실행 전에 멈춘다.
    try:
        evaluator = load_evaluator(args.evaluator)
    except (ImportError, AttributeError, ValueError) as exc:
        parser.error(f"보고서 품질 평가 노드를 불러올 수 없습니다({args.evaluator}): {exc}")

    # Supervisor의 결정 이력은 State가 아니라 trace_id가 붙은 로그로 남긴다.
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("supervisor").setLevel(logging.INFO)

    print("FAISS 논문 인덱스를 불러옵니다.", flush=True)
    index = get_paper_index()
    graph = build_agent_graph(index, max_technical_retries=args.max_technical_retries, evaluator=evaluator,
                              judge=make_judge(), checkpointer=InMemorySaver())

    trace_id = uuid.uuid4().hex
    print(f"평가 그래프를 실행합니다. trace_id={trace_id}", flush=True)
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": trace_id},
              "metadata": {"trace_id": trace_id, "pattern": "supervisor"}}
    result = graph.invoke(initial_state(args.domain, trace_id), config=config)

    # 최종 State를 함께 저장해 각 에이전트의 근거와 한계를 추적할 수 있게 한다.
    args.state_output.parent.mkdir(parents=True, exist_ok=True)
    args.state_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"State: {args.state_output}", flush=True)
    for line in outcome_lines(result):
        print(line, flush=True)


if __name__ == "__main__":
    main()
