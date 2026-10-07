"""선택지가 둘 이상인 지점에서 쓰는 LLM 구조화 판정."""

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from service.agent.supervisor.policy import LABEL_OF, NODE_OF, PERSPECTIVES, RESULT_OF, Choice

MAX_LIMITATIONS = 5

JUDGE_SYSTEM_PROMPT = """당신은 KV cache 기술 평가 그래프의 Supervisor다.
직접 조사하거나 글을 쓰지 않고, 다음에 실행할 노드를 candidates 안에서 하나만 고른다.

kind가 sufficiency이면 네 관점의 근거가 종합으로 넘어가기에 충분한지 판단한다.
- 충분하면 synthesis_agent를 고르고 instructions는 빈 목록으로 둔다.
- 다시 조사하면 실제로 채울 수 있는 누락이 있으면 그 관점의 노드를 고르고, 무엇을 다시 찾거나 고칠지 instructions에 구체적으로 적는다.
- limitations가 "이번 검색에서 확인되지 않음"처럼 다시 찾아도 채우기 어려운 공개 정보의 한계라면 부족으로 보지 않는다.
- status가 partial이라는 이유만으로 재작업을 고르지 않는다.

kind가 fail_analysis이면 eval_result의 미달 항목과 issues를 읽고 원인이 어디에 있는지 판단한다.
- 표현, 구성, 인용 표기처럼 보고서를 다시 쓰면 해결되는 문제면 report_agent를 고른다.
- 근거가 없거나 한쪽으로 치우쳤거나 특정 관점이 빠진 문제면 그 관점의 노드를 고르고 보강할 내용을 instructions에 적는다.

next에는 candidates에 있는 문자열을 그대로 적는다. 입력에 없는 사실을 지어내지 않는다. 한국어로 쓴다."""


class JudgeOutput(BaseModel):
    next: str = Field(description="candidates 중 하나를 그대로 적는다")
    reason: str = Field(description="그 후보를 고른 근거를 한두 문장으로 적는다")
    instructions: list[str] = Field(description="재작업 대상에게 줄 구체적 지시. 진행을 고르면 빈 목록")


def build_digest(state: Mapping[str, Any]) -> dict:
    """판정에 필요한 만큼만 State를 요약한다. 근거 원문은 넣지 않는다."""
    perspectives = {}
    for perspective in PERSPECTIVES:
        result = state.get(RESULT_OF[perspective]) or {}
        perspectives[perspective] = {
            "node": NODE_OF[perspective], "label": LABEL_OF[perspective], "status": result.get("status"),
            "limitations": list(result.get("limitations") or [])[:MAX_LIMITATIONS],
            "findings": len(result.get("findings") or []), "evidence": len(result.get("evidence") or []),
            "rework_count": (state.get("rework_counts") or {}).get(perspective, 0),
        }
    digest: dict[str, Any] = {"perspectives": perspectives}
    if state.get("eval_result"):
        digest["eval_result"] = state["eval_result"]
    return digest


def make_judge(model=None):
    """판정 함수를 만든다. model을 주지 않으면 첫 호출 때 공용 채팅 모델을 만든다."""
    structured = None

    def judge(choice: Choice, state: Mapping[str, Any]) -> JudgeOutput:
        nonlocal structured
        if structured is None:
            if model is None:
                from config.model import get_chat_model
                base = get_chat_model()
            else:
                base = model
            structured = base.with_structured_output(JudgeOutput)
        payload = {"kind": choice.kind, "candidates": list(choice.candidates), "situation": choice.reason,
                   **build_digest(state)}
        return structured.invoke([("system", JUDGE_SYSTEM_PROMPT), ("human", json.dumps(payload, ensure_ascii=False))])

    return judge
