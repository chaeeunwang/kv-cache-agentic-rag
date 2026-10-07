"""여러 에이전트가 공유하는 채팅 모델 생성 함수."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config import settings


def get_chat_model(max_tokens: int = 4096) -> BaseChatModel:
    """환경 설정을 사용해 기본 LLM 클라이언트를 생성한다."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        max_tokens=max_tokens,
    )


def get_judge_model() -> BaseChatModel:
    """보고서 품질 평가용 Judge 모델. 재현성을 위해 temperature를 0으로 고정한다."""
    return ChatOpenAI(
        model=settings.judge_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=4096,
    )
