"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        # temperature=0.3 -> output ổn định hơn (CODELAB Bài Tập 1.2)
        temperature=float(os.getenv("OPENROUTER_TEMPERATURE", "0.3")),
        # Giới hạn max_tokens để vừa với credit của tài khoản OpenRouter free-tier
        # (mặc định model yêu cầu tới 64000 tokens -> lỗi 402). Có thể chỉnh qua .env.
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "1500")),
    )