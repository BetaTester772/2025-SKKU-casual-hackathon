"""
LLM API 클라이언트 모듈

OpenAI API 또는 로컬 스텁을 통해 텍스트 완성을 수행합니다.
환경변수 OPENAI_API_KEY 설정 여부에 따라 자동으로 프로바이더를 선택합니다.

Functions:
    complete: 고수준 완성 진입점 (자동 프로바이더 선택)
    _openai_complete: OpenAI API 호출
    _local_stub_complete: 로컬 스텁 (테스트용)

Environment Variables:
    OPENAI_API_KEY: OpenAI API 키 (필수, OpenAI 사용 시)
    OPENAI_API_BASE: OpenAI API 베이스 URL (선택)

Usage:
    from services.rag.llm_client import complete
    answer = complete("gpt-4o-mini", "질문 내용", system="시스템 프롬프트")
"""

from __future__ import annotations

import os
from typing import Optional

_DEFAULT_SYSTEM = (
    "You are a concise, helpful assistant. Cite key facts from the given context when useful."
)


def _openai_complete(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("openai package not installed. `pip install openai`.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAI provider.")

    base_url = os.getenv("OPENAI_API_BASE") 
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    messages = []
    sys_text = system or _DEFAULT_SYSTEM
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _local_stub_complete(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Deterministic offline fallback: returns a short, safe summary-like string.
    This is NOT an LLM?it's only for wiring tests when no API key is available.
    """
    # extract last user question heuristically
    tail = prompt.splitlines()[-1].strip()
    if len(tail) < 8:
        tail = (prompt[-120:] if len(prompt) > 120 else prompt)
    out = (
        "[local-stub] Based on the provided context, here is a concise answer:\n"
        f"- Key question: {tail[:200]}\n"
        "- Note: This is a non-LLM placeholder. Configure OPENAI_API_KEY to enable real completions."
    )
    # enforce max_tokens approximately by characters (~4 chars per token rough)
    max_chars = max(40, int(max_tokens * 4))
    return out[:max_chars]


def complete(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """High-level completion entry point.

    Chooses provider automatically:
    - If `OPENAI_API_KEY` is set → OpenAI
    - Else → local stub
    """
    if os.getenv("OPENAI_API_KEY"):
        return _openai_complete(
            model,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _local_stub_complete(
        model,
        prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )
