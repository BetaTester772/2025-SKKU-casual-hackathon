"""
대화 요약 및 저장 모듈

대화 세션 종료 시 대화 내용을 LLM으로 요약하고,
임베딩으로 변환하여 벡터 DB에 저장합니다.

요약 포함 내용:
    - 대화 목적
    - 핵심 사실
    - 결정 사항
    - 담당자 및 기한
    - 다음 행동 항목

Functions:
    summarize_and_store: 대화 요약 생성 및 DB 저장
        Returns: {'embedding_id', 'text_ref', 'summary_preview'}

Usage:
    from services.rag.conversation_summarizer import summarize_and_store
    result = summarize_and_store(me_id=1, messages=transcript, visibility="group")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Union
from uuid import uuid4

from services.embeddings.generator import get_embedder
from services.rag.llm_client import complete as llm_complete
from services.rag.pgvector_adapter import PgVectorEmbeddingsStore
from db.session import SessionLocal
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

_SYSTEM_SUMMARY = (
        "You are an expert note-taker. Summarize the conversation clearly. "
        "Include: purpose, key facts, decisions, owners & due dates, next actions. "
        "Keep it concise (<= 12 lines). Use bullet points when helpful."
)


def _format_messages(messages: Union[str, Sequence[Dict[str, str]]]) -> str:
    if isinstance(messages, str):
        return messages
    lines: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip()
        content = (m.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def summarize_and_store(
        *,
        me_id: int,
        messages: Union[str, Sequence[Dict[str, str]]],
        visibility: str = "group",
        session_factory=SessionLocal,
        text_ref_prefix: str = "conv:",
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 512,
) -> Dict[str, str]:
    if visibility not in ("self", "group"):
        visibility = "group"

    transcript = _format_messages(messages)
    prompt = (
            f"Summarize the following conversation.\n\n"  # user content
            f"Conversation (UTC):\n{transcript}\n\n"
            f"Output:"
    )
    summary = llm_complete(model=model_name, prompt=prompt, system=_SYSTEM_SUMMARY, max_tokens=max_tokens)
    if not summary or not summary.strip():
        summary = "(no summary)"

    # 2) Embed summary
    embed_fn = get_embedder()
    vec = embed_fn(summary)

    # 3) Store via PgVectorEmbeddingsStore (content = summary)
    store = PgVectorEmbeddingsStore(session_factory)
    text_ref = f"{text_ref_prefix}{me_id}:{_now_iso()}:{uuid4().hex}"
    node = TextNode(
            text=summary,
            metadata={
                    "owner_id"  : me_id,
                    "visibility": visibility,
                    "text_ref"  : text_ref,
            },
            embedding=vec,
    )
    ids = store.add([node])
    embedding_id = ids[0]

    logger.info(
            "[conv.summary] owner=%s visibility=%s text_ref=%s id=%s chars=%s",
            me_id, visibility, text_ref, embedding_id, len(summary),
    )

    return {
            "embedding_id"   : embedding_id,
            "text_ref"       : text_ref,
            "summary_preview": summary[:180],
    }