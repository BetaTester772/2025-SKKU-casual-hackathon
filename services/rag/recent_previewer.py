"""
최근 대화/질의 미리보기 모듈

사용자의 최근 대화 요약 및 질의를 조회하여 미리보기 문자열로 반환합니다.
UI에서 사용자의 최근 활동을 표시하는 데 사용됩니다.

Functions:
    _recent_previews: 최근 N개의 요약/질의 미리보기 반환

반환 형식:
    - "[conv] 대화 요약 내용..."
    - "[query] 질의 내용..."
"""

from __future__ import annotations
import sys
from typing import List


from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from db.models import Embedding


def _recent_previews(db: Session, owner_id: int, n: int = 2) -> List[str]:
    """Return up to n recent summaries/queries (mix of conv:/query:) as short strings."""
    rows = (
        db.execute(
            select(Embedding.text_ref, Embedding.content)
            .where(Embedding.owner_id == owner_id)
            .order_by(desc(Embedding.embedding_id))
            .limit(max(1, n))
        )
        .all()
    )
    out: List[str] = []
    for ref, content in rows:
        prefix = "conv" if str(ref or "").startswith("conv:") else (
        "query" if str(ref or "").startswith("query:") else "other"
        )
        text = (content or ref or "").strip()
        if len(text) > 140:
            text = text[:140] + "…"
        out.append(f"[{prefix}] {text}")
    return out