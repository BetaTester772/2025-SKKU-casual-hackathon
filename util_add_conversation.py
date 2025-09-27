from services.rag.conversation_summarizer import summarize_and_store
from services.rag.rag_connecter import get_rag_response

SESSION_USER = "hoseong"
sh_session_group = "family"
sh_transcript = []

from db.session import SessionLocal
from db.models import User


def get_user_id_by_name(name: str) -> int | None:
    """주어진 name에 해당하는 user_id 반환 (없으면 None)"""
    with SessionLocal() as session:
        user = session.query(User).filter(User.name == name).first()
        return user.user_id if user else None


ASR_TEXT = "Hello?"
sh_transcript.append({"role": "user", "content": ASR_TEXT})
stream = get_rag_response(
        user_id=get_user_id_by_name(SESSION_USER),
        query=str(ASR_TEXT),
        target="team",
        group_name=sh_session_group,
)  # openai.Stream
answer = ""
for event in stream:
    if event.type == "response.output_text.delta":
        answer += event.delta
        print(event.delta, end="")  # 실시간 출력
    elif event.type == "response.completed":
        print("\n--- 응답 완료 ---")

sh_transcript.append({"role": "assistant", "content": answer})
print(sh_transcript)

# res = summarize_and_store(me_id=get_user_id_by_name(SESSION_USER), messages=sh_transcript, visibility="group")
# print(f"(conversation saved) text_ref={res['text_ref']} id={res['embedding_id']}")
