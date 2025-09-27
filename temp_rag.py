from services.rag.rag_connecter import get_rag_response

# TODO: answer 그대로 유저에게 표시(TTS)
answer = get_rag_response(
    user_id=me_id, #TODO: int type. dict로 중간 변환
    query="오늘 7시에 오피스 미팅 있나?", # TODO: 쿼리
    target="team",
    group_name="Demo Team",  # 얼굴 인식으로 받은 그룹명 TODO: str type
)

