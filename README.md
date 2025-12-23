# Fam_iso - Family Shared Voice AI Assistant

가족 구성원 간 정보를 공유하고 음성으로 소통할 수 있는 AI 어시스턴트입니다. 얼굴 인식을 통해 사용자를 식별하고, 음성 명령을 RAG(Retrieval-Augmented Generation) 기반으로 처리하여 가족 그룹 단위의 정보를 공유합니다.

## 주요 기능

- **얼굴 인식 기반 사용자 식별**: FaceNet + MediaPipe를 활용한 실시간 얼굴 인식
- **음성 활동 감지 (VAD)**: Silero VAD를 통한 음성 구간 자동 감지
- **음성 인식 (ASR)**: OpenAI Whisper를 활용한 음성-텍스트 변환
- **RAG 기반 대화**: 가족 구성원의 과거 대화/일정을 검색하여 맥락에 맞는 응답 생성
- **실시간 TTS**: Kokoro TTS를 통한 스트리밍 음성 합성
- **가족 브리핑**: 가족 구성원들의 최근 소식을 40-60초 분량으로 요약 재생

## 기술 스택

| 분류 | 기술 |
|------|------|
| **UI** | Streamlit |
| **얼굴 인식** | FaceNet (InceptionResnetV1), MediaPipe |
| **음성 인식** | OpenAI Whisper |
| **음성 활동 감지** | Silero VAD |
| **TTS** | Kokoro TTS |
| **LLM** | OpenAI GPT API |
| **벡터 DB** | PostgreSQL + pgvector |
| **ORM** | SQLAlchemy |
| **마이그레이션** | Alembic |

## 프로젝트 구조

```
2025-SKKU-casual-hackathon/
├── app.py                      # 메인 애플리케이션 (Streamlit UI + 상태머신)
├── pyproject.toml              # 프로젝트 의존성 정의
├── requirements.txt            # pip 의존성 목록
├── alembic.ini                 # Alembic 마이그레이션 설정
├── uv.lock                     # uv 패키지 매니저 락 파일
│
├── assets/                     # 정적 리소스
│   ├── dad.png                 # 가족 사진 (아빠)
│   ├── mom.png                 # 가족 사진 (엄마)
│   ├── bro.png                 # 가족 사진 (형)
│   ├── me.png                  # 가족 사진 (나)
│   ├── hackathon_me.png        # 해커톤 참가 사진
│   └── family_news.wav         # 가족 브리핑 오디오
│
├── db/                         # 데이터베이스 관련 모듈
│   ├── models.py               # SQLAlchemy ORM 모델 정의
│   ├── repositories.py         # 데이터 접근 레이어 (CRUD)
│   ├── session.py              # DB 세션 팩토리 및 연결 관리
│   ├── seeds.py                # 초기 데이터 시드 스크립트
│   └── _reset_db.py            # DB 스키마 초기화 마이그레이션
│
├── migrations/                 # Alembic 마이그레이션
│   ├── env.py                  # Alembic 환경 설정
│   └── versions/               # 마이그레이션 버전 파일
│       ├── 1c3eff4d6561_.py
│       └── 74ecdaf28434_.py
│
├── services/                   # 비즈니스 로직 서비스
│   ├── embeddings/
│   │   └── generator.py        # 임베딩 생성기 (OpenAI / 로컬 스텁)
│   │
│   ├── groups/
│   │   └── membership.py       # 그룹 멤버십 관리 유틸리티
│   │
│   └── rag/                    # RAG 파이프라인
│       ├── rag_connecter.py    # RAG 메인 연결기 (검색 + LLM 호출)
│       ├── retriever.py        # 벡터 검색 리트리버
│       ├── pgvector_adapter.py # pgvector VectorStore 어댑터
│       ├── llm_client.py       # LLM API 클라이언트
│       ├── prompt_builder.py   # 프롬프트 템플릿 생성기
│       ├── conversation_summarizer.py  # 대화 요약 및 저장
│       └── recent_previewer.py # 최근 대화 미리보기
│
├── tts.py                      # TTS 테스트/생성 스크립트
├── temp_rag.py                 # RAG 테스트 스크립트 (개발용)
├── util_add_conversation.py    # 대화 추가 유틸리티 스크립트
└── test_realtime_tts.py        # 실시간 TTS 테스트 스크립트
```

## 파일별 상세 설명

### 핵심 애플리케이션

#### `app.py`
메인 애플리케이션으로 Streamlit 기반 UI와 상태 머신을 포함합니다.

**상태 머신 흐름:**
1. `IDLE`: 대기 상태 - 얼굴 감지 대기
2. `USER_CHECK`: 사용자 확인 - 얼굴 임베딩으로 등록된 사용자인지 확인
3. `ENROLL`: 신규 등록 - 미등록 사용자 등록 폼 표시
4. `WELCOME`: 환영 인사 - VAD를 통한 음성 녹음 시작
5. `ASR`: 음성 인식 - Whisper로 텍스트 변환 및 RAG 응답 생성
6. `BYE`: 종료 - 대화 요약 저장 및 세션 종료

**주요 클래스:**
- `VADRecorder`: 음성 활동 감지 및 녹음
- `RealtimeSpeechOrchestrator`: 스트리밍 TTS 오케스트레이션

---

### 데이터베이스 (`db/`)

#### `models.py`
SQLAlchemy ORM 모델 정의

| 모델 | 설명 |
|------|------|
| `User` | 사용자 정보 (이름, 프로필 JSON) |
| `Group` | 그룹 정보 (가족, 팀 등) |
| `GroupMember` | 그룹-사용자 관계 (N:M) |
| `Event` | 일정/이벤트 정보 |
| `Embedding` | 벡터 임베딩 (pgvector, 1536차원) |
| `AuditAccess` | 접근 감사 로그 |

**가시성 레벨 (`VisibilityLevel`):**
- `self`: 본인만 열람 가능
- `group`: 같은 그룹 멤버만 열람 가능
- `public`: 모두 열람 가능

#### `repositories.py`
데이터 접근 레이어로 사용자/그룹/이벤트 관련 CRUD 및 쿼리 함수 제공

#### `session.py`
DB 세션 팩토리 및 연결 관리. 동기/비동기 세션 모두 지원

#### `seeds.py`
초기 데이터 시드 스크립트. 테스트용 사용자(철수, 영희), 그룹(demo-team), 이벤트 생성

---

### RAG 서비스 (`services/rag/`)

#### `rag_connecter.py`
RAG 파이프라인의 메인 진입점

- `get_rag_response()`: 사용자 질의에 대한 RAG 응답 스트림 반환
- 대상 범위 설정: `self`, `team`, `user`
- OpenAI GPT API 스트리밍 호출

#### `retriever.py`
`PastQueryRAGRetriever` 클래스

- `search_self()`: 본인 데이터만 검색
- `search_group_split()`: 그룹 멤버 데이터 포함 검색 (가중치 적용)
- `search_by_target()`: 대상별 검색 라우팅
- `save_past_query()`: 질의 저장

#### `pgvector_adapter.py`
`PgVectorEmbeddingsStore` 클래스 - LlamaIndex VectorStore 인터페이스 구현

- pgvector 코사인 거리 기반 검색
- 메타데이터 필터링 (소유자, 가시성, 텍스트 참조)

#### `llm_client.py`
LLM API 클라이언트

- `complete()`: OpenAI API 또는 로컬 스텁 자동 선택
- 환경변수 `OPENAI_API_KEY` 설정 여부에 따라 분기

#### `prompt_builder.py`
가드레일이 적용된 프롬프트 생성

- 소유자 명시 규칙 적용
- 다른 사람의 정보를 본인의 것으로 오인하지 않도록 방지

#### `conversation_summarizer.py`
대화 요약 및 임베딩 저장

- LLM으로 대화 요약 생성
- 요약 벡터화 후 DB 저장

---

### 임베딩 서비스 (`services/embeddings/`)

#### `generator.py`
임베딩 생성 팩토리

- OpenAI `text-embedding-3-small` (환경변수 설정 시)
- 로컬 해시 기반 스텁 (기본값, 테스트용)

---

### 그룹 서비스 (`services/groups/`)

#### `membership.py`
그룹 멤버십 관리 CLI 헬퍼

- 그룹 존재 여부 확인 및 생성
- 사용자 가입 처리

---

## 실행 방법

### uv 사용 (권장)

```shell
uv sync
uv run streamlit run app.py
```

### pip 사용

```shell
pip install -r requirements.txt
streamlit run app.py
```

## 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `DATABASE_URL` | PostgreSQL 연결 URL | `postgresql+psycopg://postgres:root@localhost:5432/postgres` |
| `OPENAI_API_KEY` | OpenAI API 키 | - |
| `OPENAI_API_BASE` | OpenAI API 베이스 URL (선택) | - |
| `EMBED_PROVIDER` | 임베딩 제공자 (`openai` / `local`) | `local` |
| `EMBED_MODEL` | OpenAI 임베딩 모델명 | `text-embedding-3-small` |
| `DB_ECHO` | SQL 로깅 활성화 | `false` |
| `DB_POOL_SIZE` | DB 커넥션 풀 크기 | `5` |

## 데이터베이스 설정

### PostgreSQL + pgvector 설치

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
```

### Alembic 마이그레이션

```shell
alembic upgrade head
```

### 초기 데이터 시드

```shell
python -m db.seeds
```

## 라이선스

2025 SKKU Casual Hackathon
