# test_realtime_tts.py
import logging
import time
import threading
import types
import numpy as np
import queue
import sounddevice as sd
from typing import Tuple, Optional

# ===== Logging =====
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
)
log = logging.getLogger("realtime-tts")

# =========================
# Kokoro TTS (실제)
# =========================
try:
    from kokoro import KPipeline
except Exception as e:
    raise RuntimeError(
            "Kokoro KPipeline을 불러오지 못했습니다. kokoro 패키지가 설치되어 있는지 확인하세요."
    ) from e


# =========================
# Realtime Speech Orchestrator (with tracing)
# =========================
class RealtimeSpeechOrchestrator:
    """
    RAG(openai.Stream) 텍스트 델타를 문장 단위로 잘라 큐에 넣고,
    별도 스레드에서 Kokoro 합성 → 즉시 sounddevice.OutputStream으로 재생.
    파일 저장 없음.

    ★추가: chunk 전달 확인용 트레이싱/메트릭스 내장
    - delta 수신/flush/synth/play 각 시점 로깅
    - 세그먼트 단위 레이턴시(Flush→Synth1, Flush→Play1) 측정
    """

    def __init__(
            self,
            pipeline,
            sr: int = 24000,
            voice: str = "af_heart",
            sentence_enders=(".", "!", "?", "…"),
            max_pending_chars: int = 80,
            blocksize: int = 1024,
    ):
        self.pipeline = pipeline
        self.sr = sr
        self.voice = voice
        self.sentence_enders = sentence_enders
        self.max_pending_chars = max_pending_chars
        self.blocksize = blocksize

        # 큐: (segment_id, text) / (segment_id, audio_chunk)
        self.text_queue: "queue.Queue[Tuple[int, str]]" = queue.Queue(maxsize=200)
        self.audio_queue: "queue.Queue[Tuple[Optional[int], Optional[np.ndarray]]]" = queue.Queue(maxsize=200)

        self.stop_text = threading.Event()
        self.stop_audio = threading.Event()
        self.kokoro_lock = threading.Lock()  # pipeline이 스레드-세이프가 아닐 수 있어 보호

        self.synth_t = None
        self.play_t = None

        self._full_text_chunks: list[str] = []
        self._t0 = time.perf_counter()

        # ---- tracing / metrics ----
        self.delta_count = 0
        self.flush_count = 0
        self.synth_chunk_count = 0
        self.play_chunk_count = 0

        self.seg_id_counter = 0
        self.seg_flush_time = {}  # seg_id -> t_flush
        self.seg_first_synth_time = {}  # seg_id -> t_first_audio_enqueued
        self.seg_first_play_time = {}  # seg_id -> t_first_audio_written

    # --- utils ---
    def _now(self) -> float:
        return time.perf_counter() - self._t0

    def _should_flush(self, s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        return s.endswith(self.sentence_enders) or len(s) >= self.max_pending_chars

    # --- threads ---
    def start(self):
        # 합성 스레드
        def synth_worker():
            try:
                while not (self.stop_text.is_set() and self.text_queue.empty()):
                    try:
                        seg_id, seg = self.text_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if not seg.strip():
                        continue

                    log.info(f"[SYNTH] seg#{seg_id} start (len={len(seg)})")
                    try:
                        with self.kokoro_lock:
                            first_chunk = True
                            for _, _, audio in self.pipeline(seg, voice=self.voice):
                                a = np.asarray(audio, dtype=np.float32)
                                # 첫 합성 시각 기록
                                if first_chunk:
                                    first_chunk = False
                                    self.seg_first_synth_time.setdefault(seg_id, self._now())
                                    log.info(
                                            f"[SYNTH] seg#{seg_id} first-chunk "
                                            f"(samples={len(a)}, dur={len(a) / self.sr:.3f}s, "
                                            f"flush→synth={self.seg_first_synth_time[seg_id] - self.seg_flush_time[seg_id]:.3f}s)"
                                    )

                                while True:
                                    try:
                                        self.audio_queue.put((seg_id, a), timeout=0.05)
                                        self.synth_chunk_count += 1
                                        break
                                    except queue.Full:
                                        if self.stop_audio.is_set():
                                            return
                    except Exception as e:
                        log.exception(f"[TTS] synth error: {e}")
                # 합성 종료 신호
                self.audio_queue.put((None, None))
            except Exception as e:
                log.exception(f"[TTS] synth_worker crashed: {e}")

        # 재생 스레드
        def playback_worker():
            try:
                with sd.OutputStream(
                        samplerate=self.sr,
                        channels=1,
                        dtype="float32",
                        blocksize=self.blocksize,
                        latency="low",
                ) as out:
                    seen_first_play_for_seg = set()
                    while True:
                        try:
                            seg_id, chunk = self.audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            if self.stop_audio.is_set():
                                break
                            continue
                        if seg_id is None and chunk is None:  # 합성 종료
                            break

                        # 세그먼트 첫 재생시각 기록
                        if seg_id is not None and seg_id not in seen_first_play_for_seg:
                            seen_first_play_for_seg.add(seg_id)
                            self.seg_first_play_time.setdefault(seg_id, self._now())
                            # flush→play 레이턴시
                            fp = self.seg_first_play_time[seg_id] - self.seg_flush_time[seg_id]
                            # synth→play 레이턴시
                            sp = (
                                    self.seg_first_play_time[seg_id]
                                    - self.seg_first_synth_time.get(seg_id, self.seg_first_play_time[seg_id])
                            )
                            log.info(
                                    f"[PLAY ] seg#{seg_id} first-write "
                                    f"(flush→play={fp:.3f}s, synth→play={sp:.3f}s)"
                            )

                        out.write(chunk)
                        self.play_chunk_count += 1
            except Exception as e:
                log.exception(f"[Audio] playback error: {e}")

        self.synth_t = threading.Thread(target=synth_worker, daemon=True)
        self.play_t = threading.Thread(target=playback_worker, daemon=True)
        self.synth_t.start()
        self.play_t.start()

    # --- stream feed ---
    def feed_openai_stream(self, stream):
        """
        openai.Stream을 읽어 문장 단위로 text_queue에 투입.
        """
        pending: list[str] = []
        for event in stream:
            et = getattr(event, "type", "")
            if et == "response.output_text.delta":
                delta = event.delta or ""
                if delta:
                    self.delta_count += 1
                    t = self._now()
                    log.info(
                        f"[DELTA] #{self.delta_count} @{t:.3f}s len={len(delta)} text='{delta[:40].replace(chr(10), ' ')}...'")
                    self._full_text_chunks.append(delta)
                    pending.append(delta)
                    seg = "".join(pending)
                    if self._should_flush(seg):
                        self.seg_id_counter += 1
                        seg_id = self.seg_id_counter
                        self.text_queue.put((seg_id, seg))
                        self.seg_flush_time[seg_id] = self._now()
                        self.flush_count += 1
                        log.info(f"[FLUSH] seg#{seg_id} @{self.seg_flush_time[seg_id]:.3f}s len={len(seg)}")
                        pending.clear()

            elif et == "response.completed":
                tail = "".join(pending).strip()
                if tail:
                    self.seg_id_counter += 1
                    seg_id = self.seg_id_counter
                    self.text_queue.put((seg_id, tail))
                    self.seg_flush_time[seg_id] = self._now()
                    self.flush_count += 1
                    log.info(f"[FLUSH] seg#{seg_id} (tail) @{self.seg_flush_time[seg_id]:.3f}s len={len(tail)}")
                break

            elif et == "response.error":
                log.error(f"[RAG] stream error: {getattr(event, 'error', None)}")
                break

            # 그 외 이벤트 타입은 무시
        self.stop_text.set()

    # --- teardown / report ---
    def wait(self):
        # 합성 종료까지 대기
        if self.synth_t:
            self.synth_t.join()
        # 재생 종료 신호
        self.stop_audio.set()
        if self.play_t:
            self.play_t.join()

    def report(self):
        log.info("==== STREAMING REPORT ====")
        log.info(f"deltas={self.delta_count}, flushes={self.flush_count}, "
                 f"synth_chunks={self.synth_chunk_count}, play_chunks={self.play_chunk_count}")

        # 세그먼트별 레이턴시
        for seg_id in sorted(self.seg_flush_time.keys()):
            t_flush = self.seg_flush_time.get(seg_id)
            t_synth = self.seg_first_synth_time.get(seg_id)
            t_play = self.seg_first_play_time.get(seg_id)
            fs = (t_synth - t_flush) if (t_flush is not None and t_synth is not None) else None
            fp = (t_play - t_flush) if (t_flush is not None and t_play is not None) else None
            log.info(
                    f"seg#{seg_id}: flush={t_flush:.3f}s, "
                    f"first_synth={t_synth:.3f}s" if t_synth is not None else f"seg#{seg_id}: flush={t_flush:.3f}s, first_synth=None"
            )
            if fp is not None:
                log.info(f"        → flush→play = {fp:.3f}s"
                         + (f", flush→synth = {fs:.3f}s" if fs is not None else ""))
        log.info("==========================")

    @property
    def full_text(self) -> str:
        return "".join(self._full_text_chunks)


# =========================
# 더미 OpenAI Stream (테스트 입력용)
# =========================
class FakeEvent(types.SimpleNamespace):
    pass


class FakeOpenAIStream:
    """
    response.output_text.delta 이벤트를 순차 전송하고,
    마지막에 response.completed를 보냅니다.
    """

    def __init__(self, deltas, delay=0.03):
        self.deltas = deltas
        self.delay = delay

    def __iter__(self):
        for d in self.deltas:
            time.sleep(self.delay)
            yield FakeEvent(type="response.output_text.delta", delta=d)
        yield FakeEvent(type="response.completed")


from openai import OpenAI
import os


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set; cannot use OpenAI provider.")

base_url = os.getenv("OPENAI_API_BASE")
client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def chatgpt_chat_stream(user_input, sys_prompt):
    """
    콜백 함수를 사용한 버전 - 더 유연한 스트리밍 처리 가능
    """
    try:
        input_list = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_input}
        ]

        stream = client.responses.create(
                model="gpt-5-mini",
                input=input_list,
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
                stream=True
        )

        return stream

    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        return None


# =========================
# 실행 예시 (실제 Kokoro로 재생 + chunk tracing)
# =========================
def run_with_real_kokoro(
        lang_code: str = "a",
        voice: str = "af_heart",
        sr: int = 24000,
        blocksize: int = 1024,
):
    log.info(f"Init Kokoro KPipeline(lang_code='{lang_code}') / voice='{voice}'")
    pipeline = KPipeline(lang_code=lang_code)

    orch = RealtimeSpeechOrchestrator(
            pipeline=pipeline,
            sr=sr,
            voice=voice,
            sentence_enders=(".", "!", "?", "…", "。", "？", "！"),
            max_pending_chars=48,  # 더 빠른 플러시
            blocksize=blocksize,
    )

    stream = chatgpt_chat_stream(
            sys_prompt="You are a helpful assistant that answers in concise American.",
            user_input="Tell me a short story about a robot learning to love."
    )

    log.info("Start realtime playback…")
    orch.start()
    orch.feed_openai_stream(stream)
    orch.wait()
    log.info(f"FULL TEXT: {orch.full_text}")
    orch.report()


if __name__ == "__main__":
    # 필요시 음성/언어 변경
    # lang_code 예시: 'a' (multi), 'en', 'ko' 등 배포본/모델에 따라 다를 수 있음
    run_with_real_kokoro(lang_code="a", voice="af_heart", sr=24000, blocksize=1024)
