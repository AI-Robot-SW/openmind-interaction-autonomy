"""
STT Provider - Speech-to-Text 변환 관리

이 모듈은 오디오 데이터를 받아 Google Cloud STT API를 통해
텍스트로 변환합니다.

Architecture:
    AudioProvider -> STTProvider -> Google Cloud STT API (Streaming)
                          ↓
                    SoundSensor (결과 콜백)

Dependencies:
    - google-cloud-speech: Google Cloud Speech-to-Text 클라이언트
    - AudioProvider: 오디오 데이터 소스

Note:
    이 Provider는 Singleton 패턴을 사용합니다.
    Google Cloud STT Streaming API의 5분 제한에 대응하여
    자동 재연결(while self.running) 루프를 사용합니다.

Environment Variables:
    - GOOGLE_APPLICATION_CREDENTIALS: Google Cloud 서비스 계정 JSON 파일 경로
    - OM_API_KEY: OpenMind API 키 (WebSocket 백엔드용)
"""

import json
import logging
import os
import queue
import threading
import time
from enum import Enum
from typing import Callable, Dict, List, Optional

from .singleton import singleton

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED or load_dotenv is None:
        return
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dotenv_path = os.path.join(repo_root, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path, override=False)
    else:
        load_dotenv(override=False)
    _ENV_LOADED = True

def _get_env_or_default(env_var: str, default: Optional[str] = None) -> Optional[str]:
    """환경 변수에서 값을 가져오거나 기본값 반환."""
    _ensure_env_loaded()
    return os.environ.get(env_var, default)


# ---- LatencyTracker (인라인) ----

class LatencyTracker:
    """오디오 → 인식 결과 지연 측정."""

    def __init__(self, sample_rate: int, channels: int = 1, sample_width_bytes: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self.stream_start_ts = None
        self.samples_sent = 0
        self.last_audio_end_ts = None

    def on_start(self):
        self.stream_start_ts = time.perf_counter()
        self.samples_sent = 0
        self.last_audio_end_ts = self.stream_start_ts

    def on_send_audio(self, audio_chunk: bytes):
        if self.stream_start_ts is None:
            self.on_start()
        samples = len(audio_chunk) // (self.sample_width_bytes * self.channels)
        self.samples_sent += samples
        self.last_audio_end_ts = self.stream_start_ts + (self.samples_sent / self.sample_rate)

    def on_result(self):
        if self.last_audio_end_ts is None:
            return None
        now = time.perf_counter()
        latency = now - self.last_audio_end_ts
        return max(0.0, latency)


class STTBackend(Enum):
    """STT 백엔드 종류."""

    GOOGLE_CLOUD = "google_cloud"
    GOOGLE_WEBSOCKET = "google_websocket"  # OpenMind 프록시 WebSocket (XX)
    LOCAL = "local"


class AudioEncoding(Enum):
    """오디오 인코딩 형식 (Google Cloud STT 지원)."""

    LINEAR16 = "LINEAR16"  # PCM 16-bit signed little-endian
    FLAC = "FLAC"
    MULAW = "MULAW"
    AMR = "AMR"
    AMR_WB = "AMR_WB"
    OGG_OPUS = "OGG_OPUS"
    SPEEX_WITH_HEADER_BYTE = "SPEEX_WITH_HEADER_BYTE"
    WEBM_OPUS = "WEBM_OPUS"


# 지원 언어 코드 매핑 (Google Cloud STT BCP-47 코드)
LANGUAGE_CODE_MAP: Dict[str, str] = {
    "english": "en-US",
    "english_uk": "en-GB",
    "korean": "ko-KR",
    "chinese": "cmn-Hans-CN",
    "chinese_traditional": "cmn-Hant-TW",
    "japanese": "ja-JP",
    "german": "de-DE",
    "french": "fr-FR",
    "spanish": "es-ES",
    "italian": "it-IT",
    "portuguese": "pt-BR",
    "russian": "ru-RU",
    "arabic": "ar-SA",
}

# Google Cloud STT 기본 설정
GOOGLE_CLOUD_STT_DEFAULT_CONFIG = {
    "encoding": AudioEncoding.LINEAR16.value,
    "sample_rate_hertz": 16000,
    "language_code": "ko-KR",
    "enable_automatic_punctuation": True,
    "model": 'telephony',
    "use_enhanced": True,
}


@singleton
class STTProvider:
    """
    STT Provider - Speech-to-Text 변환 관리자.

    오디오 데이터를 Google Cloud STT API로 전송하고, 변환된 텍스트를
    등록된 콜백으로 전달합니다.

    gRPC Streaming API의 5분 제한에 대응하여 자동 재연결 루프를 사용합니다.
    """

    def __init__(
        self,
        backend: STTBackend = STTBackend.GOOGLE_CLOUD,
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        language: str = "korean",
        enable_interim_results: bool = True,
        sample_rate: int = 16000,
        encoding: str = "LINEAR16",
        # Google Cloud STT 전용 파라미터
        model: str = "latest_long",
        use_enhanced: bool = True,
        enable_automatic_punctuation: bool = True,
        enable_speaker_diarization: bool = False,
        diarization_speaker_count: int = 2,
        single_utterance: bool = False,
        speech_contexts: Optional[List[Dict]] = None,
    ):
        self.running: bool = False
        self.backend = backend
        self.ws_url = ws_url or "wss://api.openmind.org/api/core/google/asr"

        # API 키 및 인증 정보는 환경 변수에서 로드 (파라미터 우선)
        self.api_key = api_key or _get_env_or_default("OM_API_KEY")
        self.credentials_path = credentials_path or _get_env_or_default(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        self.enable_interim_results = enable_interim_results
        self.sample_rate = sample_rate
        self.encoding = encoding

        # Google Cloud STT 전용 설정
        self._model = model
        self._use_enhanced = use_enhanced
        self._enable_automatic_punctuation = enable_automatic_punctuation
        self._enable_speaker_diarization = enable_speaker_diarization
        self._diarization_speaker_count = diarization_speaker_count
        self._single_utterance = single_utterance
        self._speech_contexts = speech_contexts or []
        self._lat = LatencyTracker(sample_rate=self.sample_rate)

        # 언어 코드 설정
        language_lower = language.strip().lower()
        if language_lower not in LANGUAGE_CODE_MAP:
            logging.warning(
                "Language '%s' not supported. "
                "Supported: %s. Defaulting to Korean.",
                language,
                list(LANGUAGE_CODE_MAP.keys()),
            )
            language_lower = "korean"
        self.language_code = LANGUAGE_CODE_MAP[language_lower]

        # 콜백 리스트
        self._result_callbacks: List[Callable[[str], None]] = []
        self._interim_callbacks: List[Callable[[str], None]] = []

        # 상태 변수
        self._is_listening: bool = False
        self._current_transcript: str = ""
        self._lock = threading.Lock()

        # Google Cloud Speech 클라이언트 (__init__에서 초기화)
        self._speech_client = None
        self._streaming_config = None
        self._audio_queue: Optional[queue.Queue] = None
        self._streaming_thread: Optional[threading.Thread] = None
        self._stream_stop_event = threading.Event()
        self._audio_drops = 0
        self._audio_sent = 0
        self._audio_consumed = 0

        # AudioProvider 큐 소비 스레드
        self._audio_provider: Optional["AudioProvider"] = None
        self._audio_consumer_thread: Optional[threading.Thread] = None
        self._audio_consumer_stop_event = threading.Event()
        self._audio_poll_timeout = 0.1

        # Google Cloud 클라이언트를 __init__에서 미리 생성
        if self.backend == STTBackend.GOOGLE_CLOUD:
            self._init_google_client()

        logging.info(
            "STTProvider initialized: backend=%s, language=%s, model=%s, sample_rate=%s",
            backend.value,
            self.language_code,
            model,
            sample_rate,
        )

    # ---- Result / Interim handlers (분리) ----

    def _on_result(self, transcript: str) -> None:
        """최종 인식 결과 처리."""
        latency = self._lat.on_result()
        if latency is not None:
            logging.info("STT latency: %.3f sec", latency)
        with self._lock:
            self._current_transcript = transcript
        for callback in list(self._result_callbacks):
            try:
                callback(transcript)
            except Exception as e:
                logging.error("STT result callback error: %s", e)

    def _on_interim(self, transcript: str) -> None:
        """중간 인식 결과 처리."""
        if not self.enable_interim_results:
            return
        for callback in list(self._interim_callbacks):
            try:
                callback(transcript)
            except Exception as e:
                logging.error("STT interim callback error: %s", e)

    # ---- Callbacks ----

    def register_result_callback(self, callback: Callable[[str], None]) -> None:
        """STT 최종 결과 콜백 등록."""
        if callback not in self._result_callbacks:
            self._result_callbacks.append(callback)
            logging.debug("STT result callback registered: %s", getattr(callback, '__name__', repr(callback)))

    def unregister_result_callback(self, callback: Callable[[str], None]) -> None:
        """STT 최종 결과 콜백 해제."""
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)
            logging.debug("STT result callback unregistered: %s", getattr(callback, '__name__', repr(callback)))

    def register_interim_callback(self, callback: Callable[[str], None]) -> None:
        """STT 중간 결과 콜백 등록."""
        if callback not in self._interim_callbacks:
            self._interim_callbacks.append(callback)
            logging.debug("STT interim callback registered: %s", getattr(callback, '__name__', repr(callback)))

    def unregister_interim_callback(self, callback: Callable[[str], None]) -> None:
        """STT 중간 결과 콜백 해제."""
        if callback in self._interim_callbacks:
            self._interim_callbacks.remove(callback)
            logging.debug("STT interim callback unregistered: %s", getattr(callback, '__name__', repr(callback)))

    # ---- WebSocket handler ----

    def _on_message(self, raw_message: str) -> None:
        """WebSocket 메시지 수신 핸들러."""
        try:
            message = json.loads(raw_message)
            if "asr_reply" in message:
                transcript = message["asr_reply"]
                is_final = message.get("is_final", True)
                if is_final:
                    self._on_result(transcript)
                else:
                    self._on_interim(transcript)
            elif "error" in message:
                logging.error("STT service error: %s", message["error"])
        except json.JSONDecodeError as e:
            logging.warning("Invalid JSON message from STT service: %s", e)
        except Exception as e:
            logging.error("Error processing STT message: %s", e)

    # ---- Google Cloud STT ----

    def _init_google_client(self) -> bool:
        try:
            from google.cloud import speech
        except Exception as e:
            logging.error("google-cloud-speech not available: %s", e)
            return False

        try:
            if self.credentials_path:
                if hasattr(speech.SpeechClient, "from_service_account_file"):
                    self._speech_client = speech.SpeechClient.from_service_account_file(
                        self.credentials_path
                    )
                else:
                    self._speech_client = speech.SpeechClient.from_service_account_json(
                        self.credentials_path
                    )
            else:
                self._speech_client = speech.SpeechClient()
        except Exception as e:
            logging.error("Failed to init SpeechClient: %s", e)
            return False

        try:
            encoding_enum = getattr(
                speech.RecognitionConfig.AudioEncoding, self.encoding
            )
        except Exception:
            logging.warning(
                "Unknown encoding '%s', defaulting to LINEAR16", self.encoding
            )
            encoding_enum = speech.RecognitionConfig.AudioEncoding.LINEAR16

        recognition_config = speech.RecognitionConfig(
            encoding=encoding_enum,
            sample_rate_hertz=self.sample_rate,
            language_code=self.language_code,
            enable_automatic_punctuation=self._enable_automatic_punctuation,
            model=self._model,
            use_enhanced=self._use_enhanced,
        )

        if self._enable_speaker_diarization:
            recognition_config.diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=1,
                max_speaker_count=self._diarization_speaker_count,
            )

        if self._speech_contexts:
            recognition_config.speech_contexts = [
                speech.SpeechContext(
                    phrases=ctx.get("phrases", []), boost=ctx.get("boost", 0)
                )
                for ctx in self._speech_contexts
            ]

        self._streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self.enable_interim_results,
            single_utterance=self._single_utterance,
        )
        return True

    def _google_request_generator(self):
        from google.cloud import speech

        if self._streaming_config is None or self._audio_queue is None:
            return None

        while not self._stream_stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break
            self._audio_consumed += 1
            if self._audio_consumed == 1 or self._audio_consumed % 50 == 0:
                stt_queue_size = self._audio_queue.qsize()
                if self._audio_provider is not None:
                    audio_stats = self._audio_provider.get_buffer_stats()
                    logging.info(
                        "Google streaming consumed audio chunks=%d "
                        "(stt_queue=%d, audio_buffer=%d/%d, drops=%d, latency_ms=%.1f)",
                        self._audio_consumed,
                        stt_queue_size,
                        int(audio_stats["queued_chunks"]),
                        int(audio_stats["max_chunks"]),
                        int(audio_stats["drops"]),
                        audio_stats["approx_latency_ms"],
                    )
                else:
                    logging.info(
                        "Google streaming consumed audio chunks=%d (stt_queue=%d)",
                        self._audio_consumed,
                        stt_queue_size,
                    )
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def _process_response(self, response) -> None:
        """Google Cloud STT 응답 처리 — _on_result / _on_interim으로 분배."""
        if not getattr(response, "results", None):
            return
        for result in response.results:
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if result.is_final:
                self._on_result(transcript)
            else:
                self._on_interim(transcript)

    def _google_streaming_worker(self) -> None:
        """Streaming worker with auto-reconnect for 5-min gRPC limit."""
        while self.running and not self._stream_stop_event.is_set():
            try:
                if self._speech_client is None:
                    if not self._init_google_client():
                        logging.error("Google Cloud client init failed, stopping worker")
                        break

                logging.info("Google streaming session started")
                self._audio_consumed = 0
                requests = self._google_request_generator()
                if requests is None:
                    break
                responses = self._speech_client.streaming_recognize(
                    config=self._streaming_config,
                    requests=requests,
                )
                for response in responses:
                    if self._stream_stop_event.is_set():
                        return
                    self._process_response(response)
            except Exception as e:
                if self._stream_stop_event.is_set():
                    break
                logging.warning(
                    "Google streaming disconnected: %s. Reconnecting in 1s...", e
                )
                time.sleep(1.0)
        logging.info("Google streaming worker stopped")

    # ---- AudioProvider 큐 소비 ----

    def _audio_consumer_loop(self) -> None:
        while not self._audio_consumer_stop_event.is_set():
            try:
                if not self.running or self._audio_provider is None:
                    self._audio_consumer_stop_event.wait(timeout=self._audio_poll_timeout)
                    continue
                if not self._audio_provider.running:
                    self._audio_consumer_stop_event.wait(timeout=self._audio_poll_timeout)
                    continue
                chunk = self._audio_provider.get_audio_chunk(
                    timeout=self._audio_poll_timeout
                )
                if chunk:
                    self.send_audio(chunk)
            except Exception as e:
                if self._audio_consumer_stop_event.is_set():
                    break
                logging.error("Audio consumer error: %s", e)

    def _start_audio_consumer(self) -> None:
        if self._audio_consumer_thread and self._audio_consumer_thread.is_alive():
            return
        self._audio_consumer_stop_event.clear()
        self._audio_consumer_thread = threading.Thread(
            target=self._audio_consumer_loop, daemon=True
        )
        self._audio_consumer_thread.start()

    def _stop_audio_consumer(self) -> None:
        self._audio_consumer_stop_event.set()
        if self._audio_consumer_thread is not None:
            self._audio_consumer_thread.join(timeout=2.0)

    def attach_audio_provider(
        self, audio_provider: "AudioProvider", poll_timeout: float = 0.1
    ) -> None:
        """AudioProvider 큐를 소비하도록 연결."""
        self._audio_provider = audio_provider
        self._audio_poll_timeout = max(0.01, poll_timeout)
        if self.running:
            self._start_audio_consumer()

    # ---- Audio send ----

    def pause(self) -> None:
        """STT 일시 중지 (에코 방지: 스피커 재생 중 오디오 드롭)."""
        with self._lock:
            if not self._is_listening:
                return
            self._is_listening = False
        logging.info("STTProvider paused (echo prevention)")

    def resume(self) -> None:
        """STT 재개."""
        with self._lock:
            if self._is_listening:
                return
            self._is_listening = True
        logging.info("STTProvider resumed")

    def send_audio(self, audio_chunk: bytes) -> None:
        """오디오 데이터를 ASR 서비스로 전송."""
        if not self.running:
            logging.warning("STTProvider is not running. Call start() first.")
            return

        if not self._is_listening:
            return

        if self.backend == STTBackend.GOOGLE_CLOUD:
            if self._audio_queue is None:
                logging.warning("STTProvider audio queue not ready")
                return
            try:
                self._audio_queue.put_nowait(audio_chunk)
                self._lat.on_send_audio(audio_chunk)
                self._audio_sent += 1
            except queue.Full:
                self._audio_drops += 1
                if self._audio_drops == 1 or self._audio_drops % 50 == 0:
                    logging.warning(
                        "STTProvider audio queue full; dropping chunk (drops=%d)",
                        self._audio_drops,
                    )
            return

        # TODO: WebSocket으로 오디오 전송
        self._lat.on_send_audio(audio_chunk)

    def end_utterance(self) -> None:
        """발화 종료 신호 전송."""
        if not self.running:
            return

        if self.backend == STTBackend.GOOGLE_CLOUD:
            if self._audio_queue is not None:
                try:
                    self._audio_queue.put_nowait(None)
                except queue.Full:
                    logging.warning("STTProvider audio queue full; end_utterance skipped")
        logging.debug("End utterance signal sent")

    # ---- Query ----

    def get_current_transcript(self) -> str:
        """현재 변환된 텍스트 반환."""
        with self._lock:
            return self._current_transcript

    def is_listening(self) -> bool:
        """현재 청취 상태 반환."""
        with self._lock:
            return self._is_listening

    # ---- Lifecycle ----

    def start(self) -> None:
        """STT 서비스 연결 시작."""
        if self.running:
            logging.warning("STTProvider is already running")
            return

        self.running = True
        self._is_listening = True

        if self.backend == STTBackend.GOOGLE_CLOUD:
            if self._speech_client is None:
                if not self._init_google_client():
                    self.running = False
                    self._is_listening = False
                    return
            self._stream_stop_event.clear()
            self._audio_queue = queue.Queue(maxsize=200)
            self._streaming_thread = threading.Thread(
                target=self._google_streaming_worker, daemon=True
            )
            self._streaming_thread.start()
            logging.info(
                "Google streaming thread started (alive=%s)",
                self._streaming_thread.is_alive(),
            )
        else:
            # TODO: WebSocket 연결 시작
            pass

        self._lat.on_start()
        logging.info("STTProvider started with language=%s", self.language_code)

        if self._audio_provider is not None:
            self._start_audio_consumer()

    def stop(self) -> None:
        """STT 서비스 연결 종료."""
        if not self.running:
            logging.warning("STTProvider is not running")
            return

        self.running = False
        self._is_listening = False

        if self.backend == STTBackend.GOOGLE_CLOUD:
            self._stream_stop_event.set()
            # Poison pill to unblock the request generator
            if self._audio_queue is not None:
                try:
                    self._audio_queue.put_nowait(None)
                except queue.Full:
                    pass
            if self._streaming_thread is not None:
                self._streaming_thread.join(timeout=5.0)
        else:
            # TODO: WebSocket 연결 종료
            pass

        self._stop_audio_consumer()
        logging.info("STTProvider stopped")

    def configure(
        self,
        language: Optional[str] = None,
        enable_interim_results: Optional[bool] = None,
        sample_rate: Optional[int] = None,
        # Google Cloud STT 전용 파라미터
        model: Optional[str] = None,
        use_enhanced: Optional[bool] = None,
        enable_automatic_punctuation: Optional[bool] = None,
        enable_speaker_diarization: Optional[bool] = None,
        diarization_speaker_count: Optional[int] = None,
        single_utterance: Optional[bool] = None,
        speech_contexts: Optional[List[Dict]] = None,
    ) -> None:
        """Provider 설정 변경."""
        restart_needed = False

        if language is not None:
            language_lower = language.strip().lower()
            if language_lower in LANGUAGE_CODE_MAP:
                new_code = LANGUAGE_CODE_MAP[language_lower]
                if new_code != self.language_code:
                    self.language_code = new_code
                    restart_needed = True

        if enable_interim_results is not None:
            self.enable_interim_results = enable_interim_results
            restart_needed = True

        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            restart_needed = True

        # Google Cloud STT 전용 설정
        if model is not None and model != self._model:
            self._model = model
            restart_needed = True

        if use_enhanced is not None:
            self._use_enhanced = use_enhanced
            restart_needed = True

        if enable_automatic_punctuation is not None:
            self._enable_automatic_punctuation = enable_automatic_punctuation
            restart_needed = True

        if enable_speaker_diarization is not None:
            self._enable_speaker_diarization = enable_speaker_diarization
            restart_needed = True

        if diarization_speaker_count is not None:
            self._diarization_speaker_count = diarization_speaker_count
            restart_needed = True

        if single_utterance is not None:
            self._single_utterance = single_utterance
            restart_needed = True

        if speech_contexts is not None:
            self._speech_contexts = speech_contexts
            restart_needed = True

        if restart_needed and self.running:
            self.stop()
            self.start()
            logging.info("STTProvider reconfigured and restarted")

    def get_recognition_config(self) -> Dict:
        """현재 Google Cloud STT 설정 반환."""
        return {
            "backend": self.backend.value,
            "language_code": self.language_code,
            "sample_rate_hertz": self.sample_rate,
            "encoding": self.encoding,
            "model": self._model,
            "use_enhanced": self._use_enhanced,
            "enable_automatic_punctuation": self._enable_automatic_punctuation,
            "enable_interim_results": self.enable_interim_results,
            "enable_speaker_diarization": self._enable_speaker_diarization,
            "diarization_speaker_count": self._diarization_speaker_count,
            "single_utterance": self._single_utterance,
            "speech_contexts": self._speech_contexts,
        }
