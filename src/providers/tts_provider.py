"""
TTS Provider - Text-to-Speech 변환 관리

이 모듈은 텍스트를 받아 TTS 서비스를 통해
오디오 데이터로 변환합니다.

Architecture:
    SpeakAction -> TTSProvider -> TTS Service (HTTP/WebSocket)
                        ↓
                  SpeakerProvider (오디오 출력)

Dependencies:
    - om1_speech (AudioOutputStream): 오디오 스트림 처리
    - SpeakerProvider: 오디오 출력

Note:
    이 Provider는 Singleton 패턴을 사용합니다.
    ElevenLabs, Naver Clova, Riva 등 다양한 TTS 백엔드를 지원합니다.

Supported Backends:
    - ElevenLabs: output_format = "mp3_44100_128", "mp3_44100_192", "pcm_16000", etc.
    - Naver Clova: output_format = "mp3", "wav" (with sampling_rate option)
    - Riva: NVIDIA Riva TTS
    - Google: Google Cloud TTS
    - Local: Local TTS engine

Environment Variables:
    - OM_API_KEY: OpenMind API 키
    - ELEVENLABS_API_KEY: ElevenLabs API 키
    - ELEVENLABS_VOICE_ID: ElevenLabs 음성 ID
    - NAVER_CLIENT_ID: Naver Cloud Platform Client ID
    - NAVER_CLIENT_SECRET: Naver Cloud Platform Client Secret
"""

import io
import logging
import os
import queue
import threading
import wave
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import requests

from .singleton import singleton


def _get_env_or_default(env_var: str, default: Optional[str] = None) -> Optional[str]:
    """환경 변수에서 값을 가져오거나 기본값 반환."""
    return os.environ.get(env_var, default)


class TTSBackend(Enum):
    """TTS 백엔드 종류."""

    ELEVENLABS = "elevenlabs"
    NAVER_CLOVA = "naver_clova"
    RIVA = "riva"
    GOOGLE = "google"
    LOCAL = "local"


class TTSOutputFormat(Enum):
    """TTS 출력 오디오 형식."""

    # ElevenLabs formats
    MP3_44100_128 = "mp3_44100_128"
    MP3_44100_192 = "mp3_44100_192"
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"
    ULAW_8000 = "ulaw_8000"

    # Naver Clova formats
    MP3 = "mp3"
    WAV = "wav"


class TTSState(Enum):
    """TTS 상태."""

    IDLE = "idle"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


# Naver Clova TTS 기본 설정
NAVER_CLOVA_DEFAULT_CONFIG = {
    "url": "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts",
    "speaker": "nara",  # 기본 음성: 아라
    "volume": 0,
    "speed": 0,
    "pitch": 0,
    "emotion": 0,
    "emotion_strength": 1,
    "format": "mp3",
    "sampling_rate": 24000,
    "alpha": 0,
    "end_pitch": 0,
}

# Naver Clova 지원 음성 목록 (주요 한국어 음성)
NAVER_CLOVA_VOICES = {
    # 일반 음성
    "nara": "아라 (여성)",
    "njinho": "진호 (남성)",
    "nmijin": "미진 (여성)",
    "njihun": "지훈 (남성)",
    "nyuna": "유나 (여성)",
    "nkyuwon": "규원 (남성)",
    # Pro 음성 (감정 지원)
    "vara": "아라 Pro (여성, 감정지원)",
    "vdaeseong": "대성 Pro (남성, 감정지원)",
    "vdain": "다인 Pro (여성, 감정지원)",
    "vgoeun": "고은 Pro (여성, 감정지원)",
    "vmikyung": "미경 Pro (여성, 감정지원)",
    "vyuna": "유나 Pro (여성, 감정지원)",
    # 아동 음성
    "ndain": "다인 (아동여)",
    "ngaram": "가람 (아동여)",
    "nhajun": "하준 (아동남)",
}


@singleton
class TTSProvider:
    """
    TTS Provider - Text-to-Speech 변환 관리자.

    텍스트를 TTS 서비스로 전송하고, 변환된 오디오를
    SpeakerProvider로 전달합니다.

    Attributes
    ----------
    running : bool
        Provider 실행 상태
    backend : TTSBackend
        사용 중인 TTS 백엔드
    voice_id : str
        음성 ID (ElevenLabs) 또는 speaker (Naver Clova)
    model_id : str
        모델 ID (ElevenLabs 전용)
    output_format : str
        출력 오디오 형식

    Methods
    -------
    start()
        TTS 서비스 연결 시작
    stop()
        TTS 서비스 연결 종료
    add_pending_message(text)
        TTS 변환 요청 추가
    create_pending_message(text) -> dict
        TTS 요청 메시지 생성
    register_tts_state_callback(callback)
        TTS 상태 콜백 등록
    get_pending_message_count() -> int
        대기 중인 메시지 수 반환
    """

    def __init__(
        self,
        url: str = "https://api.openmind.org/api/core/elevenlabs/tts",
        api_key: Optional[str] = None,
        backend_api_key: Optional[str] = None,
        backend: TTSBackend = TTSBackend.ELEVENLABS,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "mp3_44100_128",
        language: str = "ko",
        enable_tts_interrupt: bool = False,
        # Naver Clova 전용 파라미터
        naver_client_id: Optional[str] = None,
        naver_client_secret: Optional[str] = None,
        speaker: str = "nara",
        volume: int = 0,
        speed: int = 0,
        pitch: int = 0,
        emotion: int = 0,
        emotion_strength: int = 1,
        sampling_rate: int = 24000,
        alpha: int = 0,
        end_pitch: int = 0,
    ):
        """
        TTS Provider 초기화.

        Parameters
        ----------
        url : str
            TTS 서비스 URL
        api_key : Optional[str]
            OM API 키
        backend_api_key : Optional[str]
            백엔드 서비스 API 키 (예: ElevenLabs API 키)
        backend : TTSBackend
            TTS 백엔드 종류, 기본값 ELEVENLABS
        voice_id : str
            음성 ID (ElevenLabs), 환경변수 ELEVENLABS_VOICE_ID에서 로드
        model_id : str
            모델 ID (ElevenLabs), 기본값 "eleven_flash_v2_5"
        output_format : str
            출력 오디오 형식
            - ElevenLabs: "mp3_44100_128", "pcm_16000", etc.
            - Naver Clova: "mp3", "wav"
        language : str
            언어 코드, 기본값 "ko"
        enable_tts_interrupt : bool
            TTS 인터럽트 활성화 여부, 기본값 False

        Naver Clova 전용 Parameters
        ---------------------------
        naver_client_id : Optional[str]
            Naver Cloud Platform Client ID
        naver_client_secret : Optional[str]
            Naver Cloud Platform Client Secret
        speaker : str
            Naver Clova 음성 종류, 기본값 "nara" (아라)
        volume : int
            음성 크기 (-5~5), 기본값 0
        speed : int
            음성 속도 (-5~10), 기본값 0
        pitch : int
            음성 높낮이 (-5~5), 기본값 0
        emotion : int
            감정 (0: 중립, 1: 슬픔, 2: 기쁨, 3: 분노), 기본값 0
        emotion_strength : int
            감정 강도 (0~2), 기본값 1
        sampling_rate : int
            샘플링 레이트 (WAV 전용: 8000, 16000, 24000, 48000), 기본값 24000
        alpha : int
            음색 (-5~5), 기본값 0
        end_pitch : int
            끝음 처리 (-5~5), 기본값 0
        """
        self.running: bool = False
        self.url = url
        self.backend = backend

        # API 키는 환경 변수에서 로드 (파라미터 우선)
        self.api_key = api_key or _get_env_or_default("OM_API_KEY")
        self.backend_api_key = backend_api_key or _get_env_or_default("ELEVENLABS_API_KEY")

        # ElevenLabs 설정 (환경 변수에서 로드)
        self._voice_id = voice_id or _get_env_or_default("ELEVENLABS_VOICE_ID")
        self._model_id = model_id
        self._output_format = output_format
        self._language = language
        self._enable_tts_interrupt = enable_tts_interrupt

        # Naver Clova 전용 설정 (환경 변수에서 로드)
        self._naver_client_id = naver_client_id or _get_env_or_default("NAVER_CLIENT_ID")
        self._naver_client_secret = naver_client_secret or _get_env_or_default("NAVER_CLIENT_SECRET")
        self._speaker = speaker
        self._volume = volume
        self._speed = speed
        self._pitch = pitch
        self._emotion = emotion
        self._emotion_strength = emotion_strength
        self._sampling_rate = sampling_rate
        self._alpha = alpha
        self._end_pitch = end_pitch

        # Naver Clova 백엔드인 경우 URL 설정
        if backend == TTSBackend.NAVER_CLOVA and url == "https://api.openmind.org/api/core/elevenlabs/tts":
            self.url = NAVER_CLOVA_DEFAULT_CONFIG["url"]

        # 요청 큐
        self._pending_requests: queue.Queue[Dict] = queue.Queue()

        # 콜백 리스트
        self._state_callbacks: List[Callable[[TTSState], None]] = []
        self._audio_callbacks: List[Callable[[bytes], None]] = []

        # 상태 변수
        self._current_state: TTSState = TTSState.IDLE
        self._lock = threading.Lock()
        self._processing_thread: Optional[threading.Thread] = None

        # TODO: AudioOutputStream 초기화
        # self._audio_stream: AudioOutputStream = AudioOutputStream(
        #     url=url,
        #     headers={"x-api-key": api_key} if api_key else None,
        #     enable_tts_interrupt=enable_tts_interrupt,
        # )

        logging.info(
            f"TTSProvider initialized: backend={backend.value}, "
            f"voice={voice_id if backend != TTSBackend.NAVER_CLOVA else speaker}, "
            f"format={output_format}"
        )

    @property
    def voice_id(self) -> str:
        """현재 음성 ID 반환."""
        return self._voice_id

    @voice_id.setter
    def voice_id(self, value: str) -> None:
        """음성 ID 설정."""
        self._voice_id = value
        logging.debug(f"Voice ID set to {value}")

    @property
    def model_id(self) -> str:
        """현재 모델 ID 반환."""
        return self._model_id

    @model_id.setter
    def model_id(self, value: str) -> None:
        """모델 ID 설정."""
        self._model_id = value
        logging.debug(f"Model ID set to {value}")

    @property
    def data(self) -> Optional[Dict]:
        """
        Get the current provider data.

        Returns
        -------
        Optional[Dict]
            Current TTS provider state information.
        """
        with self._lock:
            return {
                "state": self._current_state.value,
                "pending_count": self._pending_requests.qsize(),
                "backend": self.backend.value,
                "speaker": self._speaker,
                "running": self.running,
            }

    def register_tts_state_callback(
        self, callback: Callable[[TTSState], None]
    ) -> None:
        """
        TTS 상태 콜백 등록.

        Parameters
        ----------
        callback : Callable[[TTSState], None]
            TTS 상태 변경 시 호출될 콜백
        """
        if callback not in self._state_callbacks:
            self._state_callbacks.append(callback)
            logging.debug(f"TTS state callback registered: {callback.__name__}")

    def unregister_tts_state_callback(
        self, callback: Callable[[TTSState], None]
    ) -> None:
        """
        TTS 상태 콜백 해제.

        Parameters
        ----------
        callback : Callable[[TTSState], None]
            해제할 콜백 함수
        """
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)
            logging.debug(f"TTS state callback unregistered: {callback.__name__}")

    def register_audio_callback(self, callback: Callable[[bytes], None]) -> None:
        """
        오디오 데이터 콜백 등록.

        Parameters
        ----------
        callback : Callable[[bytes], None]
            생성된 오디오 데이터를 받을 콜백
        """
        if callback not in self._audio_callbacks:
            self._audio_callbacks.append(callback)
            logging.debug(f"TTS audio callback registered: {callback.__name__}")

    def unregister_audio_callback(self, callback: Callable[[bytes], None]) -> None:
        """
        오디오 데이터 콜백 해제.

        Parameters
        ----------
        callback : Callable[[bytes], None]
            해제할 콜백 함수
        """
        if callback in self._audio_callbacks:
            self._audio_callbacks.remove(callback)
            logging.debug(f"TTS audio callback unregistered: {callback.__name__}")

    def _set_state(self, state: TTSState) -> None:
        """
        TTS 상태 변경 및 알림.

        Parameters
        ----------
        state : TTSState
            새 상태
        """
        with self._lock:
            self._current_state = state

        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logging.error(f"TTS state callback error: {e}")

    def create_pending_message(self, text: str) -> Dict:
        """
        TTS 요청 메시지 생성.

        백엔드 종류에 따라 다른 형식의 메시지를 생성합니다.

        Parameters
        ----------
        text : str
            변환할 텍스트

        Returns
        -------
        Dict
            TTS 요청 파라미터 딕셔너리
        """
        if self.backend == TTSBackend.NAVER_CLOVA:
            return self._create_naver_clova_message(text)
        else:
            return self._create_elevenlabs_message(text)

    def _create_elevenlabs_message(self, text: str) -> Dict:
        """
        ElevenLabs TTS 요청 메시지 생성.

        Parameters
        ----------
        text : str
            변환할 텍스트

        Returns
        -------
        Dict
            ElevenLabs TTS 요청 파라미터
        """
        message = {
            "text": text,
            "voice_id": self._voice_id,
            "model_id": self._model_id,
            "output_format": self._output_format,
            "language": self._language,
            "backend": "elevenlabs",
        }

        if self.backend_api_key:
            message["backend_api_key"] = self.backend_api_key

        return message

    def _create_naver_clova_message(self, text: str) -> Dict:
        """
        Naver Clova TTS 요청 메시지 생성.

        Parameters
        ----------
        text : str
            변환할 텍스트

        Returns
        -------
        Dict
            Naver Clova TTS 요청 파라미터
        """
        message = {
            "text": text,
            "speaker": self._speaker,
            "volume": self._volume,
            "speed": self._speed,
            "pitch": self._pitch,
            "format": self._output_format,  # "mp3" or "wav"
            "backend": "naver_clova",
        }

        # 감정 지원 음성인 경우 감정 파라미터 추가
        emotion_supported_voices = [
            "nara", "vara", "vmikyung", "vdain", "vyuna", "vgoeun", "vdaeseong"
        ]
        if self._speaker in emotion_supported_voices:
            message["emotion"] = self._emotion
            # vara, vmikyung 등 Pro 음성은 emotion_strength도 지원
            if self._speaker.startswith("v"):
                message["emotion-strength"] = self._emotion_strength

        # WAV 형식인 경우 샘플링 레이트 추가
        if self._output_format.lower() == "wav":
            message["sampling-rate"] = self._sampling_rate

        # 추가 파라미터
        message["alpha"] = self._alpha

        # 끝음 처리 지원 음성 확인
        end_pitch_supported = self._speaker.startswith("d") or self._speaker in [
            "clara", "matt", "meimei", "liangliang", "chiahua", "kuanlin", "carmen", "jose"
        ]
        if end_pitch_supported:
            message["end-pitch"] = self._end_pitch

        # API 인증 정보
        if self._naver_client_id and self._naver_client_secret:
            message["naver_client_id"] = self._naver_client_id
            message["naver_client_secret"] = self._naver_client_secret

        return message

    def add_pending_message(self, message: Union[str, Dict]) -> None:
        """
        TTS 변환 요청 추가.

        Parameters
        ----------
        message : Union[str, Dict]
            변환할 텍스트 또는 요청 딕셔너리
        """
        if not self.running:
            logging.warning(
                "TTSProvider is not running. Call start() before adding messages."
            )
            return

        if isinstance(message, str):
            message = self.create_pending_message(message)

        self._pending_requests.put(message)
        logging.info(f"TTS request queued: {message.get('text', '')[:50]}...")

    def get_pending_message_count(self) -> int:
        """
        대기 중인 메시지 수 반환.

        Returns
        -------
        int
            대기 중인 TTS 요청 수
        """
        return self._pending_requests.qsize()

    def get_current_state(self) -> TTSState:
        """
        현재 TTS 상태 반환.

        Returns
        -------
        TTSState
            현재 상태
        """
        with self._lock:
            return self._current_state

    def _synthesize_naver_clova(self, request: Dict) -> Optional[bytes]:
        """
        Naver Clova TTS API 호출하여 오디오 바이트 반환.

        Parameters
        ----------
        request : Dict
            TTS 요청 파라미터

        Returns
        -------
        Optional[bytes]
            PCM16 오디오 데이터 또는 실패 시 None
        """
        if not self._naver_client_id or not self._naver_client_secret:
            logging.error("Naver Clova credentials not configured")
            return None

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self._naver_client_id,
            "X-NCP-APIGW-API-KEY": self._naver_client_secret,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        # API 요청 본문 구성
        body = {
            "speaker": request.get("speaker", self._speaker),
            "text": request.get("text", ""),
            "format": request.get("format", self._output_format),
            "volume": request.get("volume", self._volume),
            "speed": request.get("speed", self._speed),
            "pitch": request.get("pitch", self._pitch),
            "alpha": request.get("alpha", self._alpha),
        }

        # WAV 형식인 경우 샘플링 레이트 추가
        if request.get("format", self._output_format).lower() == "wav":
            body["sampling-rate"] = request.get("sampling-rate", self._sampling_rate)

        # 감정 파라미터 (지원 음성인 경우)
        if "emotion" in request:
            body["emotion"] = request["emotion"]
        if "emotion-strength" in request:
            body["emotion-strength"] = request["emotion-strength"]

        # 끝음 처리 (지원 음성인 경우)
        if "end-pitch" in request:
            body["end-pitch"] = request["end-pitch"]

        try:
            response = requests.post(
                self.url, headers=headers, data=body, timeout=30
            )
            response.raise_for_status()

            # WAV 형식인 경우 PCM 데이터 추출
            if request.get("format", self._output_format).lower() == "wav":
                return self._extract_pcm_from_wav(response.content)

            # MP3 등 다른 형식은 그대로 반환 (추후 디코딩 필요)
            logging.warning(
                f"Non-WAV format ({request.get('format', self._output_format)}) "
                "returned. Consider using WAV for direct playback."
            )
            return response.content

        except requests.RequestException as e:
            logging.error(f"Naver Clova TTS request failed: {e}")
            return None

    def _extract_pcm_from_wav(self, wav_data: bytes) -> bytes:
        """
        WAV 파일에서 PCM16 데이터 추출.

        Parameters
        ----------
        wav_data : bytes
            WAV 형식의 오디오 데이터

        Returns
        -------
        bytes
            PCM16 raw 오디오 데이터
        """
        try:
            with io.BytesIO(wav_data) as buf:
                with wave.open(buf, "rb") as wf:
                    # WAV 정보 로깅
                    logging.debug(
                        f"WAV info: channels={wf.getnchannels()}, "
                        f"sample_width={wf.getsampwidth()}, "
                        f"framerate={wf.getframerate()}, "
                        f"nframes={wf.getnframes()}"
                    )
                    return wf.readframes(wf.getnframes())
        except Exception as e:
            logging.error(f"Failed to extract PCM from WAV: {e}")
            # 실패 시 원본 데이터 반환 (WAV 헤더 포함)
            return wav_data

    def _processing_loop(self) -> None:
        """
        TTS 처리 루프 (별도 스레드에서 실행).

        요청 큐에서 TTS 요청을 가져와 합성하고,
        오디오 콜백을 통해 SpeakerProvider로 전달합니다.
        """
        while self.running:
            try:
                # 큐에서 요청 가져오기 (1초 타임아웃)
                request = self._pending_requests.get(timeout=1.0)

                # 처리 시작
                self._set_state(TTSState.PROCESSING)
                text_preview = request.get("text", "")[:30]
                logging.debug(f"Processing TTS: {text_preview}...")

                # TTS 합성
                audio_data: Optional[bytes] = None
                backend = request.get("backend", self.backend.value)

                if backend == "naver_clova" or self.backend == TTSBackend.NAVER_CLOVA:
                    audio_data = self._synthesize_naver_clova(request)
                else:
                    # TODO: ElevenLabs 및 다른 백엔드 구현
                    logging.warning(f"Backend {backend} not yet implemented")

                if audio_data:
                    # 오디오 콜백 호출 (SpeakerProvider.queue_audio)
                    self._set_state(TTSState.SPEAKING)
                    for callback in self._audio_callbacks:
                        try:
                            callback(audio_data)
                        except Exception as e:
                            logging.error(f"Audio callback error: {e}")

                    logging.info(
                        f"TTS audio sent to speaker: {len(audio_data)} bytes"
                    )
                else:
                    logging.error(f"TTS synthesis failed for: {text_preview}...")
                    self._set_state(TTSState.ERROR)

                # 완료
                self._set_state(TTSState.IDLE)
                self._pending_requests.task_done()

            except queue.Empty:
                # 타임아웃 - 계속 대기
                continue
            except Exception as e:
                logging.error(f"TTS processing error: {e}")
                self._set_state(TTSState.ERROR)

    def interrupt(self) -> None:
        """
        현재 TTS 처리 중단.
        """
        if not self._enable_tts_interrupt:
            logging.warning("TTS interrupt is not enabled")
            return

        # 큐 비우기
        while not self._pending_requests.empty():
            try:
                self._pending_requests.get_nowait()
                self._pending_requests.task_done()
            except queue.Empty:
                break

        # TODO: 현재 재생 중단
        # self._audio_stream.interrupt()

        self._set_state(TTSState.IDLE)
        logging.info("TTS interrupted")

    def start(self) -> None:
        """
        TTS 서비스 시작.
        """
        if self.running:
            logging.warning("TTSProvider is already running")
            return

        self.running = True

        # TODO: AudioOutputStream 시작
        # self._audio_stream.start()

        # 처리 스레드 시작
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()

        self._set_state(TTSState.IDLE)
        logging.info("TTSProvider started")

    def stop(self) -> None:
        """
        TTS 서비스 정지.
        """
        if not self.running:
            logging.warning("TTSProvider is not running")
            return

        self.running = False

        # 큐 비우기
        while not self._pending_requests.empty():
            try:
                self._pending_requests.get_nowait()
                self._pending_requests.task_done()
            except queue.Empty:
                break

        # TODO: AudioOutputStream 정지
        # self._audio_stream.stop()

        # 처리 스레드 종료 대기
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        logging.info("TTSProvider stopped")

    def configure(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        backend_api_key: Optional[str] = None,
        backend: Optional[TTSBackend] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        output_format: Optional[str] = None,
        language: Optional[str] = None,
        enable_tts_interrupt: Optional[bool] = None,
        # Naver Clova 전용 파라미터
        naver_client_id: Optional[str] = None,
        naver_client_secret: Optional[str] = None,
        speaker: Optional[str] = None,
        volume: Optional[int] = None,
        speed: Optional[int] = None,
        pitch: Optional[int] = None,
        emotion: Optional[int] = None,
        emotion_strength: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        alpha: Optional[int] = None,
        end_pitch: Optional[int] = None,
    ) -> None:
        """
        Provider 설정 변경.

        Parameters
        ----------
        url : Optional[str]
            새 TTS 서비스 URL
        api_key : Optional[str]
            새 API 키
        backend_api_key : Optional[str]
            새 백엔드 API 키
        backend : Optional[TTSBackend]
            새 TTS 백엔드
        voice_id : Optional[str]
            새 음성 ID (ElevenLabs)
        model_id : Optional[str]
            새 모델 ID (ElevenLabs)
        output_format : Optional[str]
            새 출력 형식
        language : Optional[str]
            새 언어 코드
        enable_tts_interrupt : Optional[bool]
            TTS 인터럽트 활성화 여부

        Naver Clova 전용 Parameters
        ---------------------------
        naver_client_id : Optional[str]
            Naver Cloud Platform Client ID
        naver_client_secret : Optional[str]
            Naver Cloud Platform Client Secret
        speaker : Optional[str]
            Naver Clova 음성 종류
        volume : Optional[int]
            음성 크기 (-5~5)
        speed : Optional[int]
            음성 속도 (-5~10)
        pitch : Optional[int]
            음성 높낮이 (-5~5)
        emotion : Optional[int]
            감정 (0~3)
        emotion_strength : Optional[int]
            감정 강도 (0~2)
        sampling_rate : Optional[int]
            샘플링 레이트 (WAV 전용)
        alpha : Optional[int]
            음색 (-5~5)
        end_pitch : Optional[int]
            끝음 처리 (-5~5)
        """
        restart_needed = False

        if url is not None and url != self.url:
            self.url = url
            restart_needed = True

        if api_key is not None:
            self.api_key = api_key
            restart_needed = True

        if backend_api_key is not None:
            self.backend_api_key = backend_api_key

        if backend is not None and backend != self.backend:
            self.backend = backend
            restart_needed = True

        if voice_id is not None:
            self._voice_id = voice_id

        if model_id is not None:
            self._model_id = model_id

        if output_format is not None:
            self._output_format = output_format

        if language is not None:
            self._language = language

        if enable_tts_interrupt is not None:
            self._enable_tts_interrupt = enable_tts_interrupt
            restart_needed = True

        # Naver Clova 전용 설정
        if naver_client_id is not None:
            self._naver_client_id = naver_client_id

        if naver_client_secret is not None:
            self._naver_client_secret = naver_client_secret

        if speaker is not None:
            self._speaker = speaker

        if volume is not None:
            self._volume = max(-5, min(5, volume))

        if speed is not None:
            self._speed = max(-5, min(10, speed))

        if pitch is not None:
            self._pitch = max(-5, min(5, pitch))

        if emotion is not None:
            self._emotion = max(0, min(3, emotion))

        if emotion_strength is not None:
            self._emotion_strength = max(0, min(2, emotion_strength))

        if sampling_rate is not None:
            valid_rates = [8000, 16000, 24000, 48000]
            if sampling_rate in valid_rates:
                self._sampling_rate = sampling_rate
            else:
                logging.warning(
                    f"Invalid sampling_rate {sampling_rate}. "
                    f"Valid values: {valid_rates}"
                )

        if alpha is not None:
            self._alpha = max(-5, min(5, alpha))

        if end_pitch is not None:
            self._end_pitch = max(-5, min(5, end_pitch))

        if restart_needed and self.running:
            self.stop()
            self.start()
            logging.info("TTSProvider reconfigured and restarted")
