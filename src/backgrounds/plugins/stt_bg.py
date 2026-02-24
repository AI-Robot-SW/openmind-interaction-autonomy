"""
STT Background - Speech-to-Text 백그라운드 서비스

이 모듈은 STTProvider를 초기화하고 관리하는 백그라운드 서비스입니다.

Architecture:
    BackgroundOrchestrator -> STTBg -> STTProvider -> Google Cloud STT API

Dependencies:
    - STTProvider: Speech-to-Text 변환 관리
    - AudioProvider: 오디오 데이터 소스 (콜백 등록)
    - google-cloud-speech: Google Cloud Speech-to-Text 클라이언트

Lifecycle:
    1. BackgroundOrchestrator가 STTBg 인스턴스 생성
    2. STTBg.__init__()에서 STTProvider 초기화 및 시작
    3. AudioProvider 큐를 STTProvider에 연결
    4. run() 메서드가 주기적으로 호출되어 상태 모니터링
    5. 시스템 종료 시 STTProvider 정리

Google Cloud STT:
    - Streaming Recognition API 사용
    - 실시간 음성 인식 지원
    - 중간 결과(interim results) 지원
    - 자동 구두점 삽입 지원
"""

import logging
import time
from typing import List, Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig

from providers.stt_provider import STTProvider, STTBackend
from providers.audio_provider import AudioProvider
from providers.speaker_provider import SpeakerProvider


class STTBgConfig(BackgroundConfig):
    """
    STT Background 설정.

    Parameters
    ----------
    backend : str
        STT 백엔드 ("google_cloud", "google_websocket", "local")
    credentials_path : Optional[str]
        Google Cloud 서비스 계정 JSON 파일 경로
    api_key : Optional[str]
        API 키 (WebSocket 백엔드용)
    base_url : Optional[str]
        ASR 서비스 기본 URL (WebSocket 백엔드용)
    language : str
        음성 인식 언어
    enable_interim_results : bool
        중간 결과 활성화 여부
    sample_rate : int
        오디오 샘플링 레이트
    reconnect_delay_sec : float
        재연결 대기 시간 (초)
    max_reconnect_attempts : int
        최대 재연결 시도 횟수
    health_check_interval_sec : float
        상태 확인 주기 (초)

    Google Cloud STT 전용 Parameters
    --------------------------------
    model : str
        음성 인식 모델 ("latest_long", "latest_short", "phone_call", "video")
    use_enhanced : bool
        Enhanced 모델 사용 여부
    enable_automatic_punctuation : bool
        자동 구두점 삽입 활성화
    enable_speaker_diarization : bool
        화자 분리 활성화
    diarization_speaker_count : int
        예상 화자 수
    """

    backend: str = Field(default="google_cloud", description="STT 백엔드")
    credentials_path: Optional[str] = Field(
        default=None, description="Google Cloud 서비스 계정 JSON 파일 경로"
    )
    api_key: Optional[str] = Field(default=None, description="API 키")
    base_url: Optional[str] = Field(default=None, description="ASR 서비스 기본 URL")
    language: str = Field(default="korean", description="음성 인식 언어")
    enable_interim_results: bool = Field(
        default=True, description="중간 결과 활성화 여부"
    )
    sample_rate: int = Field(default=16000, description="오디오 샘플링 레이트")
    reconnect_delay_sec: float = Field(default=5.0, description="재연결 대기 시간 (초)")
    max_reconnect_attempts: int = Field(default=3, description="최대 재연결 시도 횟수")
    health_check_interval_sec: float = Field(
        default=10.0, description="상태 확인 주기 (초)"
    )

    # Google Cloud STT 전용 설정
    model: str = Field(default="latest_long", description="음성 인식 모델")
    use_enhanced: bool = Field(default=True, description="Enhanced 모델 사용 여부")
    enable_automatic_punctuation: bool = Field(
        default=True, description="자동 구두점 삽입 활성화"
    )
    enable_speaker_diarization: bool = Field(
        default=False, description="화자 분리 활성화"
    )
    diarization_speaker_count: int = Field(default=2, description="예상 화자 수")


class STTBg(Background[STTBgConfig]):
    """
    STT Background - Speech-to-Text 백그라운드 서비스.

    STTProvider를 초기화하고 관리합니다.
    AudioProvider로부터 오디오 데이터를 받아 STT 서비스로 전달합니다.

    Attributes
    ----------
    config : STTBgConfig
        백그라운드 설정
    stt_provider : STTProvider
        STT Provider 인스턴스

    Methods
    -------
    run()
        백그라운드 루프 실행 (상태 모니터링)
    """

    def __init__(self, config: STTBgConfig):
        """
        STT Background 초기화.

        Parameters
        ----------
        config : STTBgConfig
            백그라운드 설정
        """
        super().__init__(config)

        # 설정 추출
        backend = config.backend
        credentials_path = config.credentials_path
        api_key = config.api_key
        base_url = config.base_url or "wss://api.openmind.org/api/core/google/asr"
        if api_key and backend == "google_websocket":
            base_url = f"{base_url}?api_key={api_key}"

        language = config.language
        enable_interim_results = config.enable_interim_results
        sample_rate = config.sample_rate

        # Google Cloud STT 전용 설정
        model = config.model
        use_enhanced = config.use_enhanced
        enable_automatic_punctuation = config.enable_automatic_punctuation
        enable_speaker_diarization = config.enable_speaker_diarization
        diarization_speaker_count = config.diarization_speaker_count

        # STTProvider 초기화
        try:
            backend_enum = STTBackend(backend)
        except ValueError:
            logging.warning(
                "Unknown STT backend '%s'. Defaulting to google_cloud.", backend
            )
            backend_enum = STTBackend.GOOGLE_CLOUD

        self.stt_provider = STTProvider(
            backend=backend_enum,
            ws_url=base_url,
            api_key=api_key,
            credentials_path=credentials_path,
            language=language,
            enable_interim_results=enable_interim_results,
            sample_rate=sample_rate,
            # Google Cloud STT 전용
            model=model,
            use_enhanced=use_enhanced,
            enable_automatic_punctuation=enable_automatic_punctuation,
            enable_speaker_diarization=enable_speaker_diarization,
            diarization_speaker_count=diarization_speaker_count,
        )
        self.stt_provider.start()

        # AudioProvider 큐 소비 연결 (AudioBg에서 stream start 담당)
        self._audio_provider = AudioProvider()
        self.stt_provider.attach_audio_provider(self._audio_provider)
        if not self._audio_provider.running:
            logging.warning(
                "AudioProvider is not running. Start AudioBg to capture microphone audio."
            )

        # Speaker↔STT 에코 방지: 스피커 재생 중 STT 일시 중지
        self._speaker_provider = SpeakerProvider()
        self._speaker_provider.register_playback_callback(self._on_playback_state)

        self._last_health_check = time.time()
        self._consecutive_failures = 0
        self._reconnect_attempts = 0

        logging.info(
            f"STTBg initialized: backend={backend}, language={language}, "
            f"model={model}"
        )

    def _on_playback_state(self, state: str) -> None:
        """
        SpeakerProvider 재생 상태 변경 콜백.

        스피커 재생 중에는 STT를 일시 중지하여
        자기 음성이 인식되는 것을 방지합니다.

        Parameters
        ----------
        state : str
            재생 상태 ("started", "playing", "paused", "stopped", "completed")
        """
        if state == "playing":
            self.stt_provider.pause()
        elif state in ("completed", "stopped"):
            self.stt_provider.resume()

    def _health_check(self) -> bool:
        """
        STTProvider 상태 확인.

        Returns
        -------
        bool
            Provider가 정상이면 True
        """
        return bool(self.stt_provider.running)

    def _reconnect(self) -> bool:
        """
        STTProvider 재연결 시도.

        Returns
        -------
        bool
            재연결 성공 시 True
        """
        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logging.error(
                f"STTProvider reconnect failed after "
                f"{self.config.max_reconnect_attempts} attempts"
            )
            return False

        self._reconnect_attempts += 1
        logging.warning(
            f"Attempting STTProvider reconnect "
            f"({self._reconnect_attempts}/{self.config.max_reconnect_attempts})..."
        )

        self.stt_provider.stop()
        time.sleep(self.config.reconnect_delay_sec)
        self.stt_provider.start()

        if self._health_check():
            self._reconnect_attempts = 0
            logging.info("STTProvider reconnected successfully")
            return True

        return False

    def run(self) -> None:
        """
        백그라운드 루프 실행.

        주기적으로 STTProvider 상태를 확인하고,
        연결이 끊어지면 재연결을 시도합니다.
        """
        current_time = time.time()

        # 상태 확인 주기 체크
        if current_time - self._last_health_check < self.config.health_check_interval_sec:
            time.sleep(1.0)
            return

        self._last_health_check = current_time

        # 상태 확인
        if self._health_check():
            self._consecutive_failures = 0
            self._reconnect_attempts = 0
            logging.debug("STTProvider health check: OK")
        else:
            self._consecutive_failures += 1
            logging.warning(
                f"STTProvider health check: FAILED ({self._consecutive_failures})"
            )

            # 재연결 시도
            if not self._reconnect():
                logging.error("STTProvider is unavailable")

        time.sleep(1.0)
