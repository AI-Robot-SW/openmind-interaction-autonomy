"""
TTS Background - Text-to-Speech 백그라운드 서비스

이 모듈은 TTSProvider를 초기화하고 관리하는 백그라운드 서비스입니다.

Architecture:
    BackgroundOrchestrator -> TTSBg -> TTSProvider -> TTS Service
                                            ↓
                                      SpeakerProvider

Dependencies:
    - TTSProvider: Text-to-Speech 변환 관리
    - SpeakerProvider: 오디오 출력 (콜백 등록)

Lifecycle:
    1. BackgroundOrchestrator가 TTSBg 인스턴스 생성
    2. TTSBg.__init__()에서 TTSProvider 초기화 및 시작
    3. SpeakerProvider에 오디오 콜백 등록
    4. run() 메서드가 주기적으로 호출되어 상태 모니터링
    5. 시스템 종료 시 TTSProvider 정리
"""

import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig

from providers.tts_provider import TTSProvider, TTSBackend
from providers.speaker_provider import SpeakerProvider


class TTSBgConfig(BackgroundConfig):
    """
    TTS Background 설정.

    Parameters
    ----------
    api_key : Optional[str]
        OM API 키
    backend_api_key : Optional[str]
        백엔드 서비스 API 키 (예: ElevenLabs)
    backend : str
        TTS 백엔드 ("elevenlabs", "naver_clova", "riva", "google", "local")
    voice_id : str
        음성 ID (ElevenLabs)
    model_id : str
        모델 ID (ElevenLabs)
    output_format : str
        출력 오디오 형식
        - ElevenLabs: "mp3_44100_128", "pcm_16000", etc.
        - Naver Clova: "mp3", "wav"
    language : str
        언어 코드
    enable_tts_interrupt : bool
        TTS 인터럽트 활성화 여부
    health_check_interval_sec : float
        상태 확인 주기 (초)

    Naver Clova 전용 Parameters
    ---------------------------
    naver_client_id : Optional[str]
        Naver Cloud Platform Client ID
    naver_client_secret : Optional[str]
        Naver Cloud Platform Client Secret
    speaker : str
        Naver Clova 음성 종류 (기본값: "nara")
    volume : int
        음성 크기 (-5~5)
    speed : int
        음성 속도 (-5~10)
    pitch : int
        음성 높낮이 (-5~5)
    emotion : int
        감정 (0: 중립, 1: 슬픔, 2: 기쁨, 3: 분노)
    emotion_strength : int
        감정 강도 (0~2)
    sampling_rate : int
        샘플링 레이트 (WAV 전용: 8000, 16000, 24000, 48000)
    """

    api_key: Optional[str] = Field(default=None, description="OM API 키")
    backend_api_key: Optional[str] = Field(
        default=None, description="백엔드 서비스 API 키"
    )
    backend: str = Field(default="elevenlabs", description="TTS 백엔드")
    voice_id: str = Field(
        default=None, description="음성 ID (ElevenLabs), 환경변수 ELEVENLABS_VOICE_ID에서 로드"
    )
    model_id: str = Field(default="eleven_flash_v2_5", description="모델 ID (ElevenLabs)")
    output_format: str = Field(default="mp3_44100_128", description="출력 오디오 형식")
    language: str = Field(default="ko", description="언어 코드")
    enable_tts_interrupt: bool = Field(
        default=False, description="TTS 인터럽트 활성화 여부"
    )
    health_check_interval_sec: float = Field(
        default=10.0, description="상태 확인 주기 (초)"
    )

    # Naver Clova 전용 설정
    naver_client_id: Optional[str] = Field(
        default=None, description="Naver Cloud Platform Client ID"
    )
    naver_client_secret: Optional[str] = Field(
        default=None, description="Naver Cloud Platform Client Secret"
    )
    speaker: str = Field(default="nara", description="Naver Clova 음성 종류")
    volume: int = Field(default=0, description="음성 크기 (-5~5)")
    speed: int = Field(default=0, description="음성 속도 (-5~10)")
    pitch: int = Field(default=0, description="음성 높낮이 (-5~5)")
    emotion: int = Field(default=0, description="감정 (0~3)")
    emotion_strength: int = Field(default=1, description="감정 강도 (0~2)")
    sampling_rate: int = Field(default=24000, description="샘플링 레이트 (WAV 전용)")


class TTSBg(Background[TTSBgConfig]):
    """
    TTS Background - Text-to-Speech 백그라운드 서비스.

    TTSProvider를 초기화하고 관리합니다.
    생성된 오디오를 SpeakerProvider로 전달합니다.

    Attributes
    ----------
    config : TTSBgConfig
        백그라운드 설정
    tts_provider : TTSProvider
        TTS Provider 인스턴스

    Methods
    -------
    run()
        백그라운드 루프 실행 (상태 모니터링)
    """

    def __init__(self, config: TTSBgConfig):
        """
        TTS Background 초기화.

        Parameters
        ----------
        config : TTSBgConfig
            백그라운드 설정
        """
        super().__init__(config)

        # 설정 추출
        api_key = config.api_key
        backend_api_key = config.backend_api_key
        backend = config.backend
        voice_id = config.voice_id
        model_id = config.model_id
        output_format = config.output_format
        language = config.language
        enable_tts_interrupt = config.enable_tts_interrupt

        # Naver Clova 전용 설정
        naver_client_id = config.naver_client_id
        naver_client_secret = config.naver_client_secret
        speaker = config.speaker
        volume = config.volume
        speed = config.speed
        pitch = config.pitch
        emotion = config.emotion
        emotion_strength = config.emotion_strength
        sampling_rate = config.sampling_rate

        # TTSProvider 초기화
        backend_enum = TTSBackend(backend)
        self.tts_provider = TTSProvider(
            api_key=api_key,
            backend_api_key=backend_api_key,
            backend=backend_enum,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            language=language,
            enable_tts_interrupt=enable_tts_interrupt,
            # Naver Clova 전용
            naver_client_id=naver_client_id,
            naver_client_secret=naver_client_secret,
            speaker=speaker,
            volume=volume,
            speed=speed,
            pitch=pitch,
            emotion=emotion,
            emotion_strength=emotion_strength,
            sampling_rate=sampling_rate,
        )
        self.tts_provider.start()

        # SpeakerProvider에 오디오 콜백 등록
        # SpeakerProvider는 Singleton이므로 이미 생성된 인스턴스를 가져옴
        speaker_provider = SpeakerProvider()
        self.tts_provider.register_audio_callback(speaker_provider.queue_audio)

        self._last_health_check = time.time()
        self._consecutive_failures = 0
        self._max_failures = 3

        # 로깅 - 백엔드에 따라 다른 정보 표시
        if backend == "naver_clova":
            logging.info(
                f"TTSBg initialized: backend={backend}, "
                f"speaker={speaker}, format={output_format}"
            )
        else:
            logging.info(
                f"TTSBg initialized: backend={backend}, "
                f"voice={voice_id}, model={model_id}"
            )

    def _health_check(self) -> bool:
        """
        TTSProvider 상태 확인.

        Returns
        -------
        bool
            Provider가 정상이면 True
        """
        return self.tts_provider.running

    def _restart_provider(self) -> None:
        """
        TTSProvider 재시작.
        """
        logging.warning("Restarting TTSProvider...")
        self.tts_provider.stop()
        time.sleep(0.5)
        self.tts_provider.start()
        self._consecutive_failures = 0

    def run(self) -> None:
        """
        백그라운드 루프 실행.

        주기적으로 TTSProvider 상태를 확인하고,
        문제가 있으면 재시작을 시도합니다.
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
            logging.debug("TTSProvider health check: OK")
        else:
            self._consecutive_failures += 1
            logging.warning(
                f"TTSProvider health check: FAILED "
                f"({self._consecutive_failures}/{self._max_failures})"
            )

            if self._consecutive_failures >= self._max_failures:
                self._restart_provider()

        time.sleep(1.0)
