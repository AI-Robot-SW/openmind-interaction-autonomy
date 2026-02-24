"""
Audio Background - 오디오 입력 백그라운드 서비스

이 모듈은 AudioProvider를 초기화하고 관리하는 백그라운드 서비스입니다.

Architecture:
    BackgroundOrchestrator -> AudioBg -> AudioProvider -> Microphone

Dependencies:
    - AudioProvider: 오디오 입력 스트림 관리

Lifecycle:
    1. BackgroundOrchestrator가 AudioBg 인스턴스 생성
    2. AudioBg.__init__()에서 AudioProvider 초기화 및 시작
    3. run() 메서드가 주기적으로 호출되어 상태 모니터링
    4. 시스템 종료 시 AudioProvider 정리
"""

import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.audio_provider import AudioProvider


class AudioBgConfig(BackgroundConfig):
    """
    Audio Background 설정.

    Parameters
    ----------
    sample_rate : int
        오디오 샘플링 레이트 (Hz)
    chunk_size : int
        오디오 청크 크기 (samples)
    channels : int
        오디오 채널 수
    device_id : Optional[int]
        마이크 디바이스 ID
    device_name : Optional[str]
        마이크 디바이스 이름
    vad_enabled : bool
        VAD 활성화 여부
    vad_threshold : float
        VAD 임계값
    health_check_interval_sec : float
        상태 확인 주기 (초)
    """

    sample_rate: int = Field(default=16000, description="오디오 샘플링 레이트 (Hz)")
    chunk_size: int = Field(default=1024, description="오디오 청크 크기")
    channels: int = Field(default=1, description="오디오 채널 수")
    device_id: Optional[int] = Field(default=None, description="마이크 디바이스 ID")
    device_name: Optional[str] = Field(default=None, description="마이크 디바이스 이름")
    vad_enabled: bool = Field(default=True, description="VAD 활성화 여부")
    vad_threshold: float = Field(default=0.5, description="VAD 임계값")
    health_check_interval_sec: float = Field(
        default=10.0, description="상태 확인 주기 (초)"
    )


class AudioBg(Background[AudioBgConfig]):
    """
    Audio Background - 오디오 입력 백그라운드 서비스.

    AudioProvider를 초기화하고 관리합니다.
    주기적으로 Provider 상태를 확인하고 필요시 재시작합니다.

    Attributes
    ----------
    config : AudioBgConfig
        백그라운드 설정
    audio_provider : AudioProvider
        오디오 입력 Provider 인스턴스

    Methods
    -------
    run()
        백그라운드 루프 실행 (상태 모니터링)
    """

    def __init__(self, config: AudioBgConfig):
        """
        Audio Background 초기화.

        Parameters
        ----------
        config : AudioBgConfig
            백그라운드 설정
        """
        super().__init__(config)

        # 설정 추출
        sample_rate = config.sample_rate
        chunk_size = config.chunk_size
        channels = config.channels
        device_id = config.device_id
        device_name = config.device_name
        vad_enabled = config.vad_enabled
        vad_threshold = config.vad_threshold

        # AudioProvider 초기화 + 스트림 시작
        self.audio_provider = AudioProvider(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            channels=channels,
            device_id=device_id,
            device_name=device_name,
            vad_enabled=vad_enabled,
            vad_threshold=vad_threshold,
        )
        self.audio_provider.start_stream()

        self._last_health_check = time.time()
        self._consecutive_failures = 0
        self._max_failures = 3

        logging.info(
            f"AudioBg initialized: rate={sample_rate}, "
            f"device={device_id or device_name or 'default'}"
        )

    def _health_check(self) -> bool:
        """
        AudioProvider 상태 확인.

        Returns
        -------
        bool
            Provider가 정상이면 True
        """
        return bool(self.audio_provider.running)

    def _restart_provider(self) -> None:
        """
        AudioProvider 재시작.
        """
        logging.warning("Restarting AudioProvider...")
        self.audio_provider.stop()
        time.sleep(0.5)
        self.audio_provider.start_stream()
        self._consecutive_failures = 0

    def run(self) -> None:
        """
        백그라운드 루프 실행.

        주기적으로 AudioProvider 상태를 확인하고,
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
            logging.debug("AudioProvider health check: OK")
        else:
            self._consecutive_failures += 1
            logging.warning(
                f"AudioProvider health check: FAILED "
                f"({self._consecutive_failures}/{self._max_failures})"
            )

            if self._consecutive_failures >= self._max_failures:
                self._restart_provider()

        time.sleep(1.0)
