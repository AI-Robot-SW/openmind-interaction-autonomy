"""
Speaker Background - 스피커 출력 백그라운드 서비스

이 모듈은 SpeakerProvider를 초기화하고 관리하는 백그라운드 서비스입니다.

Architecture:
    BackgroundOrchestrator -> SpeakerBg -> SpeakerProvider -> Speaker Device

Dependencies:
    - SpeakerProvider: 오디오 출력 스트림 관리

Lifecycle:
    1. BackgroundOrchestrator가 SpeakerBg 인스턴스 생성
    2. SpeakerBg.__init__()에서 SpeakerProvider 초기화 및 시작
    3. run() 메서드가 주기적으로 호출되어 상태 모니터링
    4. 시스템 종료 시 SpeakerProvider 정리
"""

import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.speaker_provider import SpeakerProvider


class SpeakerBgConfig(BackgroundConfig):
    """
    Speaker Background 설정.

    Parameters
    ----------
    sample_rate : int
        오디오 샘플링 레이트 (Hz)
    channels : int
        오디오 채널 수
    device_id : Optional[int]
        스피커 디바이스 ID
    device_name : Optional[str]
        스피커 디바이스 이름
    volume : float
        초기 볼륨 레벨 (0.0 ~ 1.0)
    buffer_size : int
        오디오 버퍼 크기
    health_check_interval_sec : float
        상태 확인 주기 (초)
    """

    sample_rate: int = Field(default=44100, description="오디오 샘플링 레이트 (Hz)")
    channels: int = Field(default=1, description="오디오 채널 수")
    device_id: Optional[int] = Field(default=None, description="스피커 디바이스 ID")
    device_name: Optional[str] = Field(default=None, description="스피커 디바이스 이름")
    volume: float = Field(default=1.0, description="초기 볼륨 레벨")
    buffer_size: int = Field(default=4096, description="오디오 버퍼 크기")
    health_check_interval_sec: float = Field(
        default=10.0, description="상태 확인 주기 (초)"
    )


class SpeakerBg(Background[SpeakerBgConfig]):
    """
    Speaker Background - 스피커 출력 백그라운드 서비스.

    SpeakerProvider를 초기화하고 관리합니다.
    주기적으로 Provider 상태를 확인하고 필요시 재시작합니다.

    Attributes
    ----------
    config : SpeakerBgConfig
        백그라운드 설정
    speaker_provider : SpeakerProvider
        스피커 출력 Provider 인스턴스

    Methods
    -------
    run()
        백그라운드 루프 실행 (상태 모니터링)
    """

    def __init__(self, config: SpeakerBgConfig):
        """
        Speaker Background 초기화.

        Parameters
        ----------
        config : SpeakerBgConfig
            백그라운드 설정
        """
        super().__init__(config)

        # 설정 추출
        sample_rate = config.sample_rate
        channels = config.channels
        device_id = config.device_id
        device_name = config.device_name
        volume = config.volume
        buffer_size = config.buffer_size

        # SpeakerProvider 초기화 및 시작
        self.speaker_provider = SpeakerProvider(
            sample_rate=sample_rate,
            channels=channels,
            device_id=device_id,
            device_name=device_name,
            volume=volume,
            buffer_size=buffer_size,
        )
        self.speaker_provider.start()

        self._last_health_check = time.time()
        self._consecutive_failures = 0
        self._max_failures = 3

        logging.info(
            f"SpeakerBg initialized: rate={sample_rate}, "
            f"device={device_id or device_name or 'default'}"
        )

    def _health_check(self) -> bool:
        """
        SpeakerProvider 상태 확인.

        Returns
        -------
        bool
            Provider가 정상이면 True
        """
        return self.speaker_provider.running

    def _restart_provider(self) -> None:
        """
        SpeakerProvider 재시작.
        """
        logging.warning("Restarting SpeakerProvider...")
        self.speaker_provider.stop()
        time.sleep(0.5)
        self.speaker_provider.start()
        self._consecutive_failures = 0

    def run(self) -> None:
        """
        백그라운드 루프 실행.

        주기적으로 SpeakerProvider 상태를 확인하고,
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
            logging.debug("SpeakerProvider health check: OK")
        else:
            self._consecutive_failures += 1
            logging.warning(
                f"SpeakerProvider health check: FAILED "
                f"({self._consecutive_failures}/{self._max_failures})"
            )

            if self._consecutive_failures >= self._max_failures:
                self._restart_provider()

        time.sleep(1.0)
