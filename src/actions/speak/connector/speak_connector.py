"""
Speak Connector - SpeakAction을 TTSProvider로 연결

이 모듈은 SpeakInput.action 텍스트를 TTSProvider 싱글턴으로 라우팅합니다.

Architecture:
    LLM -> SpeakAction -> SpeakConnector.connect() -> TTSProvider().add_pending_message()
                                                           ↓ (TTSBg에서 등록됨)
                                                     SpeakerProvider().queue_audio()

Dependencies:
    - TTSProvider: Text-to-Speech 변환 (Singleton)
    - SpeakerProvider: 오디오 출력 (Singleton, interrupt용)
"""

import logging
from typing import Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput

from providers.tts_provider import TTSProvider
from providers.speaker_provider import SpeakerProvider


class SpeakConnectorConfig(ActionConfig):
    """
    Speak Connector 설정.

    Parameters
    ----------
    enable_tts_interrupt : bool
        새 발화 시 현재 재생을 중단할지 여부
    """

    enable_tts_interrupt: bool = Field(
        default=True, description="새 발화 시 현재 재생을 중단할지 여부"
    )


class SpeakConnector(ActionConnector[SpeakConnectorConfig, SpeakInput]):
    """
    SpeakAction을 TTSProvider 싱글턴으로 연결하는 커넥터.

    SpeakInput.action 텍스트를 TTSProvider에 전달하여
    TTS 변환 및 스피커 출력을 수행합니다.
    """

    def __init__(self, config: SpeakConnectorConfig):
        super().__init__(config)

        self._tts_provider = TTSProvider()
        self._speaker_provider = SpeakerProvider()

        self.tts_enabled = True

        logging.info("SpeakConnector initialized")

    async def connect(self, output_interface: SpeakInput) -> None:
        """
        SpeakInput을 TTSProvider로 전달.

        Parameters
        ----------
        output_interface : SpeakInput
            발화할 텍스트를 담은 인터페이스
        """
        if not self.tts_enabled:
            logging.warning("TTS is disabled. Ignoring speak request.")
            return

        text = output_interface.action
        if not text or not text.strip():
            logging.debug("Empty text, skipping TTS")
            return

        # interrupt가 활성화되면 현재 재생을 중단
        if getattr(output_interface, "interrupt", self.config.enable_tts_interrupt):
            self._speaker_provider.stop_playback()

        # TTSProvider에 텍스트 전달
        self._tts_provider.add_pending_message(text)

        logging.debug("SpeakConnector: text queued for TTS: %s", text[:50])
