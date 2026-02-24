"""
Sound Sensor - Input Orchestrator Sound 모듈

이 모듈은 STTProvider의 인식 결과를 콜백으로 수신하여
Fuser에 전달합니다.

Architecture:
    STTProvider -> (register_result_callback) -> SoundSensor -> Fuser

Dependencies:
    - STTProvider: Speech-to-Text 결과 콜백 소스
"""

import asyncio
import logging
import time
from queue import Empty, Queue
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput

from providers.stt_provider import STTProvider


class SoundSensorConfig(SensorConfig):
    """
    Sound Sensor 설정.

    Parameters
    ----------
    language : str
        음성 인식 언어 코드
    """

    language: str = Field(default="korean", description="음성 인식 언어")


class SoundSensor(FuserInput[SoundSensorConfig, Optional[str]]):
    """
    Sound Sensor - 음성 입력 처리 센서.

    STTProvider로부터 인식 결과를 콜백으로 수신하고,
    InputOrchestrator에 전달합니다.

    Attributes
    ----------
    descriptor_for_LLM : str
        LLM에 전달될 입력 소스 설명자
    message_buffer : Queue[str]
        STT 결과 메시지 버퍼
    messages : list[str]
        처리된 메시지 목록
    """

    def __init__(self, config: SoundSensorConfig):
        super().__init__(config)

        # LLM 입력 설명자
        self.descriptor_for_LLM = "Voice"

        # 메시지 버퍼
        self.message_buffer: Queue[str] = Queue()
        self.messages: list[str] = []

        # STTProvider 싱글턴에 결과 콜백 등록
        self._stt_provider = STTProvider()
        self._stt_provider.register_result_callback(self._handle_stt_result)

        logging.info(
            "SoundSensor initialized: language=%s, connected to STTProvider",
            config.language,
        )

    def _handle_stt_result(self, text: str) -> None:
        """STT 결과 콜백 핸들러."""
        if text and len(text.strip()) > 0:
            self.message_buffer.put(text)
            logging.debug("STT result received: %s", text)

    async def _poll(self) -> Optional[str]:
        """메시지 버퍼에서 새 메시지를 폴링."""
        await asyncio.sleep(0.1)
        try:
            message = self.message_buffer.get_nowait()
            return message
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """원시 입력을 Message 객체로 변환."""
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]) -> None:
        """원시 입력을 처리하고 메시지 목록에 추가."""
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message.message)
            else:
                # 연속된 메시지는 하나로 결합
                self.messages[-1] = f"{self.messages[-1]} {pending_message.message}"

    def formatted_latest_buffer(self) -> Optional[str]:
        """최신 버퍼 내용을 LLM 입력 형식으로 포맷팅."""
        if len(self.messages) == 0:
            return None

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{self.messages[-1]}
// END
"""
        # 버퍼 초기화
        self.messages = []
        return result
