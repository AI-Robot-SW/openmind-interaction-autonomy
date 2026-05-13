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

from providers.io_provider import IOProvider
from providers.stt_provider import STTProvider

# 목적지 키워드 — 매칭 시 IOProvider에 저장하여 모드 전환 후에도 기억
_DESTINATION_KEYWORDS = ["L1", "L2", "L3", "북문","본관"]
# 도착 키워드 — 매칭 시 저장된 목적지 초기화
_ARRIVAL_KEYWORDS = ["도착", "arrived"]


class SoundSensorConfig(SensorConfig):
    """
    Sound Sensor 설정.

    Parameters
    ----------
    language : str
        음성 인식 언어 코드
    """

    language: str = Field(default="korean", description="음성 인식 언어")
    dedupe_window_s: float = Field(
        default=2.0,
        description="동일 STT 최종 결과를 중복으로 간주할 시간 창 (초)",
    )


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
    messages : list[Message]
        처리된 메시지 목록
    """

    def __init__(self, config: SoundSensorConfig):
        super().__init__(config)

        # LLM 입력 설명자
        self.descriptor_for_LLM = "Voice"

        # 메시지 버퍼
        self.message_buffer: Queue[str] = Queue()
        self.messages: list[Message] = []
        self._last_stt_text: Optional[str] = None
        self._last_stt_time: float = 0.0

        # IOProvider 싱글턴 — 모드 전환 입력 전달용 + Fuser 입력 기록용
        self.io_provider = IOProvider()

        # STTProvider 싱글턴에 결과 콜백 등록
        self._stt_provider = STTProvider()
        self._stt_provider.register_result_callback(self._handle_stt_result)

        logging.info(
            "SoundSensor initialized: language=%s, connected to STTProvider",
            config.language,
        )

    def cleanup(self) -> None:
        """모드 전환 시 STTProvider 콜백 해제."""
        self._stt_provider.unregister_result_callback(self._handle_stt_result)
        logging.debug("SoundSensor cleanup: STT callback unregistered")

    def _handle_stt_result(self, text: str) -> None:
        """STT 결과 콜백 핸들러."""
        if text and len(text.strip()) > 0:
            now = time.time()
            normalized_text = " ".join(text.strip().split())
            if (
                self._last_stt_text == normalized_text
                and now - self._last_stt_time < self.config.dedupe_window_s
            ):
                logging.debug("Duplicate STT result ignored: %s", text)
                return

            self._last_stt_text = normalized_text
            self._last_stt_time = now

            self.message_buffer.put(text)
            self.io_provider.add_mode_transition_input(text)

            # 목적지 키워드 감지 시 IOProvider에 저장 (모드 전환 후에도 기억)
            text_lower = text.lower()
            for dest in _DESTINATION_KEYWORDS:
                if dest.lower() in text_lower:
                    self.io_provider.add_dynamic_variable("last_destination", dest)
                    logging.info("Destination detected and stored: %s", dest)
                    break

            # 도착 키워드 감지 시 저장된 목적지 초기화
            for kw in _ARRIVAL_KEYWORDS:
                if kw.lower() in text_lower:
                    self.io_provider.add_dynamic_variable("last_destination", None)
                    logging.info("Arrival detected, destination cleared")
                    break

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
                self.messages.append(pending_message)
            else:
                # 연속된 메시지는 하나로 결합
                prev = self.messages[-1]
                self.messages[-1] = Message(
                    timestamp=pending_message.timestamp,
                    message=f"{prev.message} {pending_message.message}",
                )

    def formatted_latest_buffer(self) -> Optional[str]:
        """최신 버퍼 내용을 LLM 입력 형식으로 포맷팅."""
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__,
            latest_message.message,
            latest_message.timestamp,
        )

        # 버퍼 초기화
        self.messages = []
        return result