"""
Console Speak Connector - TTS 없이 터미널에 LLM 응답을 출력하는 커넥터

TTS/Speaker 하드웨어 없이 텍스트만으로 LLM과 소통할 때 사용합니다.
LLM 응답을 stdout과 로그에 출력합니다.

Architecture:
    LLM -> SpeakAction -> ConsoleConnector.connect() -> stdout
"""

import logging

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput


class ConsoleConnector(ActionConnector[ActionConfig, SpeakInput]):
    """
    LLM의 speak 응답을 터미널에 출력하는 커넥터.

    TTS, Speaker 하드웨어 의존성 없음.
    """

    def __init__(self, config: ActionConfig):
        super().__init__(config)
        logging.info("ConsoleConnector initialized")

    async def connect(self, output_interface: SpeakInput) -> None:
        text = output_interface.action
        if not text or not text.strip():
            return

        print(f"\n LLM: {text}\n", flush=True)
        logging.info("LLM response: %s", text)
