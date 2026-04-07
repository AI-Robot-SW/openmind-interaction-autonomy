"""
Text Sensor - WebSocket 기반 텍스트 입력 센서

STT 없이 WebSocket으로 텍스트를 직접 LLM에 전달합니다.
주로 HW 테스트 환경에서 음성 대신 텍스트로 LLM을 테스트할 때 사용합니다.

사용법:
    # wscat 설치: npm install -g wscat
    wscat -c ws://localhost:8765
    > 북문으로 가줘

    # 또는 websocat 사용
    websocat ws://localhost:8765
"""

import asyncio
import logging
import threading
import time
from queue import Empty, Queue
from typing import List, Optional

import websockets
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class TextSensorConfig(SensorConfig):
    """
    Text Sensor 설정.

    Parameters
    ----------
    host : str
        WebSocket 서버 호스트 주소
    port : int
        WebSocket 서버 포트 번호
    """

    host: str = Field(default="0.0.0.0", description="WebSocket 서버 호스트 주소")
    port: int = Field(default=8765, description="WebSocket 서버 포트 번호")


class TextSensor(FuserInput[TextSensorConfig, Optional[str]]):
    """
    WebSocket을 통해 텍스트 입력을 수신하는 센서.

    STT 없이 텍스트를 직접 LLM에 전달합니다.
    SoundSensor와 동일한 descriptor_for_LLM("Voice")을 사용하므로
    기존 system_prompt를 그대로 활용할 수 있습니다.
    """

    def __init__(self, config: TextSensorConfig):
        super().__init__(config)

        self.descriptor_for_LLM = "Voice"
        self.messages: List[Message] = []
        self.message_buffer: Queue[str] = Queue()
        self.io_provider = IOProvider()

        self._host = config.host
        self._port = config.port
        self._server = None
        self._connected_clients: set = set()

        self._server_thread = threading.Thread(
            target=self._start_server_thread, daemon=True
        )
        self._server_thread.start()

    def _start_server_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._start_server())
        try:
            loop.run_forever()
        finally:
            loop.close()

    async def _start_server(self) -> None:
        try:
            self._server = await websockets.serve(
                self._handle_client, self._host, self._port
            )
            logging.info(
                "TextSensor WebSocket server started at ws://%s:%d",
                self._host,
                self._port,
            )
        except Exception as e:
            logging.error("Failed to start TextSensor WebSocket server: %s", e)

    async def _handle_client(self, websocket, path: str) -> None:
        self._connected_clients.add(websocket)
        logging.info(
            "TextSensor: client connected (total: %d)", len(self._connected_clients)
        )
        try:
            async for raw in websocket:
                text = raw if isinstance(raw, str) else raw.decode("utf-8")
                text = text.strip()
                if text:
                    logging.info("TextSensor received: %s", text)
                    self.message_buffer.put(text)
                    self.io_provider.add_mode_transition_input(text)
                    await websocket.send(f"[수신됨] {text}")
        except websockets.exceptions.ConnectionClosed:
            logging.info("TextSensor: client disconnected")
        finally:
            self._connected_clients.discard(websocket)

    async def _poll(self) -> Optional[str]:
        await asyncio.sleep(0.1)
        try:
            return self.message_buffer.get_nowait()
        except Empty:
            return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        if raw_input is None:
            return None
        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]) -> None:
        pending = await self._raw_to_text(raw_input)
        if pending is None:
            return

        if not self.messages:
            self.messages.append(pending)
        else:
            prev = self.messages[-1]
            self.messages[-1] = Message(
                timestamp=pending.timestamp,
                message=f"{prev.message} {pending.message}",
            )

    def formatted_latest_buffer(self) -> Optional[str]:
        if not self.messages:
            return None

        latest = self.messages[-1]
        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__,
            latest.message,
            latest.timestamp,
        )

        self.messages = []
        return result
