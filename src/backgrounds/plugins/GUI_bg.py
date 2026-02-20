"""
GUI Background - 음성 볼륨 WebSocket 브로드캐스트 서비스

이 모듈은 AudioProvider의 계산된 볼륨 값을 읽어
WebSocket(`/voice_spectrum`)으로 주기적으로 브로드캐스트합니다.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Optional

import websockets
from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.audio_provider import AudioProvider


class GUIBgConfig(BackgroundConfig):
    """GUI Background 설정."""

    host: str = Field(default="0.0.0.0", description="WebSocket host")
    port: int = Field(default=8767, description="WebSocket port")
    ws_path: str = Field(default="/voice_spectrum", description="WebSocket path")
    broadcast_interval_sec: float = Field(
        default=0.05, description="볼륨 브로드캐스트 주기 (초)"
    )
    health_check_interval_sec: float = Field(
        default=10.0, description="상태 확인 주기 (초)"
    )


class GUIBg(Background[GUIBgConfig]):
    """
    AudioProvider의 볼륨 값을 WebSocket으로 브로드캐스트하는 background.
    """

    def __init__(self, config: GUIBgConfig):
        super().__init__(config)

        self.audio_provider = AudioProvider()
        if not self.audio_provider.running:
            logging.warning(
                "AudioProvider is not running. Start AudioBg first for live volume updates."
            )

        self._connections = set()
        self._server = None
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._shutdown_event = threading.Event()
        self._last_health_check = time.time()

        self._start_server_thread()
        logging.info(
            "GUIBg initialized: ws://%s:%s%s",
            self.config.host,
            self.config.port,
            self.config.ws_path,
        )

    # ---- WebSocket server ----

    def _start_server_thread(self) -> None:
        if self._server_thread is not None and self._server_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._server_thread = threading.Thread(
            target=self._run_server_loop,
            daemon=True,
        )
        self._server_thread.start()
        time.sleep(0.2)

    def _run_server_loop(self) -> None:
        self._server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._server_loop)
        try:
            self._server_loop.run_until_complete(self._start_server())
            self._broadcast_task = self._server_loop.create_task(
                self._broadcast_loop()
            )
            self._server_loop.run_forever()
        except Exception as e:
            logging.error("GUIBg server loop error: %s", e)
        finally:
            try:
                if self._broadcast_task is not None:
                    self._broadcast_task.cancel()
                    try:
                        self._server_loop.run_until_complete(self._broadcast_task)
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logging.error("GUIBg broadcast task cancel error: %s", e)
                    self._broadcast_task = None
                self._server_loop.run_until_complete(self._cleanup_server())
            except Exception as e:
                logging.error("GUIBg cleanup error: %s", e)
            self._server_loop.close()
            self._server_loop = None

    async def _start_server(self) -> None:
        self._server = await websockets.serve(
            self._handle_client,
            self.config.host,
            self.config.port,
        )
        logging.info(
            "GUIBg WebSocket server started: ws://%s:%s%s",
            self.config.host,
            self.config.port,
            self.config.ws_path,
        )

    async def _handle_client(self, websocket) -> None:
        path = getattr(websocket, "path", None)
        if path is None:
            request = getattr(websocket, "request", None)
            path = getattr(request, "path", None)

        if path is not None and path != self.config.ws_path:
            await websocket.close(code=1008, reason=f"Use {self.config.ws_path}")
            return

        self._connections.add(websocket)
        try:
            await websocket.send(json.dumps(self._build_payload()))
            async for _ in websocket:
                # Client messages are ignored; server is push-only.
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logging.error("GUIBg client handler error: %s", e)
        finally:
            self._connections.discard(websocket)

    async def _broadcast_loop(self) -> None:
        interval = max(0.01, float(self.config.broadcast_interval_sec))
        try:
            while not self._shutdown_event.is_set():
                if self._connections:
                    payload = json.dumps(self._build_payload())
                    stale = []
                    for conn in list(self._connections):
                        try:
                            await conn.send(payload)
                        except Exception:
                            stale.append(conn)
                    for conn in stale:
                        self._connections.discard(conn)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    async def _cleanup_server(self) -> None:
        for conn in list(self._connections):
            try:
                await conn.close()
            except Exception:
                pass
        self._connections.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def _build_payload(self) -> float:
        return float(self.audio_provider.get_audio_level())

    # ---- Lifecycle ----

    def _health_check(self) -> bool:
        return bool(self._server_thread and self._server_thread.is_alive())

    def _stop_server(self) -> None:
        self._shutdown_event.set()
        if self._server_loop is not None and self._server_loop.is_running():
            self._server_loop.call_soon_threadsafe(self._server_loop.stop)
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)

    def _restart_server(self) -> None:
        logging.warning("Restarting GUIBg WebSocket server...")
        self._stop_server()
        time.sleep(0.2)
        self._start_server_thread()

    def run(self) -> None:
        current_time = time.time()
        if current_time - self._last_health_check < self.config.health_check_interval_sec:
            time.sleep(1.0)
            return

        self._last_health_check = current_time

        if not self._health_check():
            self._restart_server()

        time.sleep(1.0)

    def __del__(self):
        self._stop_server()
