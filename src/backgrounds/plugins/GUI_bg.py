"""
GUI Background - 음성/내비게이션 WebSocket 브로드캐스트 서비스

이 모듈은 AudioProvider의 계산된 볼륨 값과
NavigationProvider의 최신 상태를 읽어 WebSocket으로 주기적으로 브로드캐스트합니다.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Optional

import websockets
from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.audio_provider import AudioProvider
from providers.navigation_provider import NavigationProvider
from providers.speaker_provider import SpeakerProvider
from providers.tts_provider import TTSProvider


class GUIBgConfig(BackgroundConfig):
    """GUI Background 설정."""

    host: str = Field(default="0.0.0.0", description="WebSocket host")
    port: int = Field(default=8767, description="WebSocket port")
    voice_ws_path: str = Field(default="/voice_spectrum", description="WebSocket path for voice spectrum")
    navi_ws_path: str = Field(default="/navigation", description="WebSocket path for navigation path status")
    tts_ws_path: str = Field(default="/tts_text", description="WebSocket path for TTS text")
    broadcast_interval_sec: float = Field(
        default=0.05, description="볼륨 브로드캐스트 주기 (초)"
    )
    health_check_interval_sec: float = Field(
        default=10.0, description="상태 확인 주기 (초)"
    )


class GUIBg(Background[GUIBgConfig]):
    """
    AudioProvider의 볼륨 값과 NavigationProvider의 최신 상태를
    WebSocket으로 브로드캐스트하는 background.
    """

    def __init__(self, config: GUIBgConfig):
        super().__init__(config)

        self._audio_connections = set()
        self._navi_connections = set()
        self._tts_connections = set()
        self._channels: dict[str, dict[str, Any]] = {
            self.config.voice_ws_path: {
                "connections": self._audio_connections,
                "builder": self._build_audio_payload,
            },
            self.config.navi_ws_path: {
                "connections": self._navi_connections,
                "builder": self._build_navigation_payload,
            },
            self.config.tts_ws_path: {
                "connections": self._tts_connections,
                "builder": self._build_tts_payload,
            },
        }
        self._server = None
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_thread: Optional[threading.Thread] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._shutdown_event = threading.Event()
        self._last_health_check = time.time()
        self._audio_missing_warned = False
        self._navigation_missing_warned = False
        self._tts_missing_warned = False

        self._start_server_thread()
        logging.info(
            "GUIBg initialized: ws://%s:%s (%s)",
            self.config.host,
            self.config.port,
            ", ".join(sorted(self._channels.keys())),
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
            "GUIBg WebSocket server started: ws://%s:%s (%s)",
            self.config.host,
            self.config.port,
            ", ".join(sorted(self._channels.keys())),
        )

    def _get_channel(self, path: Optional[str]) -> Optional[dict[str, Any]]:
        if path is None:
            return None
        return self._channels.get(path)

    @staticmethod
    async def _send_payload(
        websocket,
        builder: Callable[[], Any],
    ) -> None:
        await websocket.send(json.dumps(builder()))

    async def _handle_client(self, websocket) -> None:
        path = getattr(websocket, "path", None)
        if path is None:
            request = getattr(websocket, "request", None)
            path = getattr(request, "path", None)

        channel = self._get_channel(path)
        if channel is None:
            await websocket.close(
                code=1008,
                reason=f"Use one of: {', '.join(sorted(self._channels.keys()))}",
            )
            return

        connections = channel["connections"]
        builder = channel["builder"]
        connections.add(websocket)
        try:
            await self._send_payload(websocket, builder)
            async for _ in websocket:
                # Client messages are ignored; server is push-only.
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logging.error("GUIBg client handler error: %s", e)
        finally:
            connections.discard(websocket)

    async def _broadcast_loop(self) -> None:
        interval = max(0.01, float(self.config.broadcast_interval_sec))
        try:
            while not self._shutdown_event.is_set():
                for channel in self._channels.values():
                    connections = channel["connections"]
                    if not connections:
                        continue
                    payload = json.dumps(channel["builder"]())
                    stale = []
                    for conn in list(connections):
                        try:
                            await conn.send(payload)
                        except Exception:
                            stale.append(conn)
                    for conn in stale:
                        connections.discard(conn)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    async def _cleanup_server(self) -> None:
        for channel in self._channels.values():
            connections = channel["connections"]
            for conn in list(connections):
                try:
                    await conn.close()
                except Exception:
                    pass
            connections.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @staticmethod
    def _get_singleton_instance(factory: Any) -> Any:
        singleton_cls = getattr(factory, "_singleton_class", None)
        if singleton_cls is None:
            return None
        return getattr(singleton_cls, "_singleton_instance", None)

    def _get_audio_provider(self) -> Any:
        provider = self._get_singleton_instance(AudioProvider)
        if provider is None:
            if not self._audio_missing_warned:
                logging.warning(
                    "GUIBg: AudioProvider not initialized yet. Start AudioBg first for live volume updates."
                )
                self._audio_missing_warned = True
            return None
        return provider

    def _get_navigation_provider(self) -> Any:
        provider = self._get_singleton_instance(NavigationProvider)
        if provider is None:
            if not self._navigation_missing_warned:
                logging.warning(
                    "GUIBg: NavigationProvider not initialized yet. Start NavigationBg first for live navigation updates."
                )
                self._navigation_missing_warned = True
            return None
        return provider
    
    def _get_tts_provider(self) -> Any:
        provider = self._get_singleton_instance(TTSProvider)
        if provider is None:
            if not self._tts_missing_warned:
                logging.warning(
                    "GUIBg: TTSProvider not initialized yet. Start TTSBg first for live TTS updates."
                )
                self._tts_missing_warned = True
            return None
        return provider

    def _get_speaker_provider(self) -> Any:
        provider = self._get_singleton_instance(SpeakerProvider)
        if provider is None:
            return None
        return provider
    
    def _build_audio_payload(self) -> dict[str, Any]:
        provider = self._get_audio_provider()
        if provider is None or not provider.running:
            return {"level": 0.0, "voice_active": False}
        return {
            "level": float(provider.get_audio_level()),
            "voice_active": bool(provider.is_voice_active()),
        }

    def _build_navigation_payload(self) -> dict[str, Any]:
        provider = self._get_navigation_provider()
        if provider is None or not provider.running:
            return {}
        data = provider.data
        if data is None:
            return {}
        return dict(data)
    
    def _build_tts_payload(self) -> dict[str, Any]:
        provider = self._get_tts_provider()
        if provider is None or not provider.running:
            return {"text": "", "state": "idle"}
        speaker = self._get_speaker_provider()
        return {
            "text": provider.get_tts_text(),
            "state": provider.get_current_state().value,
            # Helps the frontend reflect "speaking" during actual audio playback.
            "speaker_playing": bool(speaker.is_playing()) if speaker is not None else False,
        }

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
