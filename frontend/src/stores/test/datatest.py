#!/usr/bin/env python3
"""
WebSocket payload debugger for frontend stores (audioStore / ttsStore / navigationStore).

Goal: debug in terminal (no browser console) whether these values are arriving:
- voice_level (AudioBg -> GUIBg /voice_spectrum payload `level`)
- voice_active (payload `voice_active`)
- tts `state` (payload `state`)
- displayText (payload `text`)
- speakerPlaying (payload `speaker_playing`)
- navigation activeGoal (GUIBg /navigation payload `active_goal`)
- navigation reached (payload `reached_goal`)

Run (recommended, uses repo venv):
  .venv/bin/python frontend/src/stores/test/datatest.py

Optional:
  .venv/bin/python frontend/src/stores/test/datatest.py --base ws://192.168.180.49:8767
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import websockets


DEFAULT_WS_BASE = "ws://localhost:8767"
ALLOWED_TTS_STATES = {"idle", "processing", "speaking", "error"}


def _read_frontend_env_ws_base() -> Optional[str]:
    """
    Read VITE_GUI_WS_BASE from frontend/.env without requiring python-dotenv.
    """
    # Prefer runtime env var (works both locally and in containers).
    env = os.environ.get("VITE_GUI_WS_BASE")
    if env:
        return env.strip()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    env_path = os.path.join(repo_root, "frontend", ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "VITE_GUI_WS_BASE":
                    return v.strip()
    except FileNotFoundError:
        return None
    except Exception:
        return None

    return None


def _coerce_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
    return None


def _coerce_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if x != x:  # NaN
        return None
    return x


@dataclass
class DebugState:
    voice_level: Optional[float] = None
    voice_active: Optional[bool] = None

    tts_state: Optional[str] = None
    display_text: str = ""
    speaker_playing: Optional[bool] = None

    nav_active_goal: Optional[str] = None
    nav_reached_goal: Optional[bool] = None

    # meta
    updates: int = 0
    last_print_ts: float = field(default_factory=lambda: 0.0)

    def _print(self, line: str) -> None:
        # Keep printing simple and reliable for terminal debugging.
        ts = time.strftime("%H:%M:%S")
        print(f"{ts} {line}", flush=True)

    def on_update(self, source: str, **fields: Any) -> None:
        changed: dict[str, Any] = {}
        for k, v in fields.items():
            if getattr(self, k) != v:
                setattr(self, k, v)
                changed[k] = v

        self.updates += 1
        if changed:
            self._print(f"[{source}] update {json.dumps(changed, ensure_ascii=False)}")

        # Periodic snapshot (every ~2s) so you can see steady-state even if nothing changes.
        now = time.time()
        if now - self.last_print_ts >= 2.0:
            self.last_print_ts = now
            snapshot = {
                "voice_level": self.voice_level,
                "voice_active": self.voice_active,
                "tts_state": self.tts_state,
                "display_text": (self.display_text[:60] + "…")
                if len(self.display_text) > 60
                else self.display_text,
                "speaker_playing": self.speaker_playing,
                "nav_active_goal": self.nav_active_goal,
                "nav_reached_goal": self.nav_reached_goal,
            }
            self._print(f"[snapshot] {json.dumps(snapshot, ensure_ascii=False)}")


async def _listen_voice_spectrum(ws_url: str, st: DebugState, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            async with websockets.connect(ws_url) as ws:
                st._print(f"[voice] connected {ws_url}")
                async for raw in ws:
                    if stop.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        st._print(f"[voice] invalid_json raw={raw!r}")
                        continue

                    level = _coerce_float(msg.get("level"))
                    voice_active_raw = msg.get("voice_active", msg.get("voiceActive"))
                    voice_active = _coerce_bool(voice_active_raw)

                    if level is None:
                        st._print(f"[voice] missing/invalid level msg={msg!r}")
                    if voice_active is None:
                        st._print(f"[voice] missing/invalid voice_active msg={msg!r}")

                    st.on_update("voice", voice_level=level, voice_active=voice_active)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            st._print(f"[voice] disconnected ({type(e).__name__}: {e}); retrying in 1s")
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def _listen_tts_text(ws_url: str, st: DebugState, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            async with websockets.connect(ws_url) as ws:
                st._print(f"[tts] connected {ws_url}")
                async for raw in ws:
                    if stop.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        st._print(f"[tts] invalid_json raw={raw!r}")
                        continue

                    text_raw = msg.get("text", "")
                    text = str(text_raw) if text_raw is not None else ""

                    state_raw = msg.get("state", msg.get("tts_state", msg.get("ttsState")))
                    state = str(state_raw) if state_raw is not None else None
                    if state is not None and state not in ALLOWED_TTS_STATES:
                        st._print(f"[tts] invalid state={state!r} msg={msg!r}")

                    speaker_playing_raw = msg.get(
                        "speaker_playing", msg.get("speakerPlaying")
                    )
                    speaker_playing = (
                        _coerce_bool(speaker_playing_raw)
                        if speaker_playing_raw is not None
                        else None
                    )
                    if speaker_playing_raw is not None and speaker_playing is None:
                        st._print(f"[tts] invalid speaker_playing msg={msg!r}")

                    st.on_update(
                        "tts",
                        tts_state=state,
                        display_text=text.strip(),
                        speaker_playing=speaker_playing,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            st._print(f"[tts] disconnected ({type(e).__name__}: {e}); retrying in 1s")
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def _listen_navigation(ws_url: str, st: DebugState, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            async with websockets.connect(ws_url) as ws:
                st._print(f"[nav] connected {ws_url}")
                st._print("[nav] waiting for messages...")

                msg_count = 0
                empty_count = 0
                invalid_count = 0
                last_recv_ts = time.time()
                last_stats_ts = 0.0

                while not stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        idle_s = int(time.time() - last_recv_ts)
                        st._print(f"[nav] heartbeat (msgs={msg_count}, idle={idle_s}s)")
                        # Force periodic snapshots even if no messages arrive.
                        st.on_update("nav_heartbeat")
                        continue

                    if raw is None:
                        break

                    last_recv_ts = time.time()
                    msg_count += 1

                    raw_str = (
                        raw.decode("utf-8", "replace")
                        if isinstance(raw, (bytes, bytearray))
                        else str(raw)
                    )
                    if msg_count == 1:
                        st._print(f"[nav] first_message raw={raw_str!r}")

                    try:
                        msg = json.loads(raw_str)
                    except Exception:
                        invalid_count += 1
                        st._print(f"[nav] invalid_json raw={raw_str!r}")
                        continue

                    if not isinstance(msg, dict):
                        invalid_count += 1
                        st._print(f"[nav] invalid_msg_type type={type(msg).__name__} raw={raw!r}")
                        continue

                    active_goal_raw = msg.get("active_goal", msg.get("activeGoal"))
                    if active_goal_raw is None:
                        active_goal = None
                    else:
                        active_goal = str(active_goal_raw)

                    reached_goal_raw = msg.get("reached_goal", msg.get("reachedGoal"))
                    reached_goal = (
                        _coerce_bool(reached_goal_raw)
                        if reached_goal_raw is not None
                        else None
                    )

                    # Empty payload usually means NavigationProvider isn't running yet.
                    if not msg:
                        empty_count += 1
                        st._print(
                            "[nav] empty_payload (NavigationProvider not running? Start NavigationBg)"
                        )
                    else:
                        if "active_goal" not in msg and "activeGoal" not in msg:
                            st._print(f"[nav] missing active_goal msg={msg!r}")
                        if "reached_goal" not in msg and "reachedGoal" not in msg:
                            st._print(f"[nav] missing reached_goal msg={msg!r}")
                        if reached_goal_raw is not None and reached_goal is None:
                            st._print(f"[nav] invalid reached_goal msg={msg!r}")

                    now = time.time()
                    if now - last_stats_ts >= 2.0:
                        last_stats_ts = now
                        st._print(
                            f"[nav] stats total={msg_count} empty={empty_count} invalid={invalid_count}"
                        )

                    st.on_update(
                        "nav",
                        nav_active_goal=active_goal,
                        nav_reached_goal=reached_goal,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            st._print(f"[nav] disconnected ({type(e).__name__}: {e}); retrying in 1s")
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


async def _main_async(base: str, duration_s: Optional[float]) -> int:
    base = base.rstrip("/")
    voice_url = f"{base}/voice_spectrum"
    tts_url = f"{base}/tts_text"
    nav_url = f"{base}/navigation"

    st = DebugState()
    st._print(f"[start] base={base}")
    st._print("[start] expecting /voice_spectrum, /tts_text, /navigation")

    stop = asyncio.Event()
    tasks = [
        asyncio.create_task(_listen_voice_spectrum(voice_url, st, stop)),
        asyncio.create_task(_listen_tts_text(tts_url, st, stop)),
        asyncio.create_task(_listen_navigation(nav_url, st, stop)),
    ]

    try:
        if duration_s is None:
            await asyncio.gather(*tasks)
        else:
            try:
                await asyncio.sleep(duration_s)
            finally:
                stop.set()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Frontend store payload debugger")
    parser.add_argument(
        "--base",
        default=_read_frontend_env_ws_base() or DEFAULT_WS_BASE,
        help="WebSocket base URL (e.g. ws://192.168.180.49:8767)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after N seconds (default: run forever)",
    )
    args = parser.parse_args()

    try:
        return asyncio.run(_main_async(args.base, args.duration))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
