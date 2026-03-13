#!/usr/bin/env python3
"""
Move action × real HW test — LLM output mocking.

Overview
--------
- connect(): 상위 목표만 설정 (예: set_goal("L1"), stand_up). 실제 이동/속도 명령은 안 보냄.
- tick(): 주기적으로 get_next_move() → (vx, vy, vyaw) 또는 None → move() / stop_move() 로 로봇 구동.

이 스크립트는:
1. YAML 시퀀스를 읽어서, 정해진 주기(output_interval_sec)마다 connect(MoveInput(...))만 호출 (LLM 출력 모사).
2. tick()은 스크립트가 직접 호출하지 않음. ActionOrchestrator.start() 시 각 connector마다
   별도 스레드에서 _run_connector_loop()가 돌면서 action.connector.tick()을 반복 호출함.
   → src/actions/orchestrator.py 의 start() / _run_connector_loop() 참고.

Usage:
  uv run python system_hw_test/move_action_hw_test.py [--config system_hw_test/move_action_sequence.yaml] [--yes]
  uv run python system_hw_test/move_action_hw_test.py --config system_hw_test/move_action_sequence.yaml --yes

Config (YAML):
  output_interval_sec: 1.0   # default delay after each output when not specified
  repeat_count: 1           # 1 = run sequence once, 2 = twice, 0 = infinite (Ctrl+C to stop)
  sequence:
    - "stand up"                                    # use default delay
    - { action: "go to L8", delay_after_sec: 10 }  # path is chosen inside MoveConnector.connect()
    - { action: "speed up", delay_after_sec: 8 }
    - { action: "stand down", delay_after_sec: 0 }
  Order and delay_after_sec are fully configurable per step.
"""

import argparse
import asyncio
import logging
import sys
import threading
import time
from pathlib import Path

if "__file__" in dir():
    _root = Path(__file__).resolve().parent.parent
    _src = _root / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import yaml

from actions import load_action
from actions.move.interface import MoveInput, MovementAction


class ConnectorTickRunner:
    """Minimal thread runner for connector.tick() used by this HW test."""

    def __init__(self, connector):
        self._connector = connector
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="MoveConnectorTickRunner",
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._connector.tick()
            except Exception as e:
                logging.error("Error in connector tick loop: %s", e)
                time.sleep(0.1)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def load_sequence_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "sequence" not in data:
        raise ValueError(f"YAML must have 'sequence' list: {path}")
    return data


def normalize_sequence(sequence: list, default_interval: float) -> list[tuple[str, float]]:
    """Convert sequence to list of (action_value, delay_after_sec)."""
    out = []
    for i, item in enumerate(sequence):
        if isinstance(item, str):
            action_val = item
            # default delay; last item can use 0 if you want to end immediately
            delay = default_interval
        elif isinstance(item, dict) and "action" in item:
            action_val = item["action"]
            delay = float(item.get("delay_after_sec", default_interval))
        else:
            raise ValueError(f"sequence[{i}] must be a string or {{ action: ..., delay_after_sec: ... }}: {item}")
        out.append((action_val, delay))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Move action HW test — mock LLM output from YAML sequence."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "move_action_sequence.yaml",
        help="Path to YAML sequence config (default: move_action_sequence.yaml in same dir).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip any confirmation prompts.",
    )
    return p.parse_args()


def build_move_connector():
    """Load and return the move action connector used by this HW test."""
    action = load_action(
        {
            "name": "move",
            "llm_label": "move",
            "connector": "move_connector",
            "config": {},
        }
    )
    return action.connector


async def run_sequence(
    connector,
    steps: list[tuple[str, float]],
    repeat_count: int,
):
    # repeat_count 0 = infinite; else run the full sequence this many times
    round_num = 0
    step_num = 0
    try:
        while True:
            if repeat_count != 0 and round_num >= repeat_count:
                break
            if round_num > 0:
                print(f"  --- round {round_num + 1} ---")
            for value, delay_after_sec in steps:
                try:
                    action_enum = MovementAction(value)
                except ValueError:
                    print(f"  Skip unknown action value: {value!r}")
                    await asyncio.sleep(delay_after_sec)
                    continue
                move_input = MoveInput(action=action_enum)
                await connector.connect(move_input)
                step_num += 1
                print(f"  [{step_num}] connect({value!r}) → wait {delay_after_sec}s")
                await asyncio.sleep(delay_after_sec)
            round_num += 1
            if repeat_count == 0:
                continue  # infinite: run sequence again
    except asyncio.CancelledError:
        pass


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config_path = args.config
    if not config_path.is_file():
        print(f"Config not found: {config_path}")
        return 1

    print(f"Load config: {config_path}")
    try:
        seq_config = load_sequence_config(config_path)
    except Exception as e:
        print(f"Failed to load YAML: {e}")
        return 1

    default_interval = float(seq_config.get("output_interval_sec", 1.0))
    repeat = int(seq_config.get("repeat_count", 1))
    try:
        steps = normalize_sequence(list(seq_config["sequence"]), default_interval)
    except ValueError as e:
        print(f"Invalid sequence: {e}")
        return 1

    print(f"  output_interval_sec (default) = {default_interval}")
    print(f"  repeat_count = {repeat}")
    print(f"  sequence steps = {len(steps)}")
    if not args.yes:
        try:
            input("Press Enter to start (Ctrl+C to abort)... ")
        except EOFError:
            pass

    connector = build_move_connector()
    tick_runner = ConnectorTickRunner(connector)
    tick_runner.start()

    try:
        asyncio.run(run_sequence(connector, steps, repeat))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        tick_runner.stop()
        print("Tick runner stopped. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
