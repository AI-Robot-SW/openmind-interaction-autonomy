#!/usr/bin/env python3
"""
Unitree Go2 Provider — Hardware test script.

Step-by-step test of UnitreeGo2Provider against a real Go2 robot on the network.
Covers all Provider APIs; every movement/posture action requires user confirmation
before execution (no auto-advance with timers).

Provider APIs tested:
  start, stop, heartbeat, data (property),
  stand_up, stand_down, damp, sit, rise_sit, recovery_stand,
  move, stop_move.

Prerequisites:
- Go2 powered on and on same network (Ethernet or WiFi as per Unitree docs).
- PYTHONPATH includes project root so 'providers' and 'unitree' are importable.

Usage:
  uv run python system_hw_test/unitree_go2_provider_hw_test.py [--channel eth0] [--timeout 10] [--yes]
  uv run python system_hw_test/unitree_go2_provider_hw_test.py --setup-only   # import + Provider creation only
  uv run python system_hw_test/unitree_go2_provider_hw_test.py --connect-only # setup + connect + heartbeat + data + stop
"""

import argparse
import sys

# Allow importing from project src (providers, unitree bridge)
if "__file__" in dir():
    import os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _src = os.path.join(_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

try:
    from providers.unitree_go2_provider import UnitreeGo2Provider
except ImportError as e:
    print(f"Import error: {e}. Run with PYTHONPATH=src (e.g. uv run python system_hw_test/unitree_go2_provider_hw_test.py)")
    sys.exit(1)


def prompt(msg: str, skip: bool) -> None:
    """Wait for user Enter unless skip is True."""
    if skip:
        return
    try:
        input(msg)
    except EOFError:
        pass


def confirm_before_action(description: str, skip: bool) -> None:
    """Ask user to confirm before a movement/posture action."""
    print(f"  >> {description}")
    prompt("     Press Enter to run (or Ctrl+C to abort)... ", skip)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unitree Go2 Provider hardware test — all APIs, user confirmation for every movement."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="",
        help="Ethernet channel for DDS (e.g. eth0). Empty to skip ChannelFactoryInitialize.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="RPC timeout in seconds (default 10.0).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip 'Press Enter' prompts (use only when no robot or for connect-only check).",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run only Phase 0 (Provider creation) and exit. No robot required.",
    )
    parser.add_argument(
        "--connect-only",
        action="store_true",
        help="Run Phases 0–2 and 5: setup, connect, heartbeat, data, stop. No posture/move.",
    )
    args = parser.parse_args()
    skip = args.yes

    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    prompt("  Press Enter to continue... ", skip)
    UnitreeGo2Provider.reset()  # type: ignore
    provider = UnitreeGo2Provider(channel=args.channel.strip(), timeout=args.timeout)
    print(f"  Provider created: channel={args.channel!r}, timeout={args.timeout}")
    print("  OK")
    if args.setup_only:
        print("  --setup-only: exiting after Phase 0.")
        return 0

    # -------------------------------------------------------------------------
    # Phase 1: Connect (start)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 1: Connect (start)\n{'='*60}")
    prompt("  Press Enter to call start()... ", skip)
    provider.start()
    if not provider._running or provider._sport_client is None:
        print("  FAIL: start() did not set _running or _sport_client. Check network and SDK.")
        return 1
    print("  OK: SportClient initialized")

    # -------------------------------------------------------------------------
    # Phase 2: Read-only — heartbeat, data
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: Read-only (heartbeat, data)\n{'='*60}")
    prompt("  Press Enter to run heartbeat()... ", skip)
    ok = provider.heartbeat()
    if not ok:
        print("  FAIL: heartbeat() returned False. Robot may be unreachable or timeout too short.")
        provider.stop()
        return 1
    print("  heartbeat() -> True  OK")

    print("  data property (read-only):")
    prompt("  Press Enter to read provider.data... ", skip)
    d = provider.data
    print(f"  provider.data = {d}")
    print("  OK")
    if args.connect_only:
        print("  --connect-only: skipping posture and move; calling stop().")
        provider.stop()
        print("  stop() done. OK")
        print("\n  Connect-only run complete. Done.")
        return 0

    # -------------------------------------------------------------------------
    # Phase 3: Posture — stand_up, stand_down, damp, sit, rise_sit, recovery_stand
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 3: Posture APIs (each requires confirmation)\n{'='*60}")

    confirm_before_action("stand_up() — robot will stand up", skip)
    ok = provider.stand_up()
    print(f"  stand_up() -> {ok}")

    confirm_before_action("stand_down() — robot will crouch/lie", skip)
    ok = provider.stand_down()
    print(f"  stand_down() -> {ok}")

    confirm_before_action("stand_up() — robot will stand again", skip)
    ok = provider.stand_up()
    print(f"  stand_up() -> {ok}")

    confirm_before_action("damp() — robot enters damping (can be moved by hand)", skip)
    ok = provider.damp()
    print(f"  damp() -> {ok}")

    confirm_before_action("stand_up() — robot will stand from damp", skip)
    ok = provider.stand_up()
    print(f"  stand_up() -> {ok}")

    confirm_before_action("sit() — robot will sit down", skip)
    ok = provider.sit()
    print(f"  sit() -> {ok}")

    confirm_before_action("rise_sit() — robot will stand from sit", skip)
    ok = provider.rise_sit()
    print(f"  rise_sit() -> {ok}")

    confirm_before_action("recovery_stand() — robot will attempt recovery stand (use if fallen)", skip)
    ok = provider.recovery_stand()
    print(f"  recovery_stand() -> {ok}")

    # -------------------------------------------------------------------------
    # Phase 4: Move — move, stop_move
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Move APIs (each requires confirmation)\n{'='*60}")

    confirm_before_action("move(0, 0, 0) — send zero velocity (no motion expected)", skip)
    ok = provider.move(0.0, 0.0, 0.0)
    print(f"  move(0,0,0) -> {ok}")

    confirm_before_action("stop_move() — stop locomotion", skip)
    ok = provider.stop_move()
    print(f"  stop_move() -> {ok}")

    # Optional: minimal forward move (user confirms)
    confirm_before_action("move(0.1, 0, 0) — robot may move forward slightly", skip)
    ok = provider.move(0.1, 0.0, 0.0)
    print(f"  move(0.1,0,0) -> {ok}")

    confirm_before_action("stop_move() — stop locomotion", skip)
    ok = provider.stop_move()
    print(f"  stop_move() -> {ok}")

    # -------------------------------------------------------------------------
    # Phase 5: Teardown (stop)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 5: Teardown (stop)\n{'='*60}")
    prompt("  Press Enter to call stop()... ", skip)
    provider.stop()
    print("  stop() done. OK")

    print("\n  All Provider APIs exercised. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
