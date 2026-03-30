# test_uwb_provider.py

"""
UwbProvider — Hardware test script.

Tested APIs:
  start, stop, data (property)

Prerequisites:
  UWB 모듈(Tag)을 USB-Serial로 연결

Usage:
  python system_hw_test/providers/test_uwb_provider.py
"""

from __future__ import annotations

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import numpy as np

from providers.uwb_provider import UwbProvider


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    UwbProvider.reset()  # type: ignore[attr-defined]

    uwb_provider = UwbProvider()
    print("  Provider created")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 1: Start
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 1: Start\n{'='*60}")
    try:
        uwb_provider.start()
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return 1
    print("  uwb_provider.start()\n  OK")

    # -------------------------------------------------------------------------
    # Phase 2: First record verification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: First record verification\n{'='*60}")
    rec = uwb_provider.data
    assert rec is not None

    print(f"  t_monotonic    : {rec.t_monotonic:.3f}")
    print(f"  x_m            : {rec.x_m}")
    print(f"  y_m            : {rec.y_m}")
    print(f"  z_m            : {rec.z_m}")
    print(f"  quality_factor : {rec.quality_factor}")

    if rec.x_m is None or rec.y_m is None:
        print("  WARN: x_m / y_m가 None — UWB 앵커 연결 확인 필요")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 3: Rate & quality stats (5초간 수집)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 3: Rate & quality stats (5s)\n{'='*60}")
    records = []
    last_t = 0.0
    t_end = time.monotonic() + 5.0

    while time.monotonic() < t_end:
        rec = uwb_provider.data
        if rec is not None and rec.t_monotonic != last_t:
            last_t = rec.t_monotonic
            records.append(rec)
        time.sleep(0.005)

    if len(records) < 2:
        print("  WARN: 수집된 레코드 부족 — UWB 신호 상태 확인 필요")
    else:
        ts = np.array([r.t_monotonic for r in records])
        diffs = np.diff(ts)
        rate = 1.0 / diffs.mean() if diffs.mean() > 0 else 0.0
        qfs = [r.quality_factor for r in records if r.quality_factor is not None]

        print(f"  samples        : {len(records)}")
        print(f"  rate           : {rate:.1f} Hz")
        print(f"  interval mean  : {diffs.mean() * 1000:.1f} ms")
        print(f"  interval p95   : {np.percentile(diffs, 95) * 1000:.1f} ms")

        if qfs:
            qf_arr = np.array(qfs)
            print(f"  quality_factor : min={qf_arr.min()}  mean={qf_arr.mean():.1f}  max={qf_arr.max()}")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 4: Live print (Ctrl+C to quit)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Live print (Ctrl+C to quit)\n{'='*60}")
    last_t = 0.0

    try:
        while True:
            rec = uwb_provider.data
            if rec is not None and rec.t_monotonic != last_t:
                last_t = rec.t_monotonic
                print(
                    f"  x={rec.x_m:+.3f}m  y={rec.y_m:+.3f}m  "
                    f"z={rec.z_m:+.3f}m  qf={rec.quality_factor}",
                    end="\r",
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    print()
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 5: Teardown
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 5: Teardown\n{'='*60}")
    uwb_provider.stop()
    print("  provider.stop()\n  OK")

    print("\n  All phases complete. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())