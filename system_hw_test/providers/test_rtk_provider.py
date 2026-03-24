# test_rtk_provider.py

"""
RtkProvider — Hardware test script.

Tested APIs:
  start, stop, data (property)

Prerequisites:
  GNSS/RTK 모듈(u-blox 등)을 USB-Serial로 연결
  NTRIP 캐스터 네트워크 접근 가능

Usage:
  python system_hw_test/providers/test_rtk_provider.py
"""

from __future__ import annotations

import argparse
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import numpy as np

from providers.rtk_provider import RtkProvider


_CARR_SOLN = {0: "No RTK", 1: "Float RTK", 2: "Fixed RTK"}
_FIX_TYPE  = {0: "No fix", 1: "Dead reckoning", 2: "2D", 3: "3D", 4: "GNSS+DR", 5: "Time only"}


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="RtkProvider hardware test")
    p.add_argument("--user",     default="",     help="NTRIP username")
    p.add_argument("--password", default="", help="NTRIP password")
    args = p.parse_args()

    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    RtkProvider.reset()  # type: ignore[attr-defined]

    rtk_provider = RtkProvider(user=args.user, password=args.password)
    print("  Provider created")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 1: Start
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 1: Start\n{'='*60}")
    try:
        rtk_provider.start()
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return 1
    print("  rtk.start()\n  OK")

    # -------------------------------------------------------------------------
    # Phase 2: First record verification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: First record verification\n{'='*60}")
    rec = rtk_provider.data
    assert rec is not None

    fix_label  = _FIX_TYPE.get(rec.fixType or 0, "?")
    carr_label = _CARR_SOLN.get(rec.carrSoln or 0, "?")

    print(f"  t_monotonic    : {rec.t_monotonic:.3f}")
    print(f"  time (UTC)     : {rec.hour:02d}:{rec.minute:02d}:{rec.second:02d}  validTime={rec.validTime}")
    print(f"  fixType        : {rec.fixType}  ({fix_label})")
    print(f"  carrSoln       : {rec.carrSoln}  ({carr_label})")
    print(f"  diffSoln       : {rec.diffSoln}")
    print(f"  numSV          : {rec.numSV}")
    print(f"  lat / lon      : {rec.lat}  /  {rec.lon}")
    print(f"  hAcc_m         : {rec.hAcc_m}")
    print(f"  pDOP           : {rec.pDOP}")

    if rec.fixType is None or rec.fixType < 3:
        print("  WARN: 3D fix 미확보 — 안테나 시야각 확인 필요")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 3: Rate & accuracy stats (10초간 수집)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 3: Rate & accuracy stats (10s)\n{'='*60}")
    records = []
    last_t = 0.0
    t_end = time.monotonic() + 10.0

    while time.monotonic() < t_end:
        rec = rtk_provider.data
        if rec is not None and rec.t_monotonic != last_t:
            last_t = rec.t_monotonic
            records.append(rec)
        time.sleep(0.005)

    if len(records) < 2:
        print("  WARN: 수집된 레코드 부족 — GNSS 신호 상태 확인 필요")
    else:
        ts      = np.array([r.t_monotonic for r in records])
        diffs   = np.diff(ts)
        rate    = 1.0 / diffs.mean() if diffs.mean() > 0 else 0.0
        hacc    = [r.hAcc_m  for r in records if r.hAcc_m  is not None]
        nsv     = [r.numSV   for r in records if r.numSV   is not None]
        carrs   = [r.carrSoln for r in records if r.carrSoln is not None]

        print(f"  samples        : {len(records)}")
        print(f"  rate           : {rate:.1f} Hz")
        print(f"  interval mean  : {diffs.mean() * 1000:.1f} ms")
        print(f"  interval p95   : {np.percentile(diffs, 95) * 1000:.1f} ms")

        if hacc:
            ha = np.array(hacc)
            print(f"  hAcc_m         : min={ha.min():.3f}  mean={ha.mean():.3f}  max={ha.max():.3f}")

        if nsv:
            print(f"  numSV          : min={min(nsv)}  mean={np.mean(nsv):.1f}  max={max(nsv)}")

        if carrs:
            rtk_fixed = sum(1 for c in carrs if c == 2)
            rtk_float = sum(1 for c in carrs if c == 1)
            print(f"  carrSoln dist  : Fixed={rtk_fixed}  Float={rtk_float}  None={len(carrs)-rtk_fixed-rtk_float}")

    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 4: Live print (Ctrl+C to quit)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Live print (Ctrl+C to quit)\n{'='*60}")
    last_t = 0.0

    try:
        while True:
            rec = rtk_provider.data
            if rec is not None and rec.t_monotonic != last_t:
                last_t = rec.t_monotonic
                carr_label = _CARR_SOLN.get(rec.carrSoln or 0, "?")
                print(
                    f"  [{carr_label:10s}]  "
                    f"lat={rec.lat}  lon={rec.lon}  "
                    f"hAcc={rec.hAcc_m:.3f}m  nSV={rec.numSV}",
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
    rtk_provider.stop()
    print("  rtk.stop()\n  OK")

    print("\n  All phases complete. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
