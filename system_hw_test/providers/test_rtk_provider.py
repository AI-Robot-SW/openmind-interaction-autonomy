# test_rtk_provider.py

"""
RtkProvider — Hardware test script.

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

from providers.rtk_provider import RtkProvider


def main() -> int:
    p = argparse.ArgumentParser(description="RtkProvider hardware test")
    p.add_argument("--user",     default="dori0126", help="NTRIP username")
    p.add_argument("--password", default="ngii", help="NTRIP password")
    args = p.parse_args()

    RtkProvider.reset()  # type: ignore[attr-defined]
    rtk_provider = RtkProvider(user=args.user, password=args.password)

    try:
        rtk_provider.start()
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return 1

    print("  Live print (Ctrl+C to quit)")
    last_t = 0.0

    def _fmt_deg(v: float | None, width: int) -> str:
        return f"{v:+{width}.6f}" if v is not None else " " * (width - 4) + "None"

    def _fmt_hacc(v: float | None) -> str:
        return f"{v:+8.3f}" if v is not None else "    None"

    def _fmt_carr(v: int | None) -> str:
        return f"{v:1d}" if v is not None else "-"

    try:
        while True:
            rec = rtk_provider.data
            if rec.t_monotonic != last_t:
                last_t = rec.t_monotonic
                print(
                    f"  lat={_fmt_deg(rec.lat, 10)}  lon={_fmt_deg(rec.lon, 11)}  "
                    f"hAcc={_fmt_hacc(rec.hAcc_m)}  carr={_fmt_carr(rec.carrSoln)}",
                    end="\r",
                )
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

    print()
    rtk_provider.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

