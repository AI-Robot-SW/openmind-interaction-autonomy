# test_uwb_provider.py

"""
UwbProvider — Hardware test script.

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

from providers.uwb_provider import UwbProvider


def main() -> int:
    UwbProvider.reset()  # type: ignore[attr-defined]
    uwb_provider = UwbProvider()

    try:
        try:
            uwb_provider.start()
        except Exception as e:
            print(f"  FAIL: {e}")
            return 1

        print("  Live print (Ctrl+C to quit)")
        last_t = 0.0

        def _fmt_pos(v: float | None) -> str:
            return f"{v:+7.2f}" if v is not None else "   None"

        def _fmt_qf(v: int | None) -> str:
            return f"{v:3d}" if v is not None else "---"

        try:
            while True:
                rec = uwb_provider.data
                if rec.t_monotonic != last_t:
                    last_t = rec.t_monotonic
                    print(
                        f"  x={_fmt_pos(rec.x_m)}  y={_fmt_pos(rec.y_m)}  "
                        f"z={_fmt_pos(rec.z_m)}  qf={_fmt_qf(rec.quality_factor)}",
                        end="\r",
                    )
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

        print()
        return 0

    finally:
        try:
            uwb_provider.stop()
        except Exception as e:
            print(f"  WARN: stop failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
