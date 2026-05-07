#!/usr/bin/env python3
"""
record_route.py — EKF 위치 자동 기록기

KfPositionProvider(RTK EKF + UWB EKF) 출력을 주기적으로 CSV에 자동 기록한다.
Ctrl+C 로 종료.

출력 CSV 컬럼:
  RTK : lat, lon
  UWB : x, y

출력 파일은 route_utils 지도 편집기의 CSV overlay 파서(parseCsvOverlay)와
직접 호환된다.

실행 예:
  # 기본 파일명, 0.5 s 간격으로 저장
  python -m scripts.record_route

  # 파일명 / 기록 간격 직접 지정
  python -m scripts.record_route \\
      --output-rtk logs/route_rtk.csv \\
      --output-uwb logs/route_uwb.csv \\
      --interval 1.0
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from src.providers.rtk_provider import RtkProvider
from src.providers.uwb_provider import UwbProvider
from src.providers.unitree_go2_provider import UnitreeGo2Provider
from src.providers.kf_position_provider import KfPositionProvider, KfPositionRecord


_STATUS_WIDTH = 80


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EKF 위치 자동 기록기 — Ctrl+C 로 종료",
    )
    p.add_argument("--output-rtk", default="logs/route_rtk.csv",
                   help="RTK 출력 CSV 경로 (기본: logs/route_rtk.csv)")
    p.add_argument("--output-uwb", default="logs/route_uwb.csv",
                   help="UWB 출력 CSV 경로 (기본: logs/route_uwb.csv)")
    p.add_argument("--interval",   default=0.1, type=float,
                   help="기록 간격 (초, 기본: 0.5)")
    p.add_argument("--ntrip-user", default="dori0126")
    p.add_argument("--ntrip-pass", default="ngii")
    p.add_argument("--uwb-port",   default="/dev/uwb")
    p.add_argument("--uwb-baud",   default=115200, type=int)
    p.add_argument("--robot-iface", default="eno1")
    return p.parse_args()


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

class _CsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()
        self._count = 0

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()
        self._count += 1

    def close(self) -> None:
        self._file.close()

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    args = _parse_args()

    UnitreeGo2Provider.reset()  # type: ignore[attr-defined]
    robot = UnitreeGo2Provider(channel=args.robot_iface)
    robot.start()
    RtkProvider.reset()       # type: ignore[attr-defined]
    rtk = RtkProvider(user=args.ntrip_user, password=args.ntrip_pass)
    rtk.start()
    UwbProvider.reset()       # type: ignore[attr-defined]
    uwb = UwbProvider(port=args.uwb_port, baud=args.uwb_baud)
    uwb.start()
    KfPositionProvider.reset()  # type: ignore[attr-defined]
    route = KfPositionProvider()

    rtk_writer = _CsvWriter(Path(args.output_rtk), ["lat", "lon"])
    uwb_writer = _CsvWriter(Path(args.output_uwb), ["x", "y"])

    print(f"RTK 출력: {args.output_rtk}")
    print(f"UWB 출력: {args.output_uwb}")
    print(f"기록 간격: {args.interval} s  |  Ctrl+C 로 종료")
    print()

    route.start()

    try:
        while True:
            rec: KfPositionRecord | None = route.data

            # ── RTK EKF ──────────────────────────────────────────
            if rec is not None and rec.rtk_ready and rec.rtk_lat is not None:
                rtk_writer.write({
                    "lat": f"{rec.rtk_lat:.9f}",
                    "lon": f"{rec.rtk_lon:.9f}",
                })
                cal = "CAL" if rec.rtk_yaw_calibrated else "uncal"
                rtk_status = (
                    f"RTK({cal}) #{rtk_writer.count:4d}  "
                    f"lat={rec.rtk_lat:.7f}  lon={rec.rtk_lon:.7f}"
                    f"  std={rec.rtk_std_xy_m:.3f}m"
                )
            else:
                rtk_status = "RTK  ---  (init...)"

            # ── UWB EKF ──────────────────────────────────────────
            if rec is not None and rec.uwb_ready and rec.uwb_x_m is not None:
                uwb_writer.write({
                    "x": f"{rec.uwb_x_m:.4f}",
                    "y": f"{rec.uwb_y_m:.4f}",
                })
                cal = "CAL" if rec.uwb_yaw_calibrated else "uncal"
                uwb_status = (
                    f"UWB({cal}) #{uwb_writer.count:4d}  "
                    f"x={rec.uwb_x_m:+.3f}  y={rec.uwb_y_m:+.3f}"
                    f"  std={rec.uwb_std_xy_m:.3f}m"
                )
            else:
                uwb_status = "UWB  ---  (init...)"

            sys.stdout.write(("\r" + f"{rtk_status}  |  {uwb_status}").ljust(_STATUS_WIDTH))
            sys.stdout.flush()

            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        print(f"기록 완료 — RTK {rtk_writer.count}개 / UWB {uwb_writer.count}개")
        rtk_writer.close()
        uwb_writer.close()
        route.stop()
        rtk.stop()
        uwb.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
