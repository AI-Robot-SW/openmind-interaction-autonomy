# examples/gnss_route_provider.py
"""
GnssRouteProvider 사용 예시.

실행 전 준비:
  - LocationProvider (GNSS/UWB) background 실행 중
  - UnitreeGo2Provider background 실행 중

사용법:
  1. txt/csv 파일에서 경로 로드
  2. 직접 waypoint 지정

파일 포맷 (txt/csv):
  time,lat,lon,hdop,quality
  1751794886.136273,37.6038667,127.0453007,0.56,2
  ...
"""

import time
import logging

from providers.gnss_route_provider import GnssRouteProvider, WaypointTracker

logger = logging.getLogger(__name__)


def example_load_from_file():
    """txt 파일에서 경로를 로드해서 주행하는 예시."""
    path = WaypointTracker.from_file("route.txt")  # providers/ 폴더 기준 상대경로 또는 절대경로
    provider = GnssRouteProvider(waypoints=path._coords)
    provider.start()

    try:
        while True:
            rec = provider.get_record()

            if not rec.heading_calibrated:
                logger.info("캘리브레이션 중...")
            elif rec.reached_goal:
                logger.info("목적지 도달!")
                break
            else:
                logger.info(
                    "주행 중 | heading=%.1f° vx=%.2f vyaw=%.3f",
                    rec.global_heading_deg,
                    rec.vx,
                    rec.vyaw,
                )

            time.sleep(1.0)
    finally:
        provider.stop()


def example_direct_waypoints():
    """waypoint를 직접 지정해서 주행하는 예시."""
    waypoints = [
        (37.6038550, 127.0452932),
        (37.6038160, 127.0452640),
        (37.6037820, 127.0452512),
        (37.6037638, 127.0452402),
    ]

    provider = GnssRouteProvider(
        waypoints=waypoints,
        reach_tol_m=5.0,
        max_vx=0.8,
    )
    provider.start()

    try:
        while True:
            rec = provider.get_record()

            if rec.reached_goal:
                logger.info("목적지 도달!")
                break

            # 주행 중 속도 변경 예시
            if rec.heading_calibrated and rec.vx > 0.5:
                provider.max_vx = 0.6

            time.sleep(1.0)
    finally:
        provider.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_load_from_file()
