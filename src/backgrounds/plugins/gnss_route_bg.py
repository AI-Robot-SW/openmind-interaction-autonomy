# gnss_route_bg.py

import time
import math
import logging
import threading
from typing import List, Optional, Tuple

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.gnss_route_provider import GnssRouteProvider

# TODO : WaypointTracker 의존성 제거 또는 수정
# from providers.gnss_route_provider import WaypointTracker


class GnssRouteBgConfig(BackgroundConfig):
    # TODO : 수정된 GnssRouteProvider의 초기화 파라미터 반영 (max_vx, max_vyaw)
    # waypoint_path: Optional[str] = Field(
    #     default=None,
    #     description="Path to waypoint CSV file with lat/lon columns",
    # )
    # waypoints: Optional[List[Tuple[float, float]]] = Field(
    #     default=None,
    #     description="Inline waypoint list as [(lat, lon), ...]",
    # )
    # reach_tol_m: float = Field(
    #     default=5.0,
    #     description="Waypoint reach tolerance in meters",
    # )

    max_vx: float = Field(
        default=0.8,
        description="Maximum forward velocity in m/s",
    )
    max_vyaw: float = Field(
        default=math.radians(45),
        description="Maximum yaw velocity in rad/s",
    )


# TODO : 불필요 필드 제거 또는 수정
# def _resolve_waypoints(
#     config: GnssRouteBgConfig,
# ) -> List[Tuple[float, float]]:
#     if config.waypoints is not None:
#         return [(float(lat), float(lon)) for lat, lon in config.waypoints]

#     if config.waypoint_path is not None:
#         tracker = WaypointTracker.from_file(
#             config.waypoint_path,
#             reach_tol=config.reach_tol_m,
#         )
#         return list(tracker._coords)

#     logging.info("GnssRouteBg: no initial waypoints configured, starting with empty path")
#     return []


class GnssRouteBg(Background[GnssRouteBgConfig]):
    """
    GNSS Route Provider Background.

    Initializes and starts GnssRouteProvider in the background.
    """

    def __init__(self, config: GnssRouteBgConfig):
        super().__init__(config)

        # TODO : GnssRouteProvider의 초기화 파라미터 반영 (max_vx, max_vyaw)
        self.gnss_route_provider = GnssRouteProvider(
            # waypoints=_resolve_waypoints(self.config),
            # reach_tol_m=self.config.reach_tol_m,
            max_vx=self.config.max_vx,
            max_vyaw=self.config.max_vyaw,
        )
        self.gnss_route_provider.start()

        logging.info(
            "GnssRouteProvider initialized and started in background "
            # f"(waypoints={len(_resolve_waypoints(self.config))}, reach_tol_m={self.config.reach_tol_m}, "
            f"max_vx={self.config.max_vx}, max_vyaw={self.config.max_vyaw})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.gnss_route_provider.stop()
            logging.info("GnssRouteProvider stopped")
            return
        time.sleep(1.0)
