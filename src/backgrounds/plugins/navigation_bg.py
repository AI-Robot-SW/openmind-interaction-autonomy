# navigation_bg.py
"""
Navigation Background — config를 받아 GnssRouteProvider / DwaRouteProvider를
생성하고 NavigationProvider에 주입한 뒤 라이프사이클을 관리합니다.
"""

from __future__ import annotations

import logging
import math
import time
from typing import List, Optional, Tuple

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.gnss_route_provider import GnssRouteProvider
from providers.dwa_route_provider import DwaRouteProvider
from providers.navigation_provider import NavigationProvider

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════


class NavigationBgConfig(BackgroundConfig):
    # ---- waypoints ----
    waypoints: List[Tuple[float, float]] = Field(
        default_factory=list, description="GPS waypoints [(lat, lon), ...]"
    )

    # ---- GnssRouteProvider ----
    gnss_reach_tol_m: float = Field(default=5.0, description="Waypoint reach tolerance (m)")
    gnss_max_vx: float = Field(default=0.8, description="Max forward speed (m/s)")
    gnss_max_vyaw: float = Field(default=math.radians(45), description="Max yaw rate (rad/s)")

    # ---- DwaRouteProvider ----
    dwa_penalty: float = Field(default=13.0)
    dwa_margin: float = Field(default=1.2)
    dwa_w_goal: float = Field(default=1.0)
    dwa_w_clear: float = Field(default=1.2)
    dwa_y_bias: float = Field(default=-0.5)
    dwa_obstacle_cost: float = Field(default=1e9)
    dwa_person_stop_dist: float = Field(default=1.2)
    dwa_person_stop_y_width: float = Field(default=0.5)
    dwa_ahead_m: float = Field(default=2.0)
    dwa_half_width_m: float = Field(default=1.2)
    dwa_stride: int = Field(default=1)
    dwa_unknown_is_obstacle: bool = Field(default=False)
    dwa_kyaw: float = Field(default=1.0)
    dwa_v_max: float = Field(default=0.9)
    dwa_w_max: float = Field(default=0.75)
    dwa_v_min: float = Field(default=0.0)
    dwa_vx_fixed: float = Field(default=0.8)
    dwa_safety_slowdown: bool = Field(default=True)
    dwa_enable_turn_in_place: bool = Field(default=True)
    dwa_theta_turn_deg: float = Field(default=40.0)
    dwa_allow_backward: bool = Field(default=False)
    dwa_dist_method: str = Field(default="bfs_cuda")
    dwa_dist_max_m: float = Field(default=3.0)
    dwa_control_rate_hz: float = Field(default=10.0)
    dwa_log_csv_path: Optional[str] = Field(default=None)

    # ---- NavigationProvider ----
    monitor_rate_hz: float = Field(default=20.0, description="NavigationProvider 폴링 주기 (Hz)")


# ═══════════════════════════════════════════════════════════
# Background
# ═══════════════════════════════════════════════════════════


class NavigationBg(Background[NavigationBgConfig]):
    """
    NavigationBg — config로부터 GnssRouteProvider / DwaRouteProvider를 생성하고
    NavigationProvider에 주입합니다.
    """

    def __init__(self, config: NavigationBgConfig) -> None:
        super().__init__(config)

        gnss = GnssRouteProvider(
            waypoints=config.waypoints,
            reach_tol_m=config.gnss_reach_tol_m,
            max_vx=config.gnss_max_vx,
            max_vyaw=config.gnss_max_vyaw,
        )

        dwa = DwaRouteProvider(
            gnss_route_provider=gnss,
            penalty=config.dwa_penalty,
            margin=config.dwa_margin,
            w_goal=config.dwa_w_goal,
            w_clear=config.dwa_w_clear,
            y_bias=config.dwa_y_bias,
            obstacle_cost=config.dwa_obstacle_cost,
            person_stop_dist=config.dwa_person_stop_dist,
            person_stop_y_width=config.dwa_person_stop_y_width,
            ahead_m=config.dwa_ahead_m,
            half_width_m=config.dwa_half_width_m,
            stride=config.dwa_stride,
            unknown_is_obstacle=config.dwa_unknown_is_obstacle,
            kyaw=config.dwa_kyaw,
            v_max=config.dwa_v_max,
            w_max=config.dwa_w_max,
            v_min=config.dwa_v_min,
            vx_fixed=config.dwa_vx_fixed,
            safety_slowdown=config.dwa_safety_slowdown,
            enable_turn_in_place=config.dwa_enable_turn_in_place,
            theta_turn_deg=config.dwa_theta_turn_deg,
            allow_backward=config.dwa_allow_backward,
            dist_method=config.dwa_dist_method,
            dist_max_m=config.dwa_dist_max_m,
            control_rate_hz=config.dwa_control_rate_hz,
            log_csv_path=config.dwa_log_csv_path,
        )

        self.navigation_provider = NavigationProvider(
            gnss=gnss,
            dwa=dwa,
            tick_dt=1.0 / max(1e-3, config.monitor_rate_hz),
        )

    def run(self) -> None:
        logger.info("NavigationBg: starting NavigationProvider")
        self.navigation_provider.start()
        try:
            while True:
                time.sleep(1.0)
        finally:
            logger.info("NavigationBg: stopping NavigationProvider")
            self.navigation_provider.stop()

