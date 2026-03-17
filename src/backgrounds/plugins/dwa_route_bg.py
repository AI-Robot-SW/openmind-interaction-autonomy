# dwa_route_bg.py

import time
import logging
import threading

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.dwa_route_provider import DwaRouteProvider


class DwaRouteProviderBgConfig(BackgroundConfig):
    penalty: float = Field(default=13.0, description="Soft obstacle penalty weight")
    margin: float = Field(default=1.2, description="Obstacle slowdown margin in meters")
    w_goal: float = Field(default=1.0, description="Goal alignment cost weight")
    w_clear: float = Field(default=1.2, description="Clearance cost weight")
    y_bias: float = Field(default=-0.5, description="Lateral bias applied to the goal")
    obstacle_cost: float = Field(
        default=1e9,
        description="Hard obstacle cost for occupied cells",
    )
    person_stop_dist: float = Field(
        default=1.2,
        description="Distance threshold for stopping near person occupancy",
    )
    person_stop_y_width: float = Field(
        default=0.5,
        description="Half-width of the person stop corridor",
    )
    ahead_m: float = Field(
        default=2.0,
        description="Forward search window length in meters",
    )
    half_width_m: float = Field(
        default=1.2,
        description="Half-width of the DWA search window in meters",
    )
    stride: int = Field(default=1, description="Sampling stride in grid cells")
    unknown_is_obstacle: bool = Field(
        default=False,
        description="Treat unknown cells as obstacle candidates",
    )
    kv: float = Field(default=0.6, description="Reserved forward gain")
    kyaw: float = Field(default=1.0, description="Yaw control gain")
    v_max: float = Field(default=0.9, description="Maximum forward speed in m/s")
    w_max: float = Field(default=0.75, description="Maximum yaw speed in rad/s")
    v_min: float = Field(default=0.0, description="Minimum forward speed in m/s")
    vx_fixed: float = Field(default=0.8, description="Base forward command in m/s")
    safety_slowdown: bool = Field(
        default=True,
        description="Reduce speed when clearance drops below margin",
    )
    enable_turn_in_place: bool = Field(
        default=True,
        description="Stop forward motion and turn in place for large heading error",
    )
    theta_turn_deg: float = Field(
        default=40.0,
        description="Turn-in-place heading threshold in degrees",
    )
    allow_backward: bool = Field(
        default=False,
        description="Allow targets behind the robot to generate backward motion",
    )


class DwaRouteProviderBg(Background[DwaRouteProviderBgConfig]):
    """
    DWA Route Provider Background.

    Initializes and starts DwaRouteProvider in the background.
    """

    def __init__(self, config: DwaRouteProviderBgConfig):
        super().__init__(config)

        self.dwa_route_provider = DwaRouteProvider(
            penalty=self.config.penalty,
            margin=self.config.margin,
            w_goal=self.config.w_goal,
            w_clear=self.config.w_clear,
            y_bias=self.config.y_bias,
            obstacle_cost=self.config.obstacle_cost,
            person_stop_dist=self.config.person_stop_dist,
            person_stop_y_width=self.config.person_stop_y_width,
            ahead_m=self.config.ahead_m,
            half_width_m=self.config.half_width_m,
            stride=self.config.stride,
            unknown_is_obstacle=self.config.unknown_is_obstacle,
            kv=self.config.kv,
            kyaw=self.config.kyaw,
            v_max=self.config.v_max,
            w_max=self.config.w_max,
            v_min=self.config.v_min,
            vx_fixed=self.config.vx_fixed,
            safety_slowdown=self.config.safety_slowdown,
            enable_turn_in_place=self.config.enable_turn_in_place,
            theta_turn_deg=self.config.theta_turn_deg,
            allow_backward=self.config.allow_backward,
        )
        self.dwa_route_provider.start()

        logging.info(
            "DwaRouteProvider initialized and started in background "
            f"(ahead_m={self.config.ahead_m}, half_width_m={self.config.half_width_m}, "
            f"vx_fixed={self.config.vx_fixed}, theta_turn_deg={self.config.theta_turn_deg})"
        )

    def run(self) -> None:
        evt = (
            self._orchestrator_stop_event
            if self._orchestrator_stop_event is not None
            else threading.Event()
        )
        if evt.is_set():
            self.dwa_route_provider.stop()
            logging.info("DwaRouteProvider stopped")
            return
        time.sleep(1.0)
