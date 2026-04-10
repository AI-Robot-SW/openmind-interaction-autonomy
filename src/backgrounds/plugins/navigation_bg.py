# navigation_bg.py

import time
import logging
import threading
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.navigation_provider import NavigationProvider


class NavigationBgConfig(BackgroundConfig):
    tick_dt: float = Field(
        default=0.05,
        description="Worker thread polling interval in seconds",
    )
    speed_step: float = Field(
        default=0.2,
        description="Speed increment/decrement step in m/s",
    )
    speed_min: float = Field(
        default=0.2,
        description="Minimum navigation speed in m/s",
    )
    speed_max: Optional[float] = Field(
        default=None,
        description="Maximum navigation speed in m/s (None uses DWA v_max)",
    )
    

class NavigationBg(Background[NavigationBgConfig]):
    """
    Navigation Background.

    Wraps NavigationProvider, which combines GnssRouteProvider and DwaRouteProvider.
    GnssRouteProvider and DwaRouteProvider must already be running via their own backgrounds
    (GnssRouteBg, DwaRouteProviderBg) before this background starts.
    """

    def __init__(self, config: NavigationBgConfig):
        super().__init__(config)

        self.navigation_provider = NavigationProvider(
            tick_dt=self.config.tick_dt,
            speed_step=self.config.speed_step,
            speed_min=self.config.speed_min,
            speed_max=self.config.speed_max
        )
        self.navigation_provider.start()

        logging.info(
            "NavigationProvider initialized and started in background "
            f"(tick_dt={self.config.tick_dt}, speed_step={self.config.speed_step}, "
            f"speed_min={self.config.speed_min}, speed_max={self.config.speed_max})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.navigation_provider.stop()
            logging.info("NavigationProvider stopped")
            return
        time.sleep(1.0)
