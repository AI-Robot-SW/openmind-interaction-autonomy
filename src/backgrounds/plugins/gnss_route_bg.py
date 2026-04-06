# gnss_route_bg.py

import time
import math
import logging
import threading

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.gnss_route_provider import GnssRouteProvider


class GnssRouteBgConfig(BackgroundConfig):
    max_vx: float = Field(
        default=0.8,
        description="Maximum forward velocity in m/s",
    )
    max_vyaw: float = Field(
        default=math.radians(45),
        description="Maximum yaw velocity in rad/s",
    )


class GnssRouteBg(Background[GnssRouteBgConfig]):
    """
    GNSS Route Provider Background.

    Initializes and starts GnssRouteProvider in the background.
    """

    def __init__(self, config: GnssRouteBgConfig):
        super().__init__(config)

        self.gnss_route_provider = GnssRouteProvider(
            max_vx=self.config.max_vx,
            max_vyaw=self.config.max_vyaw,
        )
        self.gnss_route_provider.start()

        logging.info(
            "GnssRouteProvider initialized and started in background "
            f"(max_vx={self.config.max_vx}, max_vyaw={self.config.max_vyaw})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.gnss_route_provider.stop()
            logging.info("GnssRouteProvider stopped")
            return
        time.sleep(1.0)
