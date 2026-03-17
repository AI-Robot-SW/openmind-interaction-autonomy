# distmap_bg.py

import time
import logging
import threading
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.distmap_provider import DistMapProvider


class DistMapConfig(BackgroundConfig):
    max_dist: Optional[float] = Field(
        default=5.0,
        description="Maximum obstacle distance represented in the distance map (meters)",
    )


class DistMapBg(Background[DistMapConfig]):
    """
    Distance map background.

    Initializes and starts the DistMapProvider in the background.
    """

    def __init__(self, config: DistMapConfig):
        super().__init__(config)

        self.distmap_provider = DistMapProvider(
            max_dist=self.config.max_dist
        )
        self.distmap_provider.start()

        logging.info(
            f"DistMap Provider initialized in background (max_dist: {self.config.max_dist})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.distmap_provider.stop()
            logging.info("DistMap Provider stopped")
            return
        time.sleep(1.0)
