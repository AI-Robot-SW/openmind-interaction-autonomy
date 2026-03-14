# pointcloud_bg.py

import time
import logging
import threading
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.pointcloud_provider import PointCloudProvider


class PointCloudConfig(BackgroundConfig):
    range_max: Optional[float] = Field(
        default=None, description="Maximum range for point cloud filtering (None = no filter)"
    )
    stride: int = Field(
        default=1, description="Downsampling stride"
    )


class PointCloudBg(Background[PointCloudConfig]):
    """
    PointCloud Background.

    Initializes and starts the PointCloudProvider in the background.
    """

    def __init__(self, config: PointCloudConfig):
        super().__init__(config)

        self.pointcloud_provider = PointCloudProvider(
            range_max=self.config.range_max,
            stride=int(self.config.stride),
        )
        self.pointcloud_provider.start()

        logging.info(
            f"PointCloud Provider initialized in background (range_max: {self.config.range_max}, stride: {int(self.config.stride)})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.pointcloud_provider.stop()
            logging.info("PointCloud Provider stopped")
            return
        time.sleep(1.0)
