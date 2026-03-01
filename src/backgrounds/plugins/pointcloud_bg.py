import logging
import threading
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.pointcloud_provider import PointCloudProvider


class PointCloudConfig(BackgroundConfig):
    """
    Configuration for PointCloud Background.

    Parameters
    ----------
    range_max : Optional[float]
        Maximum range for point cloud filtering (default: 0.0, no limit).
    stride : Optional[int]
        Downsampling stride (default: 1, no downsampling).
    sync_slop : Optional[float]
        Time synchronization tolerance in seconds (default: 0.05).
    """

    range_max: Optional[float] = Field(
        default=0.0, description="Maximum range for point cloud filtering"
    )
    stride: Optional[int] = Field(
        default=1, description="Downsampling stride"
    )
    sync_slop: Optional[float] = Field(
        default=0.05, description="Time synchronization tolerance in seconds"
    )


class PointCloudBg(Background[PointCloudConfig]):
    """
    PointCloud Background.

    Initializes and starts the PointCloudProvider in the background.
    """

    def __init__(self, config: PointCloudConfig):
        """
        Initialize the PointCloud Background.

        Parameters
        ----------
        config : PointCloudConfig
            Configuration for the background task.
        """
        super().__init__(config)

        range_max = self.config.range_max or 0.0
        stride = self.config.stride or 1
        sync_slop = self.config.sync_slop or 0.05

        # Initialize Provider (singleton, so same instance shared)
        self.pointcloud_provider = PointCloudProvider(
            range_max=range_max,
            stride=stride,
            sync_slop=sync_slop,
        )

        # Start Provider
        self.pointcloud_provider.start()

        logging.info(
            f"PointCloud Provider initialized in background (range_max: {range_max}, stride: {stride}, sync_slop: {sync_slop})"
        )

    def run(self) -> None:
        evt = getattr(self, "_orchestrator_stop_event", None)
        evt = evt if evt is not None else threading.Event()
        try:
            while not evt.is_set():
                time.sleep(1.0)
        finally:
            self.pointcloud_provider.stop()
            logging.info("PointCloud Provider stopped")
