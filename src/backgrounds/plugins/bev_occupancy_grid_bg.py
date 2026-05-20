# bev_occupancy_grid_bg.py

import time
import logging
import threading
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider


class BEVOccupancyGridConfig(BackgroundConfig):
    res: Optional[float] = Field(
        default=0.05, description="Resolution of the grid in meters per pixel"
    )
    width: Optional[int] = Field(
        default=140, description="Width of the grid in pixels"
    )
    height: Optional[int] = Field(
        default=120, description="Height of the grid in pixels"
    )
    origin_x: Optional[float] = Field(
        default=-0.5, description="X origin of the grid in meters"
    )
    origin_y: Optional[float] = Field(
        default=-3.0, description="Y origin of the grid in meters"
    )
    dx: Optional[float] = Field(
        default=-0.34, description="X offset for coordinate transformation"
    )
    dy: Optional[float] = Field(
        default=0.0, description="Y offset for coordinate transformation"
    )
    closing_kernel_size: Optional[int] = Field(
        default=3, description="Size of morphological closing kernel"
    )
    camera_height_m: float = Field(
        default=0.413, description="Camera optical center height above ground (meters)"
    )
    ground_projection_stride: int = Field(
        default=4, description="Pixel stride for driveable ground-plane projection"
    )


class BEVOccupancyGridBg(Background[BEVOccupancyGridConfig]):
    """
    BEV Occupancy Grid Background.

    Initializes and starts the BEVOccupancyGridProvider in the background.
    """

    def __init__(self, config: BEVOccupancyGridConfig):
        super().__init__(config)

        res = self.config.res or 0.05
        width = self.config.width or 140
        height = self.config.height or 120
        origin_x = self.config.origin_x if self.config.origin_x is not None else -0.5
        origin_y = self.config.origin_y if self.config.origin_y is not None else -3.0
        dx = self.config.dx if self.config.dx is not None else -0.34
        dy = self.config.dy if self.config.dy is not None else 0.0
        closing_kernel_size = self.config.closing_kernel_size or 3

        # Initialize Provider (singleton, so same instance shared)
        self.bev_occupancy_grid_provider = BEVOccupancyGridProvider(
            res=res,
            width=width,
            height=height,
            origin_x=origin_x,
            origin_y=origin_y,
            dx=dx,
            dy=dy,
            closing_kernel_size=closing_kernel_size,
            camera_height_m=self.config.camera_height_m,
            ground_projection_stride=self.config.ground_projection_stride,
        )

        # Start Provider
        self.bev_occupancy_grid_provider.start()

        logging.info(
            f"BEV Occupancy Grid Provider initialized in background "
            f"(res: {res}, size: ({width},{height}), origin: ({origin_x},{origin_y}), "
            f"dx: {dx}, dy: {dy}, camera_height_m: {self.config.camera_height_m})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.bev_occupancy_grid_provider.stop()
            logging.info("BEV Occupancy Grid Provider stopped")
            return
        time.sleep(1.0)
