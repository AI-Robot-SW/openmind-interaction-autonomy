import logging
import threading
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider


class BEVOccupancyGridConfig(BackgroundConfig):
    res: Optional[float] = Field(
        default=0.05, description="Resolution of the grid in meters per pixel"
    )
    width: Optional[int] = Field(
        default=60, description="Camera X (lateral) grid cell count"
    )
    height: Optional[int] = Field(
        default=50, description="Camera Z (forward) grid cell count"
    )
    origin_z: Optional[float] = Field(
        default=0.0, description="Camera Z (forward) origin of the grid in meters"
    )
    origin_x: Optional[float] = Field(
        default=-1.5, description="Camera X (lateral) origin of the grid in meters"
    )
    dz: Optional[float] = Field(
        default=-0.34, description="Camera Z (forward) mount offset in meters"
    )
    dx: Optional[float] = Field(
        default=0.0, description="Camera X (lateral) mount offset in meters"
    )
    closing_kernel_size: Optional[int] = Field(
        default=1, description="Size of morphological closing kernel"
    )


class BEVOccupancyGridBg(Background[BEVOccupancyGridConfig]):
    """
    BEV Occupancy Grid Background.

    Initializes and starts the BEVOccupancyGridProvider in the background.
    """

    def __init__(self, config: BEVOccupancyGridConfig):
        super().__init__(config)

        self.bev_occupancy_grid_provider = BEVOccupancyGridProvider(
            res=self.config.res,
            width=self.config.width,
            height=self.config.height,
            origin_z=self.config.origin_z,
            origin_x=self.config.origin_x,
            dz=self.config.dz,
            dx=self.config.dx,
            closing_kernel_size=self.config.closing_kernel_size,
        )
        self.bev_occupancy_grid_provider.start()
        logging.info(
            f"BEV Occupancy Grid Provider initialized in background "
            f"(res: {self.config.res}, size: ({self.config.width},{self.config.height}), origin: ({self.config.origin_z},{self.config.origin_x}), "
            f"dz: {self.config.dz}, dx: {self.config.dx})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.bev_occupancy_grid_provider.stop()
            logging.info("BEV Occupancy Grid Provider stopped")
            return
        time.sleep(1.0)
