import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider


class BEVOccupancyGridConfig(BackgroundConfig):
    """
    Configuration for BEV Occupancy Grid Background.

    Parameters
    ----------
    res : Optional[float]
        Resolution of the grid in meters per pixel (default: 0.05).
    width : Optional[int]
        Width of the grid in pixels (default: 50).
    height : Optional[int]
        Height of the grid in pixels (default: 60).
    origin_x : Optional[float]
        X origin of the grid in meters (default: 0.0).
    origin_y : Optional[float]
        Y origin of the grid in meters (default: -1.5).
    dx : Optional[float]
        X offset for coordinate transformation (default: -0.34).
    dy : Optional[float]
        Y offset for coordinate transformation (default: 0.0).
    closing_kernel_size : Optional[int]
        Size of morphological closing kernel (default: 1).
    """

    res: Optional[float] = Field(
        default=0.05, description="Resolution of the grid in meters per pixel"
    )
    width: Optional[int] = Field(
        default=50, description="Width of the grid in pixels"
    )
    height: Optional[int] = Field(
        default=60, description="Height of the grid in pixels"
    )
    origin_x: Optional[float] = Field(
        default=0.0, description="X origin of the grid in meters"
    )
    origin_y: Optional[float] = Field(
        default=-1.5, description="Y origin of the grid in meters"
    )
    dx: Optional[float] = Field(
        default=-0.34, description="X offset for coordinate transformation"
    )
    dy: Optional[float] = Field(
        default=0.0, description="Y offset for coordinate transformation"
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
        """
        Initialize the BEV Occupancy Grid Background.

        Parameters
        ----------
        config : BEVOccupancyGridConfig
            Configuration for the background task.
        """
        super().__init__(config)

        res = self.config.res or 0.05
        width = self.config.width or 50
        height = self.config.height or 60
        origin_x = self.config.origin_x or 0.0
        origin_y = self.config.origin_y or -1.5
        dx = self.config.dx or -0.34
        dy = self.config.dy or 0.0
        closing_kernel_size = self.config.closing_kernel_size or 1

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
        )

        # Start Provider
        self.bev_occupancy_grid_provider.start()

        logging.info(
            f"BEV Occupancy Grid Provider initialized in background "
            f"(res: {res}, size: ({width},{height}), origin: ({origin_x},{origin_y}), "
            f"dx: {dx}, dy: {dy})"
        )

    def run(self) -> None:
        """
        Background process loop.

        This Background only keeps the BEVOccupancyGridProvider running; BEV
        updates are handled inside the Provider. No additional work is done here.
        """
        time.sleep(60)
