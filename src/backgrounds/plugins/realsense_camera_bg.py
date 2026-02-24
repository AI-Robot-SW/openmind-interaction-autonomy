import time
import logging
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.realsense_camera_provider import RealSenseCameraProvider


class RealSenseCameraBgConfig(BackgroundConfig):
    camera_index: int = Field(default=0)
    width: int = Field(default=640)
    height: int = Field(default=480)
    fps: int = Field(default=30)
    align_depth_to_color: bool = Field(default=True)


class RealSenseCameraBg(Background[RealSenseCameraBgConfig]):
    """
    RealSense Camera Background.

    Initializes and starts the RealSenseCameraProvider in the background.
    """

    def __init__(self, config: RealSenseCameraBgConfig):
        super().__init__(config)

        self.realsense_camera_provider = RealSenseCameraProvider(
            camera_index=self.config.camera_index,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            align_depth_to_color=self.config.align_depth_to_color,
        )

    def run(self) -> None:
        logging.info(
            f"Starting RealSenseCameraProvider (index={self.config.camera_index}, "
            f"{self.config.width}x{self.config.height}@{self.config.fps}, "
            f"align={self.config.align_depth_to_color})"
        )

        self.realsense_camera_provider.start()

        try:
            # Background의 생명주기 유지를 위한 루프
            while True:
                time.sleep(1.0)
        finally:
            logging.info("Stopping RealSenseCameraProvider")
            self.realsense_camera_provider.stop()

