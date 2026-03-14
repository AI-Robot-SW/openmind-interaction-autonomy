# realsense_camera_bg.py

import time
import logging
import threading

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.realsense_camera_provider import RealSenseCameraProvider


class RealSenseCameraBgConfig(BackgroundConfig):
    camera_index: int = Field(default=0)
    width: int = Field(default=640)
    height: int = Field(default=480)
    fps: int = Field(default=30)


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
        )
        self.realsense_camera_provider.start()
        logging.info(
            f"RealSenseCameraProvider initialized and started in background "
            f"(index={self.config.camera_index}, "
            f"{self.config.width}x{self.config.height}@{self.config.fps})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.realsense_camera_provider.stop()
            logging.info("RealSenseCameraProvider stopped")
            return
        time.sleep(1.0)

