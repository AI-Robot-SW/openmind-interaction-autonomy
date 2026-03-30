# uwb_bg.py

import time
import logging
import threading

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.uwb_provider import UwbProvider


class UwbBgConfig(BackgroundConfig):
    port: str = Field(default="/dev/uwb", description="UWB serial port")
    baud: int = Field(default=115200, description="UWB baudrate")


class UwbBg(Background[UwbBgConfig]):
    """
    UWB Background.

    Initializes and starts the UwbProvider in the background.
    """

    def __init__(self, config: UwbBgConfig):
        super().__init__(config)

        self.uwb_provider = UwbProvider(
            port=self.config.port,
            baud=self.config.baud,
        )
        self.uwb_provider.start()

        logging.info(
            f"UwbProvider initialized in background "
            f"(port: {self.config.port}, baud: {self.config.baud})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.uwb_provider.stop()
            logging.info("UwbProvider stopped")
            return
        time.sleep(1.0)