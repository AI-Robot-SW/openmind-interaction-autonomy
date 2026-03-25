# rtk_bg.py

import time
import logging
import threading

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.rtk_provider import RtkProvider


class RtkBgConfig(BackgroundConfig):
    port: str = Field(default="/dev/rtk", description="RTK serial port")
    baud: int = Field(default=115200, description="RTK baudrate")
    meas_rate_ms: int = Field(default=100, description="GNSS measRate (ms)")

    caster: str = Field(default="rts2.ngii.go.kr", description="NTRIP caster host")
    ntrip_port: int = Field(default=2101, description="NTRIP caster port")
    mountpoint: str = Field(default="VRS-RTCM32", description="NTRIP mountpoint")
    user: str = Field(default="", description="NTRIP username")
    password: str = Field(default="", description="NTRIP password")


class RtkBg(Background[RtkBgConfig]):
    """
    RTK Background.

    Initializes and starts the RtkProvider in the background.
    """

    def __init__(self, config: RtkBgConfig):
        super().__init__(config)

        self.rtk_provider = RtkProvider(
            port=self.config.port,
            baud=self.config.baud,
            measRate_ms=self.config.meas_rate_ms,
            caster=self.config.caster,
            ntrip_port=self.config.ntrip_port,
            mountpoint=self.config.mountpoint,
            user=self.config.user,
            password=self.config.password,
        )
        self.rtk_provider.start()

        logging.info(
            f"RtkProvider initialized in background "
            f"(port: {self.config.port}, baud: {self.config.baud}, "
            f"caster: {self.config.caster}/{self.config.mountpoint})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.rtk_provider.stop()
            logging.info("RtkProvider stopped")
            return
        time.sleep(1.0)
