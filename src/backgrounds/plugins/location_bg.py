# location_bg.py

from __future__ import annotations

import logging
import serial
import threading
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.rtk_provider import RtkProvider
from providers.uwb_provider import UwbProvider
from providers.location_provider import LocationProvider

logger = logging.getLogger(__name__)


class LocationBgConfig(BackgroundConfig):
    gnss_port: Optional[str] = Field(default=None, description="GNSS serial port")
    gnss_baud: int = Field(default=115200, description="GNSS baudrate")

    uwb_port: Optional[str] = Field(default=None, description="UWB serial port")
    uwb_baud: int = Field(default=115200, description="UWB baudrate")

    gnss_meas_rate_ms: int = Field(default=100, description="GNSS measRate (ms)")

    rtk_caster: Optional[str] = Field(default="rts2.ngii.go.kr", description="NTRIP caster host")
    rtk_port: int = Field(default=2101, description="NTRIP caster port")
    rtk_mountpoint: str = Field(default="VRS-RTCM32", description="NTRIP mountpoint")
    rtk_user: str = Field(default="", description="NTRIP username")
    rtk_password: str = Field(default="ngii", description="NTRIP password")


class LocationBg(Background[LocationBgConfig]):
    def __init__(self, config: LocationBgConfig):
        super().__init__(config)

        self.location_provider: Optional[LocationProvider] = None
        self._gnss_ser: Optional[serial.Serial] = None
        self._uwb_ser: Optional[serial.Serial] = None

        self._gnss_ser = serial.Serial(self.config.gnss_port, self.config.gnss_baud, timeout=1.0)
        self._uwb_ser = serial.Serial(self.config.uwb_port, self.config.uwb_baud, timeout=0.2)

        rtk = RtkProvider(
            ser=self._gnss_ser,
            measRate_ms=self.config.gnss_meas_rate_ms,
            caster=self.config.rtk_caster,
            port=self.config.rtk_port,
            mountpoint=self.config.rtk_mountpoint,
            user=self.config.rtk_user,
            password=self.config.rtk_password,
        )
        uwb = UwbProvider(ser=self._uwb_ser)

        self.location_provider = LocationProvider(gnss=rtk, uwb=uwb)
        self.location_provider.start()
        logger.info("LocationProvider initialized and started in background")

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            if self.location_provider is not None:
                self.location_provider.stop()
                logger.info("LocationProvider stopped")
            return
        time.sleep(1.0)