# location_provider.py

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .gnss_provider import GnssProvider, UbxPvtRecord
from .uwb_provider import UwbProvider, UwbPosRecord
from .singleton import singleton

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocationRecord:
    t_monotonic: float
    gnss: Optional[UbxPvtRecord]
    uwb: Optional[UwbPosRecord]


@singleton
class LocationProvider:
    def __init__(
        self,
        *,
        gnss: GnssProvider,
        uwb: UwbProvider,
    ) -> None:
        self._gnss = gnss
        self._uwb = uwb

        self._lock = threading.Lock()
        now = time.monotonic()
        self._latest = LocationRecord(
            t_monotonic=now,
            gnss=None,
            uwb=None
        )

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._gnss.start()
        self._uwb.start()

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="LocationReader")
        self._thread.start()

        logger.info("LocationProvider started")

    def stop(self) -> None:
        if self._thread is None:
            return

        self._stop_evt.set()
        self._uwb.stop()
        self._gnss.stop()
        self._thread.join(timeout=2.0)

        if not self._thread.is_alive():
            self._thread = None
        else:
            logger.warning("LocationProvider worker thread did not stop within timeout")

        logger.info("LocationProvider stopped")

    def get_record(self) -> LocationRecord:
        with self._lock:
            return self._latest

    @staticmethod
    def _gnss_dict(rec: Optional[UbxPvtRecord]) -> Optional[Dict[str, Any]]:
        if rec is None:
            return None
        return {
            "t_monotonic": rec.t_monotonic,
            "hour": rec.hour,
            "minute": rec.minute,
            "second": rec.second,
            "validTime": rec.validTime,
            "fixType": rec.fixType,
            "diffSoln": rec.diffSoln,
            "carrSoln": rec.carrSoln,
            "numSV": rec.numSV,
            "lat": rec.lat,
            "lon": rec.lon,
            "hAcc_m": rec.hAcc_m,
            "pDOP": rec.pDOP,
        }

    @staticmethod
    def _uwb_dict(rec: Optional[UwbPosRecord]) -> Optional[Dict[str, Any]]:
        if rec is None:
            return None
        return {
            "t_monotonic": rec.t_monotonic,
            "x_m": rec.x_m,
            "y_m": rec.y_m,
            "z_m": rec.z_m,
            "quality_factor": rec.quality_factor,
        }

    def get(self) -> Optional[Dict[str, Any]]:
        rec = self.get_record()
        if rec.gnss is None and rec.uwb is None:
            return None

        return {
            "t_monotonic": rec.t_monotonic,
            "gnss": self._gnss_dict(rec.gnss),
            "uwb": self._uwb_dict(rec.uwb),
        }

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                rec = LocationRecord(
                    t_monotonic=time.monotonic(),
                    gnss = self._gnss.get_record(),
                    uwb = self._uwb.get_record()
                )
                with self._lock:
                    self._latest = rec

            except Exception:
                logger.exception("Error in LocationProvider worker loop")
            
            self._stop_evt.wait(0.1)