# gnss_provider.py

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pyubx2 import UBXReader, UBXMessage, SET, UBX_PROTOCOL


@dataclass(frozen = True)
class UbxPvtRecord:
    t_monotonic: float
    hour: Optional[int]
    minute: Optional[int]
    second: Optional[int]
    validTime: Optional[bool]
    fixType: Optional[int]
    diffSoln: Optional[int]
    carrSoln: Optional[int]
    numSV: Optional[int]
    lon: Optional[float]
    lat: Optional[float]
    hAcc_m: Optional[float]
    pDOP: Optional[float]


class GnssProvider:
    def __init__(self, ser, measRate_ms: int = 100, write_lock: Optional[threading.RLock] = None) -> None:
        self.ser = ser
        self.measRate_ms = measRate_ms
        self.write_lock = write_lock or threading.RLock()

        self._lock = threading.Lock()
        self._latest: Optional[UbxPvtRecord] = None

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="GnssReader")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        
        self._stop_evt.set()
        self._thread.join(timeout=2.0)

        if not self._thread.is_alive():
            self._thread = None

    def get_record(self) -> Optional[UbxPvtRecord]:
        with self._lock:
            return self._latest

    def get(self) -> Optional[Dict[str, Any]]:
        rec = self.get_record()
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
            "lon": rec.lon,
            "lat": rec.lat,
            "hAcc_m": rec.hAcc_m,
            "pDOP": rec.pDOP,
        }

    def _set_latest(self, rec: UbxPvtRecord) -> None:
        with self._lock:
            self._latest = rec

    def _cfg_interface(self) -> None:
        with self.write_lock:
            self.ser.write(
                UBXMessage(
                    "CFG", "CFG-RATE", SET,
                    measRate=self.measRate_ms,
                    navRate=1,
                    timeRef=0,
                ).serialize()
            )
            time.sleep(0.1)

            self.ser.write(
                UBXMessage(
                    "CFG", "CFG-MSG", SET,
                    msgClass=0x01, msgID=0x07, rateUSB=1
                ).serialize()
            )
            time.sleep(0.1)

    @staticmethod
    def _parse_navpvt(parsed: UBXMessage) -> UbxPvtRecord:
        hAcc_mm = getattr(parsed, "hAcc", None)
        hAcc_m = (hAcc_mm * 1e-3) if hAcc_mm is not None else None

        return UbxPvtRecord(
            t_monotonic=time.monotonic(),
            hour=getattr(parsed, "hour", None),
            minute=getattr(parsed, "min", None),
            second=getattr(parsed, "second", None),
            validTime=getattr(parsed, "validTime", None),
            fixType=getattr(parsed, "fixType", None),
            diffSoln=getattr(parsed, "diffSoln", None),
            carrSoln=getattr(parsed, "carrSoln", None),
            numSV=getattr(parsed, "numSV", None),
            lon=getattr(parsed, "lon", None),
            lat=getattr(parsed, "lat", None),
            hAcc_m=hAcc_m,
            pDOP=getattr(parsed, "pDOP", None),
        )

    def _run(self) -> None:
        self._cfg_interface()
        ubr = UBXReader(self.ser, protfilter=UBX_PROTOCOL)

        while not self._stop_evt.is_set():
            try:
                _, parsed = ubr.read()
            except Exception:
                self._stop_evt.wait(0.01)
                continue

            if isinstance(parsed, UBXMessage) and parsed.identity == "NAV-PVT":
                self._set_latest(self._parse_navpvt(parsed))
