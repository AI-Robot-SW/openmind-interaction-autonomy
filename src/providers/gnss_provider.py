# gnss_provider.py

import time
import logging
import threading

import serial

from dataclasses import dataclass
from typing import Optional

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
    def __init__(
        self,
        port: str = "/dev/rtk",
        baud: int = 115200,
        measRate_ms: int = 100
    ):
        self._port = port
        self._baud = baud
        self.measRate_ms = measRate_ms
        self.ser: Optional[serial.Serial] = None
        
        self._write_lock = threading.RLock()
        self._data: Optional[UbxPvtRecord] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("Gnss(RTK)Provider already running")
            return

        self.ser = serial.Serial(self._port, self._baud, timeout=1.0)

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="GnssReader")
        self._thread.start()
        
        # 첫 레코드 도착까지 대기
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("Gnss(RTK)Provider: timed out waiting for first record")

        logging.info("Gnss(RTK)Provider started")

    def stop(self) -> None:
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self.ser is not None:
            self.ser.close()
            self.ser = None

        logging.info("Gnss(RTK)Provider stopped")

    def _cfg_interface(self) -> None:
        with self._write_lock:
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

    @property
    def data(self) -> Optional[UbxPvtRecord]:
        """최신 UbxPvtRecord. 데이터 수신 전에는 None."""
        with self._lock:
            return self._data
        
    def _run(self) -> None:
        self._cfg_interface()
        ubr = UBXReader(self.ser, protfilter=UBX_PROTOCOL)

        while self.running:
            try:
                _, parsed = ubr.read()
            except Exception:
                time.sleep(0.01)
                continue

            if isinstance(parsed, UBXMessage) and parsed.identity == "NAV-PVT":
                with self._lock:
                    self._data = self._parse_navpvt(parsed)
