# uwb_provider.py

import time
import logging
import threading

import serial

from dataclasses import dataclass
from typing import Optional

from .singleton import singleton


@dataclass(frozen=True)
class UwbPosRecord:
    t_monotonic: float
    x_m: Optional[float]
    y_m: Optional[float]
    z_m: Optional[float]
    quality_factor: Optional[int]


@singleton
class UwbProvider:
    def __init__(
        self,
        port: str = "/dev/uwb",
        baud: int = 115200,
    ):
        self._port = port
        self._baud = baud
        self.ser: Optional[serial.Serial] = None

        self._write_lock = threading.RLock()
        self._data: Optional[UwbPosRecord] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("UwbProvider already running")
            return

        self.ser = serial.Serial(self._port, self._baud, timeout=0.2)

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="UwbReader")
        self._thread.start()

        # 첫 레코드 도착까지 대기 — _cfg_interface가 ~1.2초 소요
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("UwbProvider: timed out waiting for first record")

        logging.info("UwbProvider started")

    def stop(self) -> None:
        self.running = False

        if self.ser is not None:
            with self._write_lock:
                self.ser.write(b"\r")
                self.ser.flush()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self.ser is not None:
            self.ser.close()
            self.ser = None

        logging.info("UwbProvider stopped")

    def _cfg_interface(self) -> None:
        with self._write_lock:
            self.ser.write(b"\r")
            time.sleep(0.1)
            self.ser.write(b"\r")
            time.sleep(1.0)
            self.ser.write(b"lep\r")
            time.sleep(0.1)

    def _read_some(self) -> bytes:
        n = int(getattr(self.ser, "in_waiting", 0) or 0)
        if n > 0:
            return self.ser.read(n)
        return b""

    @staticmethod
    def _extract_lines(buf: bytearray) -> list[bytes]:
        out: list[bytes] = []
        start = 0
        while True:
            idx = buf.find(b"\r\n", start)
            if idx < 0:
                break
            line = bytes(buf[start:idx])
            if line:
                out.append(line)
            start = idx + 2
        if start:
            del buf[:start]
        return out

    @staticmethod
    def _parse_lep(line: bytes) -> Optional[UwbPosRecord]:
        idx = line.find(b"POS,")
        if idx < 0:
            return None

        parts = line[idx:].split(b",")
        if len(parts) < 5:
            return None

        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            qf = int(float(parts[4]))
        except Exception:
            return None

        return UwbPosRecord(
            t_monotonic=time.monotonic(),
            x_m=x,
            y_m=y,
            z_m=z,
            quality_factor=qf,
        )

    @property
    def data(self) -> Optional[UwbPosRecord]:
        """최신 UwbPosRecord. 데이터 수신 전에는 None."""
        with self._lock:
            return self._data
        
    def _run(self) -> None:
        self._cfg_interface()

        buf = bytearray()

        while self.running:
            chunk = self._read_some()

            if not chunk:
                time.sleep(0.01)
                continue
            
            buf.extend(chunk)
            for line in self._extract_lines(buf):
                rec = self._parse_lep(line)
                with self._lock:
                    self._data = rec