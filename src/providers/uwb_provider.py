# uwb_provider.py

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class UwbPosRecord:
    t_monotonic: float
    x_m: Optional[float]
    y_m: Optional[float]
    z_m: Optional[float]
    quality_factor: Optional[int]


class UwbProvider:
    def __init__(self, ser, *, write_lock: Optional[threading.RLock] = None) -> None:
        self.ser = ser
        self.write_lock = write_lock or threading.RLock()

        self._lock = threading.Lock()
        self._latest: Optional[UwbPosRecord] = None

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="UwbReader")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        
        self._stop_evt.set()
        with self.write_lock:
            self.ser.write(b"\r")
            self.ser.flush()
        self._thread.join(timeout=2.0)

        if not self._thread.is_alive():
            self._thread = None

    def get_record(self) -> Optional[UwbPosRecord]:
        with self._lock:
            return self._latest

    def get(self) -> Optional[Dict[str, Any]]:
        rec = self.get_record()
        if rec is None:
            return None
        return {
            "t_monotonic": rec.t_monotonic,
            "x_m": rec.x_m,
            "y_m": rec.y_m,
            "z_m": rec.z_m,
            "quality_factor": rec.quality_factor,
        }

    def _set_latest(self, rec: UwbPosRecord) -> None:
        with self._lock:
            self._latest = rec

    def _cfg_interface(self) -> None:
        with self.write_lock:
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

    def _run(self) -> None:
        self._cfg_interface()

        buf = bytearray()

        while not self._stop_evt.is_set():
            chunk = self._read_some()

            if not chunk:
                self._stop_evt.wait(0.01)
                continue
            
            buf.extend(chunk)
            for line in self._extract_lines(buf):
                rec = self._parse_lep(line)
                if rec is not None:
                    self._set_latest(rec)
