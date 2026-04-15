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
    @staticmethod
    def _make_empty_record() -> UwbPosRecord:
        return UwbPosRecord(
            t_monotonic=time.monotonic(),
            x_m=None,y_m=None, z_m=None,
            quality_factor=None
        )

    def __init__(
        self,
        port: str = "/dev/uwb",
        baud: int = 115200,
    ):
        self._port = port
        self._baud = baud
        self.ser: Optional[serial.Serial] = None

        self._write_lock = threading.RLock()
        self._data: UwbPosRecord = self._make_empty_record()
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None



    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("UwbProvider already running")
            return

        self.ser = serial.Serial(self._port, self._baud, timeout=0.2)

        with self._lock:
            self._data = self._make_empty_record()

        try:
            self._cfg_interface()
        except Exception:
            self.ser.close()
            self.ser = None
            raise

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="UwbReader")
        self._thread.start()

        logging.info("UwbProvider started")

    def stop(self) -> None:
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self.ser is not None:
            try:
                exited = self._exit_shell()
                if not exited:
                    logging.debug("UwbProvider: shell exit not confirmed")
            except Exception:
                logging.exception("UwbProvider: _exit_shell failed")
            finally:
                self.ser.close()
                self.ser = None

        logging.info("UwbProvider stopped")

    def _enter_shell(self, timeout: float = 3.0) -> None:
        if self.ser is None:
            raise RuntimeError("UwbProvider serial is not open")

        with self._write_lock:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            self.ser.write(b"\r")
            self.ser.flush()
            time.sleep(0.1)

            self.ser.write(b"\r")
            self.ser.flush()

        deadline = time.monotonic() + timeout
        buf = bytearray()

        while time.monotonic() < deadline:
            chunk = self._read_some()
            if chunk:
                buf.extend(chunk)
                if b"dwm>" in buf:
                    return
            else:
                time.sleep(0.01)

        raise RuntimeError("UwbProvider: failed to enter DWM shell")

    def _exit_shell(self, timeout: float = 1.0) -> bool:
        if self.ser is None:
            return False

        with self._write_lock:
            self.ser.reset_input_buffer()

            self.ser.write(b"lec\r")
            self.ser.flush()
            time.sleep(0.1)

            self.ser.write(b"quit\r")
            self.ser.flush()

        deadline = time.monotonic() + timeout
        buf = bytearray()

        while time.monotonic() < deadline:
            chunk = self._read_some()
            if chunk:
                buf.extend(chunk)
                if b"bye!" in buf:
                    return True
            else:
                time.sleep(0.01)

        return False

    def _cfg_interface(self) -> None:
        if self.ser is None:
            raise RuntimeError("UwbProvider serial is not open")

        self._enter_shell()

        with self._write_lock:
            self.ser.reset_input_buffer()
            self.ser.write(b"lec\r")
            self.ser.flush()
    
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
    def _parse_pos_line(line: bytes) -> Optional[UwbPosRecord]:
        idx = line.find(b"POS,")
        if idx < 0:
            return None

        parts = line[idx:].split(b",")
        if len(parts) < 5:
            return None

        try:
            x_m = float(parts[1])
            y_m = float(parts[2])
            z_m = float(parts[3])
            quality_factor = int(float(parts[4]))
        except Exception:
            return None

        return UwbPosRecord(
            t_monotonic=time.monotonic(),
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            quality_factor=quality_factor,
        )
    
    @property
    def data(self) -> UwbPosRecord:
        """최신 UwbPosRecord. x_m/y_m/quality_factor 가 None이면 포지션 미수신."""
        with self._lock:
            return self._data        
    def _run(self) -> None:
        buf = bytearray()

        while self.running:
            chunk = self._read_some()

            if not chunk:
                time.sleep(0.01)
                continue

            buf.extend(chunk)
            for line in self._extract_lines(buf):
                rec = self._parse_pos_line(line)
                if rec is None:
                    rec = self._make_empty_record()
                with self._lock:
                    self._data = rec