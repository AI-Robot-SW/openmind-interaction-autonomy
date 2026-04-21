# rtk_provider.py

import time
import base64
import socket
import logging
import threading

from typing import Optional, Tuple

from providers.gnss_provider import GnssProvider, UbxPvtRecord
from providers.singleton import singleton


def _nmea_checksum(payload: str) -> str:
    c = 0
    for ch in payload:
        c ^= ord(ch)
    return f"{c:02X}"


def _nmea_lat(lat_pvt: float) -> Tuple[str, str]:
    hemi = "N" if lat_pvt >= 0 else "S"
    lat = abs(lat_pvt)
    d = int(lat)
    m = (lat - d) * 60.0
    return f"{d:02d}{m:07.4f}", hemi


def _nmea_lon(lon_pvt: float) -> Tuple[str, str]:
    hemi = "E" if lon_pvt >= 0 else "W"
    lon = abs(lon_pvt)
    d = int(lon)
    m = (lon - d) * 60.0
    return f"{d:03d}{m:07.4f}", hemi


def _nmea_gga(lat_ubx: float, lon_ubx: float, hour: int, minute: int, second: int) -> bytes:
    hhmmss = f"{hour:02d}{minute:02d}{second:02d}.00"
    lat_nmea, ns = _nmea_lat(lat_ubx)
    lon_nmea, ew = _nmea_lon(lon_ubx)
    payload = f"GNGGA,{hhmmss},{lat_nmea},{ns},{lon_nmea},{ew},,,,,,,,,"
    cs = _nmea_checksum(payload)
    return f"${payload}*{cs}\r\n".encode("ascii")


@singleton
class RtkProvider(GnssProvider):
    def __init__(
        self,
        port: str = "/dev/rtk",
        baud: int = 115200,
        measRate_ms: int = 100,
        caster: str = "rts2.ngii.go.kr",
        ntrip_port: int = 2101,
        mountpoint: str = "VRS-RTCM32",
        user: str = "dori0126",
        password: str = "ngii",
    ) -> None:
        super().__init__(port=port, baud=baud, measRate_ms=measRate_ms)

        self.caster = caster
        self.ntrip_port = int(ntrip_port)
        self.mountpoint = mountpoint.lstrip("/")
        self.user = user
        self.password = password

        self._ntrip_thread: Optional[threading.Thread] = None
        self._last_gga_ts = 0.0

    def start(self) -> None:
        super().start()

        if self.caster and (self._ntrip_thread is None or not self._ntrip_thread.is_alive()):
            self._ntrip_thread = threading.Thread(target=self._ntrip_loop, daemon=True, name="NtripClient")
            self._ntrip_thread.start()

    def stop(self) -> None:
        self.running = False

        if self._ntrip_thread and self._ntrip_thread.is_alive():
            self._ntrip_thread.join(timeout=2.0)
        self._ntrip_thread = None

        super().stop()

    def _connect_ntrip_caster(self) -> socket.socket:
        auth = base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        req = (
            f"GET /{self.mountpoint} HTTP/1.0\r\n"
            "User-Agent: NTRIP PythonClient/1.0\r\n"
            f"Authorization: Basic {auth}\r\n"
            "\r\n"
        ).encode("ascii")

        sock = socket.create_connection((self.caster, self.ntrip_port), timeout=10)
        sock.sendall(req)
        return sock

    @staticmethod
    def _read_header(sock: socket.socket, max_bytes: int = 8192) -> tuple[bytes, bytes]:
        buf = bytearray()
        marker = b"\r\n\r\n"
        while marker not in buf:
            chunk = sock.recv(1024)
            if not chunk:
                raise ConnectionError("No NTRIP header (connection closed)")
            buf.extend(chunk)
            if len(buf) > max_bytes:
                raise ConnectionError("NTRIP header too large")
        raw = bytes(buf)
        head, rest = raw.split(marker, 1)
        return head, rest

    @staticmethod
    def _status_check(status_line: bytes) -> bool:
        return (b" 200" in status_line) or (b"ICY 200" in status_line)

    def _send_nmea_gga(self, sock: socket.socket) -> None:
        now = time.monotonic()
        if (now - self._last_gga_ts) < 1.0:
            return
        self._last_gga_ts = now

        rec: Optional[UbxPvtRecord] = self.data
        if rec is None:
            return

        if rec.lat is None or rec.lon is None:
            return
        if rec.hour is None or rec.minute is None or rec.second is None:
            return
        
        sock.sendall(_nmea_gga(rec.lat, rec.lon, rec.hour, rec.minute, rec.second))

    def _ntrip_loop(self) -> None:
        log = logging.getLogger(__name__)

        while self.running:
            sock: Optional[socket.socket] = None
            try:
                sock = self._connect_ntrip_caster()
                header, rest = self._read_header(sock)

                status_line = header.split(b"\r\n", 1)[0]
                if not self._status_check(status_line):
                    raise ConnectionError(f"NTRIP bad status: {status_line!r}")

                sock.settimeout(1.0)
                self._last_gga_ts = 0.0

                if rest:
                    with self._write_lock:
                        self.ser.write(rest)

                while self.running:
                    self._send_nmea_gga(sock)

                    try:
                        chunk = sock.recv(1024)
                    except socket.timeout:
                        continue

                    if chunk == b"":
                        raise ConnectionError("NTRIP socket closed")

                    with self._write_lock:
                        self.ser.write(chunk)

            except Exception as e:
                log.warning(f"[NTRIP] {e} (retry in 5s)")
                deadline = time.monotonic() + 5.0
                while self.running and time.monotonic() < deadline:
                    time.sleep(0.1)

            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass