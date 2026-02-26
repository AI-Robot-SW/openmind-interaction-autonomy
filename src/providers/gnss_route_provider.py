# gnss_route_provider.py

from __future__ import annotations

import csv
import logging
import math
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from .location_provider import LocationProvider
from .unitree_go2_provider import UnitreeGo2Provider

logger = logging.getLogger(__name__)


# ===================================================================================
# helpers (주 목적은 main thread._run 내부 로직을 단순화, 기하학적 수식 또는 제어 수식)
# ===================================================================================
EARTH_R = 6_371_000.0  # meter, 지구 반지름

def _wrap_deg(a: float) -> float:
    """Wrap angle (degrees) to [-180, 180]"""
    return (a + 180.0) % 360.0 - 180.0

def _haversine_xy(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> Tuple[float, float]:
    """
    (lat1, lon1)과 (lat2, lon2) 사이의 거리를 (west, north) 기준으로 반환, 단위는 meter
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat_avg = math.radians((lat1 + lat2) * 0.5)
    north = EARTH_R * dlat
    west = -EARTH_R * math.cos(lat_avg) * dlon
    return west, north

def _init_vyaw_control(yaw_hold_deg: float, yaw_current_deg: float):
    """
    처음 로봇의 yaw 방향으로 직진할 수 있도록 PD제어를 적용하여 vyaw를 제어
    """
    kp = 0.6
    vyaw_limit = 0.6    # rad/s

    yaw_error_deg = _wrap_deg(yaw_hold_deg - yaw_current_deg)
    yaw_error_rad = math.radians(yaw_error_deg)
    vyaw = kp * yaw_error_rad

    return max(-vyaw_limit, min(vyaw_limit, vyaw))

class _LinearPath:
    """
    GPS 좌표 목록(waypoints)을 순서대로 추종하는 경로 추적기.

    현재 위치가 목표 waypoint의 reach_tol(m) 이내에 들어오면
    자동으로 다음 waypoint로 전진한다.
    마지막 waypoint에 도달하면 update()가 None을 반환한다.
    """

    def __init__(self, coords: List[Tuple[float, float]], reach_tol: float = 5.0):
        self._coords = coords
        self._reach_tol = reach_tol
        self._idx = 0

    @classmethod
    def from_file(cls, path_name: str, reach_tol: float = 5.0, min_quality: int = 5) -> "_LinearPath":
        """
        txt/csv 파일에서 waypoint를 로드해 인스턴스를 생성한다.

        파일 형식: time,lat,lon,hdop,quality (헤더 포함)
        """
        p = Path(path_name)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / path_name

        coords: List[Tuple[float, float]] = []
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["quality"]) >= min_quality:
                    coords.append((float(row["lat"]), float(row["lon"])))

        if not coords:
            logger.warning("_LinearPath.from_file: quality >= %d 인 waypoint가 없습니다 (%s)", min_quality, p)

        logger.info("_LinearPath.from_file: %d waypoints loaded from %s", len(coords), p)
        return cls(coords=coords, reach_tol=reach_tol)

    def update(self, lat: float, lon: float) -> Optional[Tuple[float, float]]:
        """현재 위치(lat, lon)를 기준으로 waypoint를 갱신하고 다음 목표를 반환한다. 경로 완료 시 None."""
        while self._idx < len(self._coords):
            g = self._coords[self._idx]
            west, north = _haversine_xy(lat, lon, g[0], g[1])
            dist = math.hypot(west, north)
            if dist < self._reach_tol:
                if self._idx == len(self._coords) - 1:
                    return None
                self._idx += 1
                continue
            return g
        return None


# ===================================================================================
# GnssRouteProvider 메인 로직
# ===================================================================================
@dataclass(frozen=True)
class GnssRouteRecord:
    t_monotonic: float
    heading_calibrated: bool = False
    reached_goal: bool = False
    dx: float = 0.0
    dy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


class GnssRouteProvider:
    """
    GNSS 경로 추종 Provider.

    waypoints : List[Tuple[float, float]]
        GPS 경유지 목록 [(lat, lon), ...].
    reach_tol_m : float
        경유지 도달 판정 반경 (m).
    """

    def __init__(
        self,
        *,
        waypoints: List[Tuple[float, float]],
        reach_tol_m: float = 5.0,
        max_vx: float = 0.8,
        max_vyaw: float = math.radians(45),
    ) -> None:
        self.waypoints = waypoints
        self.reach_tol_m = reach_tol_m
        self.max_vx = max_vx
        self.max_vyaw = max_vyaw

        self._lock = threading.Lock()
        self._latest: Optional[GnssRouteRecord] = None

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ---- runtime state (accessed only from control thread) ----
        self._heading_calibrated: bool = False
        self._reached_goal: bool = False

        self._yaw_offset_deg: float = 0.0
        self.yaw_update_move_thresh_m: float = 16.0
        self.yaw_update_alpha: float = 0.1
        self.heading_margin_rad: float = math.radians(45)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="GnssRouteCtrl")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return

        # 안전 정지, stop되면 모든 제어값을 0으로 초기화
        self._set_latest(GnssRouteRecord(
            t_monotonic = time.monotonic(),
            heading_calibrated = self._heading_calibrated,
            reached_goal = self._reached_goal,
            dx = 0.0, dy = 0.0,
            vx = 0.0, vy = 0.0, vyaw = 0.0
        ))

        self._stop_evt.set()
        self._thread.join(timeout=5.0)

        if not self._thread.is_alive():
            self._thread = None

    def get_record(self) -> Optional[GnssRouteRecord]:
        with self._lock:
            return self._latest

    def get(self) -> Optional[Dict[str, Any]]:
        rec = self.get_record()
        if rec is None:
            return None
        return {
            "t_monotonic": rec.t_monotonic,
            "heading_calibrated": rec.heading_calibrated,
            "reached_goal": rec.reached_goal,
            "dx": rec.dx,
            "dy": rec.dy,
            "vx": rec.vx,
            "vy": rec.vy,
            "vyaw": rec.vyaw,
        }

    def _set_latest(self, rec: GnssRouteRecord) -> None:
        with self._lock:
            self._latest = rec

    def _wait_for_rtk(self, location_provider: LocationProvider) -> bool:
        """RTK 신호가 준비될 때까지 대기. 활성화 시 True, 비활성화 시 False."""
        logger.info("GnssRouteProvider: waiting for reliable RTK state …")
        while not self._stop_evt.is_set():
            gnss = location_provider.get_record().gnss
            if gnss is not None and (gnss.carrSoln or 0) >= 1:
                logger.info("GnssRouteProvider: GNSS ready")
                return True
            self._stop_evt.wait(0.1)
        return False

    def _calibrate_heading(self, location_provider: LocationProvider, unitree_go2_provider: UnitreeGo2Provider) -> bool:
        """초기 직진으로 yaw_offset_deg 캘리브레이션. 성공 시 True, 중단 시 False."""
        logger.info("GnssRouteProvider: initial straight moving for yaw calibration")
        total_west, total_north = 0.0, 0.0
        odom_init = unitree_go2_provider.get_odometry()
        yaw_init_deg = math.degrees(odom_init.yaw)

        while not self._stop_evt.is_set():
            t_monotonic = time.monotonic()
            odom_prev = unitree_go2_provider.get_odometry()
            gnss_prev = location_provider.get_record().gnss
            lat_prev, lon_prev = gnss_prev.lat, gnss_prev.lon

            vyaw = _init_vyaw_control(yaw_init_deg, math.degrees(odom_prev.yaw))
            self._set_latest(GnssRouteRecord(t_monotonic=t_monotonic, vx=0.5, vy=0.0, vyaw=vyaw))  # dx/dy는 의미 없음(초기값 0.0). 소비자는 heading_calibrated 확인
            if self._stop_evt.wait(0.1):
                return False

            odom = unitree_go2_provider.get_odometry()
            gnss = location_provider.get_record().gnss
            dw, dn = _haversine_xy(lat_prev, lon_prev, gnss.lat, gnss.lon)
            total_west += dw
            total_north += dn

            traveled = math.hypot(odom.x - odom_init.x, odom.y - odom_init.y)  # meter
            if traveled >= 3.0 and math.hypot(total_west, total_north) > 2.5:
                gnss_heading = math.degrees(math.atan2(total_west, total_north))
                self._yaw_offset_deg = _wrap_deg(gnss_heading - math.degrees(odom.yaw))
                self._heading_calibrated = True
                logger.info("GnssRouteProvider: heading calibrated, yaw_offset_deg=%.2f°", self._yaw_offset_deg)
                return True

        return False

    def _follow_path(self, location_provider: LocationProvider, unitree_go2_provider: UnitreeGo2Provider, lat_snap: float, lon_snap: float) -> None:
        """
        메인 루프. waypoint 경로를 순서대로 따라가면서 현재 상태를 GnssRouteRecord에 저장.

        path는 [(lat_0, lon_0), (lat_1, lon_1), ...] 형태의 GPS 좌표 목록이며,
        현재 위치와 현재 목표 waypoint 사이의 거리가 reach_tol 이내가 되면
        다음 waypoint로 목표를 갱신한다. 마지막 waypoint에 도달하면 루프를 종료.
        """
        path = _LinearPath(self.waypoints, reach_tol=self.reach_tol_m)
        logger.info("GnssRouteProvider: mission active, following %d waypoints", len(self.waypoints))

        while not self._stop_evt.is_set():
            gnss = location_provider.get_record().gnss
            lat_cur, lon_cur = gnss.lat, gnss.lon

            odom = unitree_go2_provider.get_odometry()
            global_heading = math.degrees(odom.yaw) + self._yaw_offset_deg

            goal = path.update(lat_cur, lon_cur)
            if goal is None:
                self._reached_goal = True
                logger.info("GnssRouteProvider: path finished!")
                self._set_latest(GnssRouteRecord(
                    t_monotonic=time.monotonic(),
                    heading_calibrated=True,
                    reached_goal=True,
                ))
                return

            # odometry yaw에 대한 drift 보정, gnss 노이즈를 감안하여 충분히 먼 거리를 이동했을 때만 적용.
            snap_w, snap_n = _haversine_xy(lat_snap, lon_snap, lat_cur, lon_cur)
            if math.hypot(snap_w, snap_n) > self.yaw_update_move_thresh_m:
                gnss_heading = math.degrees(math.atan2(snap_w, snap_n))
                yaw_offset_new_deg = _wrap_deg(gnss_heading - math.degrees(odom.yaw))
                delta = _wrap_deg(yaw_offset_new_deg - self._yaw_offset_deg)
                self._yaw_offset_deg = _wrap_deg(self._yaw_offset_deg + self.yaw_update_alpha * delta)
                global_heading = math.degrees(odom.yaw) + self._yaw_offset_deg
                lat_snap, lon_snap = lat_cur, lon_cur

            dW, dN = _haversine_xy(lat_cur, lon_cur, goal[0], goal[1])  # goal[0] = lat_goal, goal[1] = lon_goal
            dth = math.radians(global_heading)
            dx = math.cos(dth) * dN + math.sin(dth) * dW
            dy = -math.sin(dth) * dN + math.cos(dth) * dW
            heading_err = math.atan2(dy, dx)  # 로봇 바디 좌표계(+x 전방) 기준으로 목표 방향이 좌/우로 얼마나 벗어났는지(방향 오차, rad)

            vyaw = max(-self.max_vyaw, min(self.max_vyaw, heading_err))
            vy = 0.0  # go2는 측면 이동 불가
            vx = 0.0 if abs(heading_err) > self.heading_margin_rad else self.max_vx * (1.0 - abs(vyaw) / self.max_vyaw)

            self._set_latest(GnssRouteRecord(
                t_monotonic=time.monotonic(),
                heading_calibrated=self._heading_calibrated,
                reached_goal=self._reached_goal,
                dx=dx,
                dy=dy,
                vx=vx,
                vy=vy,
                vyaw=vyaw,
            ))
            self._stop_evt.wait(0.1)

    def _run(self) -> None:
        self._heading_calibrated = False
        self._reached_goal = False
        self._yaw_offset_deg = 0.0

        if LocationProvider._singleton_class._singleton_instance is None:
            logger.error("GnssRouteProvider: LocationProvider가 초기화되지 않았습니다.")
            return
        if UnitreeGo2Provider._singleton_class._singleton_instance is None:
            logger.error("GnssRouteProvider: UnitreeGo2Provider가 초기화되지 않았습니다.")
            return

        location_provider = LocationProvider()
        unitree_go2_provider = UnitreeGo2Provider()

        if not self._wait_for_rtk(location_provider):
            return

        if not self._calibrate_heading(location_provider, unitree_go2_provider):
            return

        gnss = location_provider.get_record().gnss
        self._follow_path(location_provider, unitree_go2_provider, gnss.lat, gnss.lon)
