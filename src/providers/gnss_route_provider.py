# gnss_route_provider.py

import csv
import time
import math
import logging
import threading

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .singleton import singleton

from .rtk_provider import RtkProvider
from .unitree_go2_provider import UnitreeGo2Provider
from .singleton import singleton


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
    north = EARTH_R * dlat
    west = -EARTH_R * math.cos(math.radians(lat1)) * dlon
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

class WaypointTracker:
    """
    GPS 좌표 목록(waypoints)을 순서대로 추종하는 경로 추적기.

    현재 위치가 목표 waypoint의 reach_tol(m) 이내에 들어오면
    자동으로 다음 waypoint로 전진한다.
    마지막 waypoint에 도달하면 update()가 None을 반환한다.
    """

    def __init__(
        self,
        coords: List[Tuple[float, float]],
        reach_tol: float = 5.0,
        start_lat: Optional[float] = None,
        start_lon: Optional[float] = None,
    ):
        self._coords = coords
        self._reach_tol = reach_tol
        if start_lat is not None and start_lon is not None and coords:
            self._idx = min(
                range(len(coords)),
                key=lambda i: math.hypot(*_haversine_xy(start_lat, start_lon, coords[i][0], coords[i][1])),
            )
        else:
            self._idx = 0

    @classmethod
    def from_file(cls, path_name: str, reach_tol: float = 5.0) -> "WaypointTracker":
        """
        txt/csv 파일에서 waypoint를 로드해 인스턴스를 생성한다.

        파일 형식: time,lat,lon,... (헤더 포함, lat/lon 컬럼 사용)
        """
        p = Path(path_name)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / path_name

        coords: List[Tuple[float, float]] = []
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                coords.append((float(row["lat"]), float(row["lon"])))

        if not coords:
            logging.warning("_WaypointTracker.from_file: waypoint가 없습니다 (%s)", p)

        logging.info("_WaypointTracker.from_file: %d waypoints loaded from %s", len(coords), p)
        return cls(coords=coords, reach_tol=reach_tol)

    @property
    def idx(self) -> int:
        return self._idx

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
    """
    t_monotonic        : GNSS 경로 추종 결과 생성 시각 (monotonic time)
    heading_calibrated : 초기 heading calibration 완료 여부
    reached_goal       : waypoint 경로 최종 도달 여부
    dx                 : 현재 목표 waypoint의 x 오프셋 (robot/body frame, meters)
    dy                 : 현재 목표 waypoint의 y 오프셋 (robot/body frame, meters)
    vx                 : GNSS 경로 추종으로 계산된 전진 속도 (m/s)
    vy                 : GNSS 경로 추종으로 계산된 측면 속도 (m/s)
    vyaw               : GNSS 경로 추종으로 계산된 yaw 속도 (rad/s)
    """
    t_monotonic: float
    heading_calibrated: bool = False
    reached_goal: bool = False
    dx: float = 0.0
    dy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


@singleton
class GnssRouteProvider:
    """
    background thread에서 GNSS/odometry 데이터를 읽어 waypoint 경로를 추종하고
    최신 결과를 GnssRouteRecord로 계산해 data 프로퍼티로 노출.
    heading calibration과 경로 추종 제어를 내부에서 수행
    """

    def __init__(
        self,
        waypoints: List[Tuple[float, float]],
        reach_tol_m: float = 5.0,
        max_vx: float = 0.8,
        max_vyaw: float = math.radians(45),
    ) -> None:
        self.waypoints = waypoints
        self.reach_tol_m = reach_tol_m
        self.max_vx = max_vx
        self.max_vyaw = max_vyaw

        self.rtk_provider = RtkProvider()
        self.unitree_go2_provider = UnitreeGo2Provider()

        self._data: Optional[GnssRouteRecord] = None
        self._lock = threading.Lock()

        self.running: bool = False
        self._thread: Optional[threading.Thread] = None

        self._tracker: Optional[WaypointTracker] = None

        # ---- runtime state (accessed only from control thread) ----
        self._heading_calibrated: bool = False
        self._reached_goal: bool = False

        self._yaw_offset_deg: float = 0.0
        self.yaw_update_move_thresh_m: float = 16.0
        self.yaw_update_alpha: float = 0.1
        self.heading_margin_rad: float = math.radians(45)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("GnssRouteProvider already running")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="GnssRouteCtrl")
        self._thread.start()

        # 첫 레코드 도착까지 대기 — 반환 후 data가 항상 GnssRouteRecord를 보장
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            if not self.running:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("GnssRouteProvider: timed out waiting for first record")

        logging.info("GnssRouteProvider started")

    def stop(self) -> None:
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        logging.info("GnssRouteProvider stopped")

    def _wait_for_rtk(self) -> bool:
        """RTK 신호가 준비될 때까지 대기. 활성화 시 True, 비활성화 시 False."""
        logging.info("GnssRouteProvider: waiting for reliable RTK state …")
        while self.running:
            gnss = self.rtk_provider.data
            if gnss is not None and (gnss.carrSoln or 0) >= 1:
                logging.info("GnssRouteProvider: GNSS ready")
                return True
            time.sleep(0.01)
        return False

    def _calibrate_heading(self) -> bool:
        """초기 직진으로 yaw_offset_deg 캘리브레이션. 성공 시 True, 중단 시 False."""
        logging.info("GnssRouteProvider: initial straight moving for yaw calibration")
        odom_init = self.unitree_go2_provider.get_odometry()
        yaw_init_deg = math.degrees(odom_init.yaw)

        gnss_init = self.rtk_provider.data
        lat_init, lon_init = gnss_init.lat, gnss_init.lon

        while self.running:
            t_monotonic = time.monotonic()
            odom_prev = self.unitree_go2_provider.get_odometry()

            vyaw = _init_vyaw_control(yaw_init_deg, math.degrees(odom_prev.yaw))
            rec = GnssRouteRecord(
                t_monotonic=t_monotonic,
                vx=0.5,
                vy=0.0,
                vyaw=vyaw,
            )
            with self._lock:
                self._data = rec  # dx/dy는 의미 없음(초기값 0.0). 소비자는 heading_calibrated 확인
            for _ in range(10):
                if not self.running:
                    return False
                time.sleep(0.01)

            odom = self.unitree_go2_provider.get_odometry()
            gnss = self.rtk_provider.data

            traveled = math.hypot(odom.x - odom_init.x, odom.y - odom_init.y)
            if traveled >= 3.0:
                odom_dx = odom.x - odom_init.x
                odom_dy = odom.y - odom_init.y
                gnss_west, gnss_north = _haversine_xy(lat_init, lon_init, gnss.lat, gnss.lon)
                gnss_heading = math.degrees(math.atan2(gnss_west, gnss_north))
                odom_heading = math.degrees(math.atan2(odom_dy, odom_dx))
                with self._lock:
                    self._yaw_offset_deg = _wrap_deg(gnss_heading - odom_heading)
                self._heading_calibrated = True
                logging.info("GnssRouteProvider: heading calibrated, yaw_offset_deg=%.2f°", self._yaw_offset_deg)
                return True

        return False

    def _follow_path(self, lat_snap: float, lon_snap: float) -> None:
        """
        waypoint 경로를 순서대로 따라가면서 현재 상태를 GnssRouteRecord에 저장.

        path는 [(lat_0, lon_0), (lat_1, lon_1), ...] 형태의 GPS 좌표 목록이며,
        현재 위치와 현재 목표 waypoint 사이의 거리가 reach_tol 이내가 되면
        다음 waypoint로 목표를 갱신한다. 마지막 waypoint에 도달하면 루프를 종료.
        """
        gnss_init = self.rtk_provider.data
        self._tracker = WaypointTracker(
            self.waypoints,
            reach_tol=self.reach_tol_m,
            start_lat=gnss_init.lat,
            start_lon=gnss_init.lon,
        )
        logging.info(
            "GnssRouteProvider: mission active, following %d waypoints from index=%d (%.6f, %.6f)",
            len(self.waypoints), self._tracker.idx, self.waypoints[self._tracker.idx][0], self.waypoints[self._tracker.idx][1],
        )

        odom_snap = self.unitree_go2_provider.get_odometry()

        while self.running:
            gnss = self.rtk_provider.data
            lat_cur, lon_cur = gnss.lat, gnss.lon

            odom = self.unitree_go2_provider.get_odometry()
            global_heading = math.degrees(odom.yaw) + self._yaw_offset_deg

            goal = self._tracker.update(lat_cur, lon_cur)
            if goal is None:
                self._reached_goal = True
                logging.info("GnssRouteProvider: path finished!")
                rec = GnssRouteRecord(
                    t_monotonic=time.monotonic(),
                    heading_calibrated=True,
                    reached_goal=True,
                )
                with self._lock:
                    self._data = rec
                return

            # odometry yaw에 대한 drift 보정, gnss 노이즈를 감안하여 충분히 먼 거리를 이동했을 때만 적용.
            snap_w, snap_n = _haversine_xy(lat_snap, lon_snap, lat_cur, lon_cur)
            if math.hypot(snap_w, snap_n) > self.yaw_update_move_thresh_m:
                gnss_heading = math.degrees(math.atan2(snap_w, snap_n))
                odom_dx = odom.x - odom_snap.x
                odom_dy = odom.y - odom_snap.y
                if abs(odom_dx) > 1e-6 or abs(odom_dy) > 1e-6:
                    odom_travel_deg = math.degrees(math.atan2(odom_dy, odom_dx))
                    yaw_offset_new_deg = _wrap_deg(gnss_heading - odom_travel_deg)
                    with self._lock:
                        delta = _wrap_deg(yaw_offset_new_deg - self._yaw_offset_deg)
                        self._yaw_offset_deg = _wrap_deg(self._yaw_offset_deg + self.yaw_update_alpha * delta)
                    global_heading = math.degrees(odom.yaw) + self._yaw_offset_deg
                lat_snap, lon_snap = lat_cur, lon_cur
                odom_snap = odom

            dW, dN = _haversine_xy(lat_cur, lon_cur, goal[0], goal[1])  # goal[0] = lat_goal, goal[1] = lon_goal
            dth = math.radians(global_heading)
            dx = math.cos(dth) * dN + math.sin(dth) * dW
            dy = -math.sin(dth) * dN + math.cos(dth) * dW
            heading_err = math.atan2(dy, dx)  # 로봇 바디 좌표계(+x 전방) 기준으로 목표 방향이 좌/우로 얼마나 벗어났는지(방향 오차, rad)

            vyaw = max(-self.max_vyaw, min(self.max_vyaw, heading_err))
            vy = 0.0  # go2는 측면 이동 불가
            vx = 0.0 if abs(heading_err) > self.heading_margin_rad else self.max_vx * (1.0 - abs(vyaw) / self.max_vyaw)

            rec = GnssRouteRecord(
                t_monotonic=time.monotonic(),
                heading_calibrated=self._heading_calibrated,
                reached_goal=self._reached_goal,
                dx=dx,
                dy=dy,
                vx=vx,
                vy=vy,
                vyaw=vyaw,
            )
            with self._lock:
                self._data = rec
            time.sleep(0.1)

    @property
    def data(self) -> Optional[GnssRouteRecord]:
        """최신 GnssRouteRecord. 첫 레코드 생성 전에는 None."""
        with self._lock:
            return self._data

    @property
    def current_waypoint_idx(self) -> int:
        """현재 목표 waypoint 인덱스. 경로 추종 시작 전에는 0."""
        tracker = self._tracker
        return tracker.idx if tracker is not None else 0

    @property
    def yaw_offset_deg(self) -> float:
        """GNSS calibration으로 구한 yaw offset (degrees). heading_calibrated 전에는 0.0."""
        with self._lock:
            return self._yaw_offset_deg
        
    def _run(self) -> None:
        try:
            # reached_goal은 항상 리셋. heading 캘리브레이션은 이미 완료된 경우 재사용.
            prev_calibrated = self._heading_calibrated
            self._reached_goal = False
            if not prev_calibrated:
                self._yaw_offset_deg = 0.0

            if not self.waypoints:
                logging.warning("GnssRouteProvider: waypoints가 비어 있습니다. set_path()로 경로를 설정하세요.")
                rec = GnssRouteRecord(
                    t_monotonic=time.monotonic(),
                    heading_calibrated=self._heading_calibrated,
                    reached_goal=False,
                )
                with self._lock:
                    self._data = rec
                return

            if not self._wait_for_rtk():
                return

            if prev_calibrated:
                logging.info("GnssRouteProvider: heading already calibrated (yaw_offset=%.2f°), skipping re-calibration", self._yaw_offset_deg)
            else:
                if not self._calibrate_heading():
                    return

            gnss = self.rtk_provider.data
            self._follow_path(gnss.lat, gnss.lon)
        finally:
            self.running = False
