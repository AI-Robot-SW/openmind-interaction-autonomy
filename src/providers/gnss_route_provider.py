# gnss_route_provider.py

import time
import math
import logging
import threading

from dataclasses import dataclass
from typing import Optional

from .singleton import singleton

from .rtk_provider import RtkProvider
from .unitree_go2_provider import UnitreeGo2Provider

from .utils.geo_utils import latlon_to_west_north_offset_m
from .utils.route_utils.graph_utils import PathTracker


# ===================================================================================
# GnssRouteProvider
# ===================================================================================

@dataclass(frozen=True)
class GnssRouteRecord:
    """
    t_monotonic        : 생성 시각 (monotonic)
    heading_calibrated : yaw offset KF 수렴 여부 (NavigationProvider가 주입)
    reached_goal       : 경로 최종 도달 여부
    dx                 : 목표 waypoint x 오프셋 (robot body frame, meters)
    dy                 : 목표 waypoint y 오프셋 (robot body frame, meters)
    vx                 : 전진 속도 (m/s)
    vy                 : 측면 속도 (m/s)
    vyaw               : yaw 속도 (rad/s)
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
    background thread에서 GNSS/odometry 데이터를 읽어 PathTracker 기반으로 경로를 추종하고
    최신 결과를 GnssRouteRecord로 계산해 data 프로퍼티로 노출한다.
    yaw offset과 heading calibration 상태는 NavigationProvider가 주입한다.
    """

    def __init__(
        self,
        max_vx: float = 0.8,
        max_vyaw: float = math.radians(45),
    ) -> None:
        self.max_vx = max_vx
        self.max_vyaw = max_vyaw

        self.rtk_provider = RtkProvider()
        self.unitree_go2_provider = UnitreeGo2Provider()

        self._data: Optional[GnssRouteRecord] = None
        self._lock = threading.Lock()

        self.running: bool = False
        self._thread: Optional[threading.Thread] = None

        self._tracker: Optional[PathTracker] = None

        # ---- runtime state (accessed only from control thread) ----
        self._reached_goal: bool = False
        self._yaw_offset_deg: float = 0.0
        self._heading_calibrated: bool = False
        self.heading_margin_rad: float = math.radians(45)

    def set_yaw_offset(self, yaw_offset_deg: float, heading_calibrated: bool = False) -> None:
        """NavigationProvider가 매 tick 주입하는 yaw offset과 KF 수렴 여부."""
        with self._lock:
            self._yaw_offset_deg = yaw_offset_deg
            self._heading_calibrated = heading_calibrated

    def set_tracker(self, tracker: Optional[PathTracker]) -> None:
        """새 PathTracker를 설정한다. None이면 경로를 초기화한다."""
        with self._lock:
            self._tracker = tracker
        self._reached_goal = False
        if tracker is None:
            logging.info("GnssRouteProvider: tracker cleared")
        else:
            logging.info("GnssRouteProvider: new PathTracker set (%d nodes)", tracker.progress[1])

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("GnssRouteProvider already running")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="GnssRouteCtrl")
        self._thread.start()

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

    def _follow_path(self, tracker: PathTracker) -> None:
        """
        PathTracker를 따라 주행한다.
        tracker가 교체되거나 경로가 완료되거나 running이 False가 되면 반환한다.
        """
        logging.info(
            "GnssRouteProvider: mission active, following %d nodes",
            tracker.progress[1],
        )

        while self.running:
            # tracker가 교체되면 중단하고 새 tracker로 재시작
            with self._lock:
                if self._tracker is not tracker:
                    return

            gnss = self.rtk_provider.data
            lat_cur, lon_cur = gnss.lat, gnss.lon
            odom = self.unitree_go2_provider.get_odometry()
            with self._lock:
                yaw_offset_deg = self._yaw_offset_deg
                heading_calibrated = self._heading_calibrated
            global_heading = math.degrees(odom.yaw) + yaw_offset_deg

            tracker.update(lat_cur, lon_cur, hAcc_m=gnss.hAcc_m)

            current = tracker.current_node
            if current is None:
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

            goal_node = current.node
            if current.graph.coordinate_frame != "wgs84" or goal_node.lat is None or goal_node.lon is None:
                time.sleep(0.1)
                continue

            dW, dN = latlon_to_west_north_offset_m(lat_cur, lon_cur, goal_node.lat, goal_node.lon)
            dth = math.radians(global_heading)
            dx =  math.cos(dth) * dN + math.sin(dth) * dW
            dy = -math.sin(dth) * dN + math.cos(dth) * dW
            heading_err = math.atan2(dy, dx)

            vyaw = max(-self.max_vyaw, min(self.max_vyaw, heading_err))
            vx = 0.0 if abs(heading_err) > self.heading_margin_rad else self.max_vx * (1.0 - abs(vyaw) / self.max_vyaw)

            rec = GnssRouteRecord(
                t_monotonic=time.monotonic(),
                heading_calibrated=heading_calibrated,
                reached_goal=self._reached_goal,
                dx=dx, dy=dy, vx=vx, vy=0.0, vyaw=vyaw,
            )
            with self._lock:
                self._data = rec
            time.sleep(0.1)

    @property
    def data(self) -> Optional[GnssRouteRecord]:
        with self._lock:
            return self._data

    @property
    def tracker(self) -> Optional[PathTracker]:
        with self._lock:
            return self._tracker

    @property
    def yaw_offset_deg(self) -> float:
        with self._lock:
            return self._yaw_offset_deg

    def _run(self) -> None:
        try:
            self._reached_goal = False

            # 초기 레코드 즉시 발행 (start()의 대기 조건)
            with self._lock:
                self._data = GnssRouteRecord(t_monotonic=time.monotonic())

            while self.running:
                with self._lock:
                    tracker = self._tracker

                if tracker is None or tracker.is_done:
                    rec = GnssRouteRecord(
                        t_monotonic=time.monotonic(),
                        reached_goal=True,
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.05)
                    continue

                self._follow_path(tracker)

        finally:
            self.running = False
