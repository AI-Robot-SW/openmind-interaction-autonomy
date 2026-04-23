# navigation_provider.py
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from .singleton import singleton
from .rtk_provider import RtkProvider
from .gnss_route_provider import GnssRouteProvider
from .dwa_route_provider import DwaRouteProvider
from .unitree_go2_provider import UnitreeGo2Provider
from .utils.geo_utils import haversine_dist_m, latlon_to_west_north_offset_m, wrap_deg
from .utils.route_utils.graph_utils import PathFinder, PathTracker

logger = logging.getLogger(__name__)


# ==============================================================================
# output state
# ==============================================================================
@dataclass(frozen=True)
class NavigationState:
    t_monotonic: float
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    mode: str = "IDLE"
    heading_calibrated: bool = False
    reached_goal: bool = False
    active_goal: Optional[str] = None


# ==============================================================================
# NavigationProvider
# ==============================================================================
@singleton
class NavigationProvider:
    """
    GnssRouteProvider + DwaRouteProvider를 wrapping하는 Navigation Provider.

      - 두 하위 provider 인스턴스를 주입받아 start/stop 관리
      - worker thread가 DwaRouteRecord를 폴링하여 NavigationState로 통합
      - get_state() / data 로 최신 상태 제공
    """

    def __init__(
        self,
        tick_dt: float = 0.05,
        speed_step: float = 0.2,
        speed_min: float = 0.2,
        speed_max: Optional[float] = None,  # None이면 dwa.v_max 사용
        calibrating_speed: float = 0.5,  # 헤딩 캘리브레이션 중 고정 전진 속도 (m/s)
        calibrating_distance_m: float = 3.0,  # 캘리브레이션에 필요한 이동 거리 (m)
    ) -> None:
        self._gnss = GnssRouteProvider()
        self._dwa = DwaRouteProvider()
        self._unitree = UnitreeGo2Provider()
        self._tick_dt = float(tick_dt)
        self._speed_step = float(speed_step)
        self._speed_min = float(speed_min)
        self._speed_max = float(speed_max) if speed_max is not None else float(self._dwa.v_max)
        self._calibrating_speed = float(calibrating_speed)
        self._calibrating_distance_m = float(calibrating_distance_m)
        self._active_goal: Optional[str] = None
        self._speed_before_pause: Optional[float] = None
        self._path_finder = PathFinder()

        # heading 캘리브레이션 상태 (singleton이므로 start/stop 간 유지)
        self._heading_calibrated: bool = False
        self._yaw_offset_deg: float = 0.0
        self._calib_odom_init = None   # OdometryData | None — None이면 미초기화
        self._calib_yaw_init_deg: float = 0.0
        self._calib_gnss_lat_init: float = 0.0
        self._calib_gnss_lon_init: float = 0.0

        self._state_lock = threading.Lock()
        self._latest_state = NavigationState(t_monotonic=time.monotonic())

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running: bool = False

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._gnss.start()
        self._dwa.start()

        self._stop_evt.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="NavigationProviderWorker",
        )
        self._thread.start()
        logger.info("NavigationProvider started")

    def stop(self) -> None:
        if self._thread is None:
            return

        self.running = False
        self._stop_evt.set()

        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("NavigationProvider worker thread did not stop within timeout")
        self._thread = None

        self._dwa.stop()
        self._gnss.stop()

        with self._state_lock:
            self._latest_state = NavigationState(
                t_monotonic=time.monotonic(),
                mode="STOP",
            )

        logger.info("NavigationProvider stopped")

    # ---------------- control API ----------------
    def set_goal(self, place_id: str) -> None:
        """place 이름으로 목표를 설정한다.

        - heading 캘리브레이션이 완료된 상태면 현재 위치에서 즉시 경로를 탐색한다.
        - 미완료 상태면 goal을 저장하고, 캘리브레이션 완료 후 _run()에서 자동 탐색한다.
          (캘리브레이션 중 로봇이 이동하므로, 완료 시점의 위치로 경로를 탐색해야 정확하다.)
        """
        if not self.running:
            self.start()
        self._active_goal = place_id
        if self._heading_calibrated:
            self._compute_and_set_path(place_id)
        else:
            logger.info(
                "NavigationProvider.set_goal: '%s' 저장 — heading 캘리브레이션 완료 후 경로 탐색",
                place_id,
            )

    def _compute_and_set_path(self, place_id: str) -> None:
        """현재 GNSS 위치 기준으로 경로를 탐색하고 GnssRouteProvider에 주입한다."""
        gnss = RtkProvider().data
        start_ref = self._path_finder.nearest_wgs_node(float(gnss.lat), float(gnss.lon))
        goal_ref  = self._path_finder.resolve_place_to_node(place_id)

        path = self._path_finder.find_path(start_ref, goal_ref)
        if path is None:
            raise ValueError(
                f"NavigationProvider: {start_ref} → {goal_ref} 경로를 찾을 수 없습니다"
            )

        tracker = PathTracker(path, self._path_finder._loader)
        self._gnss.set_yaw_offset(self._yaw_offset_deg)
        self._gnss.set_tracker(tracker)
        logger.info(
            "NavigationProvider: path set '%s' → %s (%d nodes)",
            place_id, goal_ref, len(path),
        )

    def step_faster(self) -> None:
        """속도를 한 단계 높입니다 (최대 speed_max)."""
        current = self._dwa.vx_fixed
        new_speed = min(self._speed_max, round(current + self._speed_step, 2))
        self._dwa.vx_fixed = new_speed
        logger.info("NavigationProvider.step_faster: %.2f -> %.2f m/s", current, new_speed)

    def step_slower(self) -> None:
        """속도를 한 단계 낮춥니다 (최소 speed_min)."""
        current = self._dwa.vx_fixed
        new_speed = max(self._speed_min, round(current - self._speed_step, 2))
        self._dwa.vx_fixed = new_speed
        logger.info("NavigationProvider.step_slower: %.2f -> %.2f m/s", current, new_speed)

    def get_next_move(self) -> Tuple[float, float, float]:
        """현재 이동 명령 (vx, vy, vyaw) 을 반환합니다."""
        st = self.get_state()
        return (st.vx, st.vy, st.vyaw)

    def pause(self) -> None:
        """속도를 0으로 설정합니다. 경로와 내비게이션 스레드는 유지됩니다."""
        if self._speed_before_pause is None:
            self._speed_before_pause = float(self._dwa.vx_fixed)
        self._dwa.vx_fixed = 0.0
        logger.info("NavigationProvider.pause: speed %.2f -> 0.0 m/s", self._speed_before_pause)

    def resume(self) -> None:
        """pause() 이전 속도로 복원합니다. pause() 없이 호출되면 speed_min으로 시작합니다."""
        target = self._speed_before_pause if self._speed_before_pause is not None else self._speed_min
        self._dwa.vx_fixed = target
        logger.info("NavigationProvider.resume: speed 0.0 -> %.2f m/s", target)
        self._speed_before_pause = None

    def clear_goal(self) -> None:
        """경로를 초기화합니다."""
        self._active_goal = None
        self._gnss.set_tracker(None)
        logger.info("NavigationProvider.clear_goal")

    def get_active_goal(self) -> Optional[str]:
        """현재 설정된 목표 place_id를 반환합니다."""
        return self._active_goal

    def get_target_speed(self) -> float:
        """현재 목표 속도 (vx_fixed) 를 반환합니다."""
        return float(self._dwa.vx_fixed)

    def get_remaining_distance(self) -> float:
        """현재 위치에서 경로 끝까지의 남은 거리 (m) 를 반환합니다."""
        tracker = self._gnss.tracker
        if tracker is None or tracker.is_done:
            return 0.0

        current = tracker.current_node
        if current is None or current.graph.coordinate_frame != "wgs84":
            return 0.0

        try:
            gnss = RtkProvider().data
            if gnss is None:
                return 0.0
            lat, lon = float(gnss.lat), float(gnss.lon)
        except Exception:
            return 0.0

        total = haversine_dist_m(lat, lon, current.node.lat, current.node.lon)

        idx, total_nodes = tracker.progress
        path = tracker.path
        loader = self._path_finder._loader
        for i in range(idx + 1, total_nodes):
            a_gid, a_nid = path[i - 1]
            b_gid, b_nid = path[i]
            if a_gid != b_gid or a_gid != "kist_outdoor":
                continue
            graph = loader.get_graph(a_gid)
            a_node, b_node = graph.nodes.get(a_nid), graph.nodes.get(b_nid)
            if a_node and b_node and a_node.lat and b_node.lat:
                total += haversine_dist_m(a_node.lat, a_node.lon, b_node.lat, b_node.lon)

        return float(total)

    # ---------------- state API ----------------

    def get_state(self) -> NavigationState:
        with self._state_lock:
            return self._latest_state

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        st = self.get_state()
        if not self.running:
            return None
        return {
            "t_monotonic": st.t_monotonic,
            "vx": st.vx,
            "vy": st.vy,
            "vyaw": st.vyaw,
            "mode": st.mode,
            "heading_calibrated": st.heading_calibrated,
            "reached_goal": st.reached_goal,
            "active_goal": st.active_goal,
        }

    # ---------------- worker ----------------

    # calibration P-controller constants
    _CALIB_KP_YAW: float = 0.6
    _CALIB_VYAW_LIMIT: float = 0.6

    def _calibration_tick(self, gnss_rtk) -> NavigationState:
        """매 tick 호출되는 heading 캘리브레이션 로직.

        직진 _calibrating_distance_m 이동 후 GNSS-odom 방향 차이로 yaw_offset을 결정한다.
        완료 시 _heading_calibrated=True 로 설정하고 경로를 탐색·주입한다.
        """
        # RTK fix 대기
        if gnss_rtk is None or (gnss_rtk.carrSoln or 0) < 1:
            now = time.monotonic()
            if not hasattr(self, '_calib_rtk_log_t') or now - self._calib_rtk_log_t >= 5.0:
                self._calib_rtk_log_t = now
                carr = gnss_rtk.carrSoln if gnss_rtk is not None else None
                logger.info(
                    "NavigationProvider: RTK float/fix 대기 중 (carrSoln=%s) — float(1) 이상 확보 후 캘리브레이션 시작",
                    carr,
                )
            return NavigationState(t_monotonic=time.monotonic(), mode="CALIBRATING",
                                   heading_calibrated=False,
                                   active_goal=self._active_goal)

        odom = self._unitree.get_odometry()
        if odom is None:
            return NavigationState(t_monotonic=time.monotonic(), mode="CALIBRATING",
                                   heading_calibrated=False,
                                   active_goal=self._active_goal)

        # 첫 tick에서 기준점 초기화
        if self._calib_odom_init is None:
            self._calib_odom_init = odom
            self._calib_yaw_init_deg = math.degrees(odom.yaw)
            self._calib_gnss_lat_init = float(gnss_rtk.lat)
            self._calib_gnss_lon_init = float(gnss_rtk.lon)
            logger.info("NavigationProvider: heading calibration started (straight %.1fm)",
                        self._calibrating_distance_m)

        # 직진 유지를 위한 P제어
        yaw_err_rad = math.radians(wrap_deg(self._calib_yaw_init_deg - math.degrees(odom.yaw)))
        vyaw = max(-self._CALIB_VYAW_LIMIT,
                   min(self._CALIB_VYAW_LIMIT, self._CALIB_KP_YAW * yaw_err_rad))

        traveled = math.hypot(odom.x - self._calib_odom_init.x,
                              odom.y - self._calib_odom_init.y)

        if traveled >= self._calibrating_distance_m:
            # yaw_offset 계산
            odom_dx = odom.x - self._calib_odom_init.x
            odom_dy = odom.y - self._calib_odom_init.y
            gnss_west, gnss_north = latlon_to_west_north_offset_m(
                self._calib_gnss_lat_init, self._calib_gnss_lon_init,
                float(gnss_rtk.lat), float(gnss_rtk.lon),
            )
            gnss_heading_deg  = math.degrees(math.atan2(gnss_west, gnss_north))
            odom_heading_deg  = math.degrees(math.atan2(odom_dy, odom_dx))
            self._yaw_offset_deg = wrap_deg(gnss_heading_deg - odom_heading_deg)
            self._heading_calibrated = True
            self._calib_odom_init = None  # 상태 초기화
            logger.info("NavigationProvider: heading calibrated, yaw_offset_deg=%.2f°",
                        self._yaw_offset_deg)

            # 캘리브레이션 완료 위치(현재 위치)에서 경로 탐색
            if self._active_goal is not None:
                try:
                    self._compute_and_set_path(self._active_goal)
                except Exception:
                    logger.exception("NavigationProvider: 캘리브레이션 후 경로 탐색 실패")

        return NavigationState(
            t_monotonic=time.monotonic(),
            mode="CALIBRATING",
            vx=self._calibrating_speed,
            vyaw=vyaw,
            heading_calibrated=self._heading_calibrated,
            active_goal=self._active_goal,
        )

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                gnss_rtk = RtkProvider().data

                # Phase 1: heading 캘리브레이션
                if self._active_goal is not None and not self._heading_calibrated:
                    st = self._calibration_tick(gnss_rtk)
                    with self._state_lock:
                        self._latest_state = st
                    self._stop_evt.wait(self._tick_dt)
                    continue

                # Phase 2: DWA 기반 경로 추종
                rec = self._dwa.data
                gnss_rec = self._gnss.data
                reached_goal = bool(gnss_rec.reached_goal if gnss_rec is not None else False)

                if rec is None:
                    st = NavigationState(
                        t_monotonic=time.monotonic(),
                        mode="IDLE",
                        heading_calibrated=self._heading_calibrated,
                        reached_goal=reached_goal,
                        active_goal=self._active_goal,
                    )
                else:
                    mode = str(rec.mode)
                    if mode == "DWA":
                        # vx_cmd > 0이면 vx_fixed(step_faster/step_slower로 조정 가능)로 override.
                        # vx_cmd == 0은 turn-in-place 신호이므로 그대로 존중.
                        vx = float(self._dwa.vx_fixed) if float(rec.vx_cmd) > 1e-6 else 0.0
                        vyaw = float(rec.vyaw_cmd) if vx > 1e-6 else 0.0
                    else:
                        vx = 0.0
                        vyaw = 0.0
                    st = NavigationState(
                        t_monotonic=time.monotonic(),
                        vx=vx,
                        vy=0.0,
                        vyaw=vyaw,
                        mode=mode,
                        heading_calibrated=self._heading_calibrated,
                        reached_goal=reached_goal,
                        active_goal=self._active_goal,
                    )
                with self._state_lock:
                    self._latest_state = st
            except Exception:
                logger.exception("Error in NavigationProvider worker loop")

            self._stop_evt.wait(self._tick_dt)
