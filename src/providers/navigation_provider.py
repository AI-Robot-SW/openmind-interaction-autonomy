# navigation_provider.py
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

from .singleton import singleton
from .rtk_provider import RtkProvider
from .gnss_route_provider import GnssRouteProvider
from .dwa_route_provider import DwaRouteProvider
from .unitree_go2_provider import UnitreeGo2Provider
from .utils.geo_utils import haversine_dist_m, latlon_to_west_north_offset_m, wrap_deg
from .utils.route_utils.graph_utils import PathFinder, PathTracker
from .utils.kf_utils import YawOffsetKF

logger = logging.getLogger(__name__)


# ==============================================================================
# helper dataclasses
# ==============================================================================
@dataclass(frozen=True)
class HeadingAlignmentSample:
    t_monotonic: float
    gnss_lat: float
    gnss_lon: float
    odom_x: float
    odom_y: float
    odom_yaw_deg: float


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


# ==============================================================================
# NavigationProvider
# ==============================================================================
@singleton
class NavigationProvider:
    """
    GnssRouteProvider + DwaRouteProvider를 wrapping하는 Navigation Provider.

    변경점:
      - 기존 calibration phase 제거
      - yaw offset은 YawOffsetKF로 실시간 추정
      - GNSS displacement heading vs odom displacement heading 차이를 측정값으로 사용
      - 충분한 관측이 쌓이면 heading_calibrated=True
    """

    def __init__(
        self,
        tick_dt: float = 0.05,
        speed_step: float = 0.2,
        speed_min: float = 0.2,
        speed_max: Optional[float] = None,   # None이면 dwa.v_max 사용

        # yaw offset KF params
        yaw_offset_p_init_deg2: float = 180.0 ** 2,
        yaw_offset_q_deg2_per_sec: float = 0.05,
        yaw_offset_r_deg2: float = 25.0,
        yaw_offset_converged_std_deg: float = 10.0,
        yaw_offset_min_updates_for_convergence: int = 3,

        # measurement gating params
        alignment_min_baseline_m: float = 1.0,
        alignment_max_window_sec: float = 3.0,
        alignment_max_turn_deg: float = 20.0,
    ) -> None:
        self._gnss = GnssRouteProvider()
        self._dwa = DwaRouteProvider()
        self._unitree = UnitreeGo2Provider()

        self._tick_dt = float(tick_dt)
        self._speed_step = float(speed_step)
        self._speed_min = float(speed_min)
        self._speed_max = float(speed_max) if speed_max is not None else float(self._dwa.v_max)

        self._active_goal: Optional[str] = None
        self._speed_before_pause: Optional[float] = None
        self._path_finder = PathFinder()

        # yaw offset estimation state
        self._yaw_offset_kf = YawOffsetKF(
            p_init_deg2=float(yaw_offset_p_init_deg2),
            q_deg2_per_sec=float(yaw_offset_q_deg2_per_sec),
            r_deg2=float(yaw_offset_r_deg2),
            converged_std_deg=float(yaw_offset_converged_std_deg),
            min_updates_for_convergence=int(yaw_offset_min_updates_for_convergence),
        )
        self._yaw_offset_deg: float = 0.0
        self._heading_calibrated: bool = False
        self._last_yaw_kf_predict_t: float = time.monotonic()

        self._alignment_min_baseline_m = float(alignment_min_baseline_m)
        self._alignment_max_window_sec = float(alignment_max_window_sec)
        self._alignment_max_turn_deg = float(alignment_max_turn_deg)
        self._heading_alignment_samples: Deque[HeadingAlignmentSample] = deque(maxlen=256)

        self._last_path_retry_log_t: float = 0.0

        self._state_lock = threading.Lock()
        self._latest_state = NavigationState(t_monotonic=time.monotonic())

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running: bool = False

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._heading_alignment_samples.clear()
        self._last_yaw_kf_predict_t = time.monotonic()

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

        self._heading_alignment_samples.clear()

        with self._state_lock:
            self._latest_state = NavigationState(
                t_monotonic=time.monotonic(),
                mode="STOP",
            )

        logger.info("NavigationProvider stopped")

    # ---------------- control API ----------------

    def set_goal(self, place_id: str) -> None:
        """
        place 이름으로 목표를 설정한다.

        이 버전에서는 calibration phase를 기다리지 않고
        가능한 즉시 현재 GNSS 위치 기준으로 경로를 탐색한다.
        yaw offset은 백그라운드에서 계속 보정된다.
        """
        if not self.running:
            self.start()

        self._active_goal = place_id

        try:
            self._compute_and_set_path(place_id)
        except Exception:
            logger.exception(
                "NavigationProvider.set_goal: '%s' 경로 탐색 실패, 이후 worker loop에서 재시도",
                place_id,
            )

    def _compute_and_set_path(self, place_id: str) -> None:
        """현재 GNSS 위치 기준으로 경로를 탐색하고 GnssRouteProvider에 주입한다."""
        gnss = RtkProvider().data
        if gnss is None or gnss.lat is None or gnss.lon is None:
            raise ValueError("NavigationProvider: GNSS 위치를 아직 사용할 수 없습니다")

        start_ref = self._path_finder.nearest_wgs_node(float(gnss.lat), float(gnss.lon))
        goal_ref = self._path_finder.resolve_place_to_node(place_id)

        path = self._path_finder.find_path(start_ref, goal_ref)
        if path is None:
            raise ValueError(
                f"NavigationProvider: {start_ref} -> {goal_ref} 경로를 찾을 수 없습니다"
            )

        tracker = PathTracker(path, self._path_finder._loader)
        self._gnss.set_yaw_offset(self._yaw_offset_deg)
        self._gnss.set_tracker(tracker)

        logger.info(
            "NavigationProvider: path set '%s' -> %s (%d nodes), yaw_offset_deg=%.2f",
            place_id, goal_ref, len(path), self._yaw_offset_deg,
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
            if (
                a_node is not None and b_node is not None
                and a_node.lat is not None and b_node.lat is not None
                and a_node.lon is not None and b_node.lon is not None
            ):
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
        }

    # ---------------- yaw offset estimation helpers ----------------

    def _publish_current_yaw_offset(self) -> None:
        self._yaw_offset_deg = float(self._yaw_offset_kf.x)
        self._heading_calibrated = bool(self._yaw_offset_kf.converged)
        self._gnss.set_yaw_offset(self._yaw_offset_deg, self._heading_calibrated)

    def _compute_heading_measurement_variance_deg2(
        self,
        gnss_horizontal_accuracy_m: float,
        baseline_m: float,
    ) -> float:
        """
        두 GNSS position 차분으로 heading을 계산할 때의 분산 근사.

        sigma_heading_rad ≈ sqrt(2) * sigma_pos / baseline
        """
        sigma_position_m = max(float(gnss_horizontal_accuracy_m), 0.05)
        baseline_m = max(float(baseline_m), 0.1)

        sigma_heading_rad = math.sqrt(2.0) * sigma_position_m / baseline_m
        sigma_heading_deg = math.degrees(sigma_heading_rad)

        # 너무 낙관적으로 작아지지 않도록 하한
        sigma_heading_deg = max(sigma_heading_deg, 1.0)
        return sigma_heading_deg ** 2

    def _select_alignment_reference_sample(
        self,
        latest_sample: HeadingAlignmentSample,
    ) -> Optional[HeadingAlignmentSample]:
        """
        latest_sample과 비교할 기준 샘플을 찾는다.

        조건:
          - 너무 오래된 샘플은 제외
          - odom baseline이 충분한 샘플만 허용
          - deque는 오래된 순서로 순회되므로, 가능한 한 baseline이 큰 샘플이 먼저 잡힌다
        """
        for old_sample in self._heading_alignment_samples:
            age_sec = latest_sample.t_monotonic - old_sample.t_monotonic
            if age_sec <= 0.0:
                continue
            if age_sec > self._alignment_max_window_sec:
                continue

            odom_dx = latest_sample.odom_x - old_sample.odom_x
            odom_dy = latest_sample.odom_y - old_sample.odom_y
            odom_baseline_m = math.hypot(odom_dx, odom_dy)

            if odom_baseline_m < self._alignment_min_baseline_m:
                continue

            return old_sample

        return None

    def _update_yaw_offset_filter(self, gnss_rtk: Any) -> None:
        """
        실시간 yaw offset 추정.

        1) KF predict
        2) 최신 GNSS + odom sample 적재
        3) baseline / RTK 품질 / turn gating을 통과하면
           GNSS heading - odom heading 으로 KF update
        """
        now = time.monotonic()
        dt_sec = now - self._last_yaw_kf_predict_t
        self._last_yaw_kf_predict_t = now

        self._yaw_offset_kf.predict(dt_sec)
        self._publish_current_yaw_offset()

        odom = self._unitree.get_odometry()
        if odom is None:
            return

        if gnss_rtk is None:
            return

        if gnss_rtk.lat is None or gnss_rtk.lon is None:
            return

        try:
            latest_sample = HeadingAlignmentSample(
                t_monotonic=now,
                gnss_lat=float(gnss_rtk.lat),
                gnss_lon=float(gnss_rtk.lon),
                odom_x=float(odom.x),
                odom_y=float(odom.y),
                odom_yaw_deg=math.degrees(float(odom.yaw)),
            )
        except Exception:
            return

        self._heading_alignment_samples.append(latest_sample)

        # RTK quality gating
        if (gnss_rtk.carrSoln or 0) < 1:
            return

        reference_sample = self._select_alignment_reference_sample(latest_sample)
        if reference_sample is None:
            return

        odom_dx = latest_sample.odom_x - reference_sample.odom_x
        odom_dy = latest_sample.odom_y - reference_sample.odom_y
        odom_baseline_m = math.hypot(odom_dx, odom_dy)
        if odom_baseline_m < self._alignment_min_baseline_m:
            return

        # 너무 많이 회전한 윈도우는 displacement heading 비교에 부적합
        odom_turn_deg = abs(wrap_deg(latest_sample.odom_yaw_deg - reference_sample.odom_yaw_deg))
        if odom_turn_deg > self._alignment_max_turn_deg:
            return

        gnss_west_m, gnss_north_m = latlon_to_west_north_offset_m(
            reference_sample.gnss_lat,
            reference_sample.gnss_lon,
            latest_sample.gnss_lat,
            latest_sample.gnss_lon,
        )
        gnss_baseline_m = math.hypot(gnss_west_m, gnss_north_m)
        if gnss_baseline_m < self._alignment_min_baseline_m:
            return

        gnss_heading_deg = math.degrees(math.atan2(gnss_west_m, gnss_north_m))
        odom_heading_deg = math.degrees(math.atan2(odom_dy, odom_dx))
        measured_yaw_offset_deg = wrap_deg(gnss_heading_deg - odom_heading_deg)

        baseline_m = min(gnss_baseline_m, odom_baseline_m)
        gnss_horizontal_accuracy_m = float(getattr(gnss_rtk, "hAcc_m", 1.0) or 1.0)

        measurement_variance_deg2 = self._compute_heading_measurement_variance_deg2(
            gnss_horizontal_accuracy_m=gnss_horizontal_accuracy_m,
            baseline_m=baseline_m,
        )

        self._yaw_offset_kf.update(
            z_deg=measured_yaw_offset_deg,
            r_deg2=measurement_variance_deg2,
        )
        self._publish_current_yaw_offset()

    def _retry_set_path_if_needed(self) -> None:
        """
        goal은 있는데 tracker가 아직 없다면 경로 탐색을 재시도한다.
        set_goal() 시점에 GNSS가 아직 준비되지 않았던 경우를 커버하기 위함.
        """
        if self._active_goal is None:
            return

        if self._gnss.tracker is not None:
            return

        try:
            self._compute_and_set_path(self._active_goal)
        except Exception:
            now = time.monotonic()
            if now - self._last_path_retry_log_t >= 5.0:
                self._last_path_retry_log_t = now
                logger.info(
                    "NavigationProvider: path retry pending for goal '%s'",
                    self._active_goal,
                )

    # ---------------- worker ----------------

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                gnss_rtk = RtkProvider().data

                # yaw offset을 항상 실시간 추정
                self._update_yaw_offset_filter(gnss_rtk)

                # 필요한 경우 path 재시도
                self._retry_set_path_if_needed()

                # DWA 기반 경로 추종
                rec = self._dwa.data
                gnss_rec = self._gnss.data
                reached_goal = bool(gnss_rec.reached_goal if gnss_rec is not None else False)

                if rec is None:
                    st = NavigationState(
                        t_monotonic=time.monotonic(),
                        mode="IDLE",
                        heading_calibrated=self._heading_calibrated,
                        reached_goal=reached_goal,
                    )
                else:
                    mode = str(rec.mode)
                    if mode == "DWA":
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
                    )

                with self._state_lock:
                    self._latest_state = st

            except Exception:
                logger.exception("Error in NavigationProvider worker loop")

            self._stop_evt.wait(self._tick_dt)


