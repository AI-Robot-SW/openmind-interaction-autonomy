# navigation_provider.py
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .singleton import singleton
from .path_follow_provider import PathFollowProvider
from .kf_position_provider import KfPositionProvider
from .dwa_route_provider import DwaRouteProvider

from .utils.geo_utils import haversine_dist_m
from .utils.route_utils import PathFinder
from .utils.route_utils.path_tracker import PathTracker
from .utils.route_utils.graph_model import NodeRef

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


# ==============================================================================
# NavigationProvider
# ==============================================================================
@singleton
class NavigationProvider:
    """
    PathFollowProvider + DwaRouteProvider를 wrapping하는 Navigation Provider.

    PathFollowProvider(KF 측위 기반 실내외 통합 경로 추종)를 경로 입력으로 사용하고,
    DwaRouteProvider가 장애물 회피 DWA 속도 명령을 생성한다.
    heading calibration은 KfPositionProvider가 내부에서 처리한다.
    """

    def __init__(
        self,
        tick_dt: float = 0.05,
        speed_step: float = 0.2,
        speed_min: float = 0.2,
        speed_max: Optional[float] = None,   # None이면 dwa.v_max 사용
    ) -> None:
        self._path_finder = PathFinder()
        self._path_follow = PathFollowProvider()
        self._kf_pos = KfPositionProvider()
        self._dwa = DwaRouteProvider()

        self._tick_dt = float(tick_dt)
        self._speed_step = float(speed_step)
        self._speed_min = float(speed_min)
        self._speed_max = float(speed_max) if speed_max is not None else float(self._dwa.v_max)

        self._active_goal: Optional[str] = None
        self._speed_before_pause: Optional[float] = None
        self._current_path: Optional[list[list[NodeRef]]] = None
        self._path_is_set: bool = False

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

        with self._state_lock:
            self._latest_state = NavigationState(
                t_monotonic=time.monotonic(),
                mode="STOP",
            )

        logger.info("NavigationProvider stopped")

    # ---------------- control API ----------------

    def set_goal(self, place_id: str) -> None:
        """place 이름으로 목표를 설정한다. GNSS 준비 전이면 worker loop에서 재시도한다."""
        self._active_goal = place_id
        self._path_is_set = False

        try:
            self._compute_and_set_path(place_id)
        except Exception:
            logger.exception(
                "NavigationProvider.set_goal: '%s' 경로 탐색 실패, 이후 worker loop에서 재시도",
                place_id,
            )

    def _compute_and_set_path(self, place_id: str) -> None:
        """현재 GNSS 위치 기준으로 경로를 탐색하고 PathFollowProvider에 주입한다."""
        kf = self._kf_pos.data
        if kf is None or not kf.rtk_ready or not kf.rtk_yaw_calibrated:
            raise ValueError("NavigationProvider: KF heading 보정이 아직 완료되지 않았습니다")

        # EKF smoothed 위치를 사용해야 nearest node 선택이 정확하다.
        # RTK raw(gnss.lat/lon)는 수 미터 오차가 있어 잘못된 start node를 선택할 수 있다.
        if kf.rtk_lat is None or kf.rtk_lon is None:
            raise ValueError("NavigationProvider: RTK EKF 위치를 아직 사용할 수 없습니다")

        start_ref = self._path_finder.nearest_wgs_node(float(kf.rtk_lat), float(kf.rtk_lon))
        goal_ref = self._path_finder.resolve_place_to_node(place_id)

        segments = self._path_finder.find_path(start_ref, goal_ref)
        if segments is None:
            raise ValueError(
                f"NavigationProvider: {start_ref} -> {goal_ref} 경로를 찾을 수 없습니다"
            )

        trackers = [PathTracker(seg, self._path_finder._loader) for seg in segments]
        self._path_follow.set_trackers(trackers)
        self._current_path = segments
        self._path_is_set = True

        total_nodes = sum(len(s) for s in segments)
        logger.info(
            "NavigationProvider: path set '%s' -> %s (%d nodes, %d segments)",
            place_id, goal_ref, total_nodes, len(segments),
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
        self._current_path = None
        self._path_is_set = False
        self._path_follow.set_trackers([])
        logger.info("NavigationProvider.clear_goal")

    def get_active_goal(self) -> Optional[str]:
        """현재 설정된 목표 place_id를 반환합니다."""
        return self._active_goal

    def get_target_speed(self) -> float:
        """현재 목표 속도 (vx_fixed) 를 반환합니다."""
        return float(self._dwa.vx_fixed)

    def get_remaining_distance(self) -> float:
        """현재 위치에서 경로 끝까지의 남은 거리 (m) 를 반환합니다."""
        segments = self._current_path
        if not segments:
            return 0.0

        route_rec = self._path_follow.data
        if route_rec is None or route_rec.is_done:
            return 0.0

        # segments를 flat하게 펼쳐 인덱스 기반으로 누적 거리 계산
        flat_path: list[NodeRef] = [ref for seg in segments for ref in seg]
        idx = route_rec.node_idx
        loader = self._path_finder._loader

        total = math.hypot(route_rec.dx, route_rec.dy)

        for i in range(idx + 1, len(flat_path)):
            a_gid, a_nid = flat_path[i - 1]
            b_gid, b_nid = flat_path[i]
            try:
                a_graph = loader.get_graph(a_gid)
                b_graph = loader.get_graph(b_gid)
                a_node = a_graph.nodes.get(a_nid)
                b_node = b_graph.nodes.get(b_nid)
                if a_node is None or b_node is None:
                    continue
                if a_graph.coordinate_frame == "wgs84" \
                        and a_node.lat is not None and a_node.lon is not None \
                        and b_node.lat is not None and b_node.lon is not None:
                    total += haversine_dist_m(a_node.lat, a_node.lon, b_node.lat, b_node.lon)
                elif a_graph.coordinate_frame != "wgs84" \
                        and a_node.x is not None and a_node.y is not None \
                        and b_node.x is not None and b_node.y is not None:
                    total += math.hypot(b_node.x - a_node.x, b_node.y - a_node.y)
            except Exception:
                continue

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

    # ---------------- helpers ----------------

    def _retry_set_path_if_needed(self) -> None:
        """
        goal은 있는데 path가 아직 설정되지 않았다면 경로 탐색을 재시도한다.
        set_goal() 시점에 GNSS가 아직 준비되지 않았던 경우를 커버하기 위함.
        """
        if self._active_goal is None or self._path_is_set:
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
                self._retry_set_path_if_needed()

                # DWA 기반 경로 추종
                rec = self._dwa.data
                route_rec = self._path_follow.data
                reached_goal = bool(route_rec.is_done if route_rec is not None else False)

                # heading_calibrated: KfPositionProvider 기준
                kf = self._kf_pos.data
                heading_calibrated = bool(
                    (kf.rtk_yaw_calibrated if kf is not None and kf.rtk_ready else False)
                    or (kf.uwb_yaw_calibrated if kf is not None and kf.uwb_ready else False)
                )

                if rec is None:
                    st = NavigationState(
                        t_monotonic=time.monotonic(),
                        mode="IDLE",
                        heading_calibrated=heading_calibrated,
                        reached_goal=reached_goal,
                    )
                else:
                    mode = str(rec.mode)
                    if mode == "DWA":
                        vx = float(self._dwa.vx_fixed) if float(rec.vx_cmd) > 1e-6 else 0.0
                        vyaw = float(rec.vyaw_cmd)
                    else:
                        vx = 0.0
                        vyaw = 0.0

                    st = NavigationState(
                        t_monotonic=time.monotonic(),
                        vx=vx,
                        vy=0.0,
                        vyaw=vyaw,
                        mode=mode,
                        heading_calibrated=heading_calibrated,
                        reached_goal=reached_goal,
                    )

                with self._state_lock:
                    self._latest_state = st

            except Exception:
                logger.exception("Error in NavigationProvider worker loop")

            self._stop_evt.wait(self._tick_dt)
