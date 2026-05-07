# path_follow_provider.py
"""
PathFollowProvider — PathTracker 기반 경로 추종 provider.

KfPositionProvider 에서 위치/heading 을 읽고,
PathTracker 가 제공하는 현재 목표 노드까지의 (dx, dy) 를
PathFollowRecord 로 노출한다.

  KfPositionProvider → (lat/lon 또는 x/y, θ) → PathFollowProvider → PathFollowRecord(dx, dy)

제어 흐름
---------
- active tracker: 현재 제어 출력을 담당하는 tracker
- passive tracker: active가 아닌 tracker — EKF 수렴 시 waypoint update만 수행 (dx/dy 출력 없음)
- active tracker 완료 시 다음 segment로 자연스럽게 전환
"""

import time
import logging
import threading

from dataclasses import dataclass
from typing import Optional

from .kf_position_provider import KfPositionProvider, KfPositionRecord

from .singleton import singleton

from .utils.route_utils.path_tracker import PathTracker
from .utils.geo_utils import latlon_to_body_frame, xy_to_body_frame

# ── 상수 ──────────────────────────────────────────────────────
TICK_DT: float = 0.05           # s (20 Hz)


# ===================================================================================
# PathFollowRecord
# ===================================================================================

@dataclass(frozen=True)
class PathFollowRecord:
    """
    PathFollowProvider 현재 상태 스냅샷.

    Fields
    ------
    t_monotonic   : 레코드 생성 시각 (monotonic, s)
    is_done       : 목적지 도달 여부
    node_idx      : 현재 목표 노드 인덱스 (전체 경로 기준)
    node_total    : 전체 노드 수
    frame         : 현재 좌표계 ('wgs84' or 'uwb')
    dx            : body frame 전방 거리 (m)
    dy            : body frame 좌방 거리 (m)
    """
    t_monotonic: float
    is_done: bool = False
    node_idx: int = 0
    node_total: int = 0
    frame: str = ""
    dx: float = 0.0
    dy: float = 0.0


# ===================================================================================
# PathFollowProvider
# ===================================================================================

@singleton
class PathFollowProvider:
    """
    PathTracker 기반 경로 추종 provider.

    segment별 PathTracker 리스트를 받아 순서대로 추종한다.
    active tracker가 dx/dy 출력을 담당하고, passive tracker는 위치만 업데이트한다.
    active tracker 완료 시 다음 segment로 자동 전환한다.

    tracker 없이 시작 가능 — set_trackers() 호출 전까지 대기 상태로 유지된다.
    주행 중 set_trackers() 로 경로를 교체할 수 있다.
    """

    def __init__(self) -> None:
        self._kf_pos = KfPositionProvider()

        self._trackers: list[PathTracker] = []
        self._active_idx: int = 0

        self._data: Optional[PathFollowRecord] = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("PathFollowProvider already running")
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PathFollowWorker"
        )
        self._thread.start()
        logging.info("PathFollowProvider started")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logging.info("PathFollowProvider stopped")

    # ------------------------------------------------------------------
    # Data API
    # ------------------------------------------------------------------

    def set_trackers(self, trackers: list[PathTracker]) -> None:
        """segment별 PathTracker 리스트를 설정한다. 주행 중에도 교체 가능."""
        total = sum(t.progress[1] for t in trackers)
        with self._lock:
            self._trackers = trackers
            self._active_idx = 0
            self._data = None
        logging.info("PathFollowProvider: %d nodes, %d segments", total, len(trackers))

    @property
    def data(self) -> Optional[PathFollowRecord]:
        with self._lock:
            return self._data

    @property
    def is_done(self) -> bool:
        rec = self.data
        return rec is not None and rec.is_done

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        with self._lock:
            self._data = None

        while not self._stop_evt.is_set():
            t0 = time.monotonic()

            with self._lock:
                trackers = self._trackers
                active_idx = self._active_idx

            if not trackers:
                with self._lock:
                    self._data = None
                time.sleep(TICK_DT)
                continue

            pos: Optional[KfPositionRecord] = self._kf_pos.data
            if pos is None:
                time.sleep(TICK_DT)
                continue

            self._update_passive_trackers(pos, trackers, active_idx)
            self._update_active_tracker(pos, trackers, active_idx, t0)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, TICK_DT - elapsed))

    def _update_active_tracker(
        self,
        pos: KfPositionRecord,
        trackers: list[PathTracker],
        active_idx: int,
        t0: float,
    ) -> None:
        """active tracker의 위치를 업데이트하고 제어 출력을 계산해 _data에 기록한다."""
        active = trackers[active_idx]

        total_nodes = sum(t.progress[1] for t in trackers)
        completed_nodes = sum(trackers[i].progress[1] for i in range(active_idx))

        if active.is_done:
            if active_idx + 1 < len(trackers):
                with self._lock:
                    if self._trackers is trackers:
                        self._active_idx += 1
                        active_idx = self._active_idx
                next_node = trackers[active_idx].current_node
                logging.info(
                    "PathFollowProvider: switching to segment %d (frame=%s)",
                    active_idx,
                    next_node.graph.coordinate_frame if next_node else "?",
                )
            else:
                logging.info("PathFollowProvider: goal reached")
                with self._lock:
                    self._data = PathFollowRecord(
                        t_monotonic=t0,
                        is_done=True,
                        node_idx=total_nodes,
                        node_total=total_nodes,
                    )
            return

        current = active.current_node
        if current is None:
            return

        frame = current.graph.coordinate_frame
        result = self._step_wgs84(pos, active) if frame == "wgs84" else self._step_uwb(pos, active)
        if result is None:
            # 필요 센서(RTK 또는 UWB)가 아직 준비되지 않음 → 이전 _data를 유지하지 않고
            # None으로 지워 DWA가 IDLE 상태로 전환되도록 한다
            with self._lock:
                if self._trackers is trackers:
                    self._data = None
            return

        dx, dy = result
        global_idx = completed_nodes + active.progress[0]

        with self._lock:
            if self._trackers is trackers:
                self._data = PathFollowRecord(
                    t_monotonic=t0,
                    is_done=False,
                    node_idx=global_idx,
                    node_total=total_nodes,
                    frame=frame,
                    dx=dx,
                    dy=dy,
                )

    def _update_passive_trackers(
        self,
        pos: KfPositionRecord,
        trackers: list[PathTracker],
        active_idx: int,
    ) -> None:
        """active가 아닌 tracker들을 위치 업데이트만 수행 (제어 출력 없음)."""
        for i, tracker in enumerate(trackers):
            if i == active_idx or tracker.is_done:
                continue
            current = tracker.current_node
            if current is None:
                continue
            frame = current.graph.coordinate_frame
            if frame == "wgs84":
                if pos.rtk_ready and pos.rtk_lat is not None \
                        and pos.rtk_lon is not None:
                    tracker.update(lat=pos.rtk_lat, lon=pos.rtk_lon)
            else:
                if pos.uwb_ready and pos.uwb_x_m is not None \
                        and pos.uwb_y_m is not None:
                    tracker.update(None, None, x=pos.uwb_x_m, y=pos.uwb_y_m)

    def _step_wgs84(self, pos: KfPositionRecord, tracker: PathTracker) -> Optional[tuple[float, float]]:
        if not pos.rtk_ready \
                or not pos.rtk_yaw_calibrated \
                or pos.rtk_lat is None or pos.rtk_lon is None or pos.rtk_theta_rad is None:
            return None

        tracker.update(lat=pos.rtk_lat, lon=pos.rtk_lon)
        if tracker.is_done:
            return None

        current = tracker.current_node
        if current is None:
            return None

        node = current.node
        return latlon_to_body_frame(
            pos.rtk_lat, pos.rtk_lon, pos.rtk_theta_rad,
            node.lat, node.lon,
        )

    def _step_uwb(self, pos: KfPositionRecord, tracker: PathTracker) -> Optional[tuple[float, float]]:
        if not pos.uwb_ready \
                or not pos.uwb_yaw_calibrated \
                or pos.uwb_x_m is None or pos.uwb_y_m is None or pos.uwb_theta_rad is None:
            return None

        tracker.update(None, None, x=pos.uwb_x_m, y=pos.uwb_y_m)
        if tracker.is_done:
            return None

        current = tracker.current_node
        if current is None:
            return None

        node = current.node
        return xy_to_body_frame(
            pos.uwb_x_m, pos.uwb_y_m, pos.uwb_theta_rad,
            node.x, node.y,
        )
