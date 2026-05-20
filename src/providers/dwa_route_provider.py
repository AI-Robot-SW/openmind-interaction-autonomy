# dwa_route_provider.py

import time
import math
import logging
import threading
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple

from .singleton import singleton

from .bev_occupancy_grid_provider import BEVOccupancyGridProvider
from .distmap_provider import DistMapProvider
from .path_follow_provider import PathFollowProvider


@dataclass(frozen=True)
class DwaRouteRecord:
    """
    t_monotonic      : DWA 결과 생성 시각 (monotonic time)
    dx_dwa           : 선택된 local goal x (robot/body frame, meters)
    dy_dwa           : 선택된 local goal y (robot/body frame, meters)
    best_clearance_m : 선택된 local goal의 obstacle clearance (meters)
    vx_cmd           : 전진 속도 명령 (m/s)
    vyaw_cmd         : yaw 속도 명령 (rad/s)
    mode             : "DWA" | "STOP" | "IDLE"
    stop_reason      : 정지/idle 이유
    """
    t_monotonic: float
    dx_dwa: float = 0.0
    dy_dwa: float = 0.0
    best_clearance_m: float = 0.0
    vx_cmd: float = 0.0
    vyaw_cmd: float = 0.0
    mode: str = "IDLE"
    stop_reason: str = "none"


@singleton
class DwaRouteProvider:
    """
    Legacy DWACommandNode 로직을 Provider로 옮긴 버전.

    Inputs:
      - PathFollowProvider.data: dx, dy (+ is_done) — 실내외 통합 경로 추종
      - BEVOccupancyGridProvider.data: BEVFrame
      - DistMapProvider.data: DistMapFrame

    Output:
      - DwaRouteRecord (vx_cmd, vyaw_cmd, etc.)

    NOTE:
      - PathFollowProvider의 vx_cmd는 중복 제어 방지를 위해 소비하지 않음 (dx, dy만 입력으로 사용).
      - BEV/DistMap provider lifecycle은 외부(backgrounds/tests)에서 관리한다.
      - _run()은 upstream provider가 이미 유효한 dataclass 데이터를 제공한다고 가정한다.
    """

    def __init__(
        self,
        # DWA cost params
        penalty: float = 13.0,
        margin: float = 1.2,
        w_goal: float = 1.0,
        w_clear: float = 1.2,
        y_bias: float = -0.5,
        obstacle_cost: float = 1e9,
        # person stop
        person_stop_dist: float = 1.8,
        person_stop_y_width: float = 0.5,
        # window
        ahead_m: float = 2.0,
        half_width_m: float = 1.2,
        stride: int = 1,
        unknown_is_obstacle: bool = True,   # unknown(-1) 셀을 장애물과 동일하게 처리
        # speed
        kv: float = 0.6,  # reserved — currently unused (vx_cmd uses vx_fixed directly)
        kyaw: float = 1.0,
        v_max: float = 1.2,
        w_max: float = 0.75,
        v_min: float = 0.0,
        vx_fixed: float = 0.8,
        # motion
        safety_slowdown: bool = True,
        enable_turn_in_place: bool = True,
        theta_turn_deg: float = 40.0,
        allow_backward: bool = False,
        yaw_limit_deg: float = 30.0,   # 후보 셀 탐색 각도 제한 (±yaw_limit_deg)
        close_obstacle_align_dist: float = 0.4,  # 이 거리 이내 장애물이면 목표 방향으로 강제 정렬
    ) -> None:
        self._route = PathFollowProvider()
        self._bev = BEVOccupancyGridProvider()
        self._dist = DistMapProvider()

        self.dist_max_m = float(self._dist.max_dist)

        # DWA cost params
        self.penalty = float(penalty)
        self.margin = float(margin)
        self.w_goal = float(w_goal)
        self.w_clear = float(w_clear)
        self.y_bias = float(y_bias)
        self.obstacle_cost = float(obstacle_cost)

        # 사람(occ=88) 정지
        self.person_stop_dist = float(person_stop_dist)
        self.person_stop_y_width = float(person_stop_y_width)

        # window
        self.ahead_m = float(ahead_m)
        self.half_width_m = float(half_width_m)
        self.stride = int(stride)
        self.unknown_is_obstacle = bool(unknown_is_obstacle)

        # speed
        self.kyaw = float(kyaw)
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.v_min = float(v_min)
        self.vx_fixed = float(vx_fixed)

        # turn-in-place
        self.safety_slowdown = bool(safety_slowdown)
        self.enable_turn_in_place = bool(enable_turn_in_place)
        self.theta_turn = math.radians(float(theta_turn_deg))
        self.allow_backward_target = bool(allow_backward)
        self.yaw_limit = math.radians(float(yaw_limit_deg))
        self.close_obstacle_align_dist = float(close_obstacle_align_dist)

        # ---- robot anchor in grid (legacy) ----
        # NOTE: BEV 쪽 dx=-0.34 보정을 이미 했다면, 여기 -0.34는 중복일 수 있음.
        self.robot_x_offset_m = -0.34
        self.robot_y_offset_m = 0.0

        self._data: Optional[DwaRouteRecord] = None
        self._lock = threading.Lock()

        self.running: bool = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("DwaRouteProvider already running")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="DwaRouteCtrl")
        self._thread.start()

        logging.info("DwaRtoueProvider started")

    def stop(self) -> None:
        self.running = False
        
        if self._thread is None:
            return
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        logging.info("DwaRouteProvider stopped")

    @property
    def data(self) -> Optional[DwaRouteRecord]:
        with self._lock:
            return self._data
    
    def _run(self) -> None:
        logging.info(
            "DwaRouteProvider: started dist_max=%.2f",
            self.dist_max_m
        )
        try:
            while self.running:
                rec: DwaRouteRecord

                # 1) 경로 추종 goal
                g = self._route.data
                if g is None:
                    rec = DwaRouteRecord(
                            t_monotonic=time.monotonic(),
                            mode="IDLE",
                            stop_reason="no_path",
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.1)
                    continue

                dx_in = float(g.dx)
                dy_in = float(g.dy)
                reached = bool(g.is_done)

                if reached:
                    rec = DwaRouteRecord(
                            t_monotonic=time.monotonic(),
                            mode="STOP",
                            stop_reason="reached_goal",
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.1)
                    continue

                # 2) BEV occupancy
                bev_frame = self._bev.data
                occ = bev_frame.occupancy_grid
                res = float(occ.resolution)
                W = int(occ.width)
                H = int(occ.height)
                x0 = float(occ.origin_x)
                y0 = float(occ.origin_y)
                grid = np.asarray(occ.data, dtype=np.int8).reshape(H, W)

                # 3) distmap
                dist_frame = self._dist.data
                if dist_frame is None:
                    rec = DwaRouteRecord(
                            t_monotonic=time.monotonic(),
                            mode="IDLE",
                            stop_reason="distmap_unavailable",
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.1)
                    continue

                dist = np.asarray(dist_frame.dist, dtype=np.float32)
                if dist.shape != (H, W):
                    rec = DwaRouteRecord(
                            t_monotonic=time.monotonic(),
                            mode="IDLE",
                            stop_reason="bad_distmap_shape",
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.1)
                    continue

                # 4) window indices (legacy)
                j0 = int(((self.robot_x_offset_m) - x0) / res)
                i0 = int(((self.robot_y_offset_m) - y0) / res)

                j_start = max(0, j0)
                j_end = min(W, j0 + int(self.ahead_m / res) + 1)
                i_start = max(0, i0 - int(self.half_width_m / res))
                i_end = min(H, i0 + int(self.half_width_m / res) + 1)

                if j_start >= j_end or i_start >= i_end:
                    rec = DwaRouteRecord(
                            t_monotonic=time.monotonic(),
                            mode="IDLE",
                            stop_reason="window_out_of_map",
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.1)
                    continue

                step = max(1, int(self.stride))

                # 5) person stop (occ==88)
                person_close = False
                for i in range(i_start, i_end, step):
                    for j in range(j_start, j_end, step):
                        if int(grid[i, j]) == 88:
                            x_cell = j * res + x0
                            y_cell = i * res + y0
                            if abs(y_cell) > self.person_stop_y_width:
                                continue
                            if math.hypot(x_cell, y_cell) <= self.person_stop_dist:
                                person_close = True
                                break
                    if person_close:
                        break

                if person_close:
                    rec = DwaRouteRecord(
                            t_monotonic=time.monotonic(),
                            mode="STOP",
                            stop_reason=f"person_occ88_within_{self.person_stop_dist:.2f}m",
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.1)
                    continue

                # 6) cost minimization (legacy)
                best: Optional[Tuple[float, int, int, float, float, float]] = None
                m = max(1e-6, float(self.margin))
                desired_y = float(dy_in) + float(self.y_bias)

                for i in range(i_start, i_end, step):
                    y = i * res + y0
                    base_y = (y - desired_y) ** 2

                    for j in range(j_start, j_end, step):
                        occ_ij = int(grid[i, j])

                        if self.unknown_is_obstacle and occ_ij < 0:
                            obs_unknown = float(self.obstacle_cost)
                        else:
                            obs_unknown = 0.0

                        x = j * res + x0

                        # 로봇 기준 각도 제한: ±yaw_limit 이내 셀만 후보로
                        dx_r = x - self.robot_x_offset_m
                        dy_r = y - self.robot_y_offset_m
                        if abs(math.atan2(dy_r, max(dx_r, 1e-6))) > self.yaw_limit:
                            continue

                        base = (x - dx_in) ** 2 + base_y

                        d = float(dist[i, j])
                        obs_soft = (
                            float(self.penalty) * (1.0 - d / m) ** 2 if d < m else 0.0
                        )
                        obs_hard = float(self.obstacle_cost) if occ_ij >= 100 else 0.0

                        cost = (
                            float(self.w_goal) * base
                            + float(self.w_clear) * obs_soft
                            + obs_hard
                            + obs_unknown
                        )

                        if best is None or cost < best[0]:
                            best = (cost, i, j, x, y, d)

                if best is None:
                    rec = DwaRouteRecord(
                            t_monotonic=time.monotonic(),
                            mode="STOP",
                            stop_reason="no_cell_in_window",
                    )
                    with self._lock:
                        self._data = rec
                    time.sleep(0.1)
                    continue

                _, _, _, bx, by, bd = best
                dx_dwa, dy_dwa = float(bx), float(by)

                # 장애물이 매우 가까우면 목표 노드 방향으로 강제 정렬
                if bd < self.close_obstacle_align_dist:
                    dx_dwa, dy_dwa = float(dx_in), float(dy_in)

                # 7) speed generation
                theta = math.atan2(dy_dwa, dx_dwa)

                vx_raw = float(self.vx_fixed)

                if not self.allow_backward_target and dx_dwa < 0.0:
                    vx_raw = 0.0

                if self.safety_slowdown and bd < float(self.margin):
                    scale = max(0.0, min(1.0, bd / float(self.margin)))
                    vx_raw *= scale

                vyaw_cmd = max(
                    -float(self.w_max),
                    min(float(self.w_max), float(self.kyaw) * theta),
                )

                vx_cmd = vx_raw

                if self.enable_turn_in_place and abs(theta) > float(self.theta_turn):
                    vx_cmd = 0.0

                vx_cmd = max(float(self.v_min), min(float(self.v_max), vx_cmd))

                rec = DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_dwa=dx_dwa,
                    dy_dwa=dy_dwa,
                    best_clearance_m=float(bd),
                    vx_cmd=float(vx_cmd),
                    vyaw_cmd=float(vyaw_cmd),
                    mode="DWA",
                    stop_reason="none",
                )
                with self._lock:
                    self._data = rec

                time.sleep(0.1)
        finally:
            self.running = False
            logging.info("DwaRouteProvider: stopped")

