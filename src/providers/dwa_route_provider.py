# dwa_route_provider.py
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .bev_occupancy_grid_provider import BEVOccupancyGridProvider
from .gnss_route_provider import GnssRouteProvider

logger = logging.getLogger(__name__)

# ==============================================================================
# Optional backends
# ==============================================================================

# CUDA (optional)
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    _CUDA_OK = True
except Exception:
    cuda = None
    SourceModule = None
    _CUDA_OK = False

# CPU fallback (optional)
try:
    from scipy.ndimage import distance_transform_edt

    _SCIPY_OK = True
except Exception:
    distance_transform_edt = None
    _SCIPY_OK = False


# ==============================================================================
# DistMapBuilder (embedded)
# ==============================================================================

class DistMapBuilder:
    """
    DistMap 생성기 (캐시 포함).

    Legacy semantics:
      - obstacle mask = (grid > 0)
      - 반환 dist_map 단위: meters, [0, max_dist]로 clip

    method:
      - "bfs_cuda" : CUDA wave propagation (반복)
      - "bf_cuda"  : CUDA brute-force
      - "cpu"      : scipy EDT
    """

    _KERNEL_BFS = r"""
    __global__ void update_distance(
        float *dist, const float *obstacle,
        int width, int height, float res, float max_dist, int *changed)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        int idx = y * width + x;
        if (obstacle[idx] > 0.5f) return;

        float min_dist = dist[idx];
        float dirs[8][2] = {
            {-1,0},{1,0},{0,-1},{0,1},
            {-1,-1},{-1,1},{1,-1},{1,1}
        };

        for (int i = 0; i < 8; i++) {
            int nx = x + (int)dirs[i][0];
            int ny = y + (int)dirs[i][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                float step = res * ((dirs[i][0] != 0 && dirs[i][1] != 0) ? 1.4142f : 1.0f);
                float new_dist = dist[nidx] + step;
                if (new_dist < min_dist && new_dist <= max_dist) {
                    min_dist = new_dist;
                }
            }
        }

        if (min_dist + 1e-6f < dist[idx]) {
            dist[idx] = min_dist;
            *changed = 1;
        }
    }
    """

    _KERNEL_BF = r"""
    __global__ void compute_dist_map(
        float *dist, const int *obstacles, int n_obs,
        int width, int height, float res, float max_dist)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // j
        int y = blockIdx.y * blockDim.y + threadIdx.y; // i
        if (x >= width || y >= height) return;

        float min_d2 = max_dist * max_dist;

        for (int i = 0; i < n_obs; i++) {
            int oy = obstacles[2 * i + 0];
            int ox = obstacles[2 * i + 1];
            float dx = (float)(x - ox);
            float dy = (float)(y - oy);
            float d2 = (dx * dx + dy * dy) * res * res;
            if (d2 < min_d2) min_d2 = d2;
        }

        int idx = y * width + x;
        dist[idx] = sqrtf(min_d2);
    }
    """

    def __init__(self, *, method: str = "bfs_cuda", max_dist: float = 3.0, use_cache: bool = True) -> None:
        self.method = (method or "bfs_cuda").lower()
        self.max_dist = float(max_dist)
        self.use_cache = bool(use_cache)

        self._cache_key: tuple | None = None
        self._cache_dist: np.ndarray | None = None

        self._fn_bfs = None
        self._fn_bf = None

        if _CUDA_OK and SourceModule is not None:
            # compile once
            mod_bfs = SourceModule(self._KERNEL_BFS)
            self._fn_bfs = mod_bfs.get_function("update_distance")
            mod_bf = SourceModule(self._KERNEL_BF)
            self._fn_bf = mod_bf.get_function("compute_dist_map")

    @property
    def cuda_available(self) -> bool:
        return _CUDA_OK and (self._fn_bfs is not None) and (self._fn_bf is not None) and (cuda is not None)

    @property
    def scipy_available(self) -> bool:
        return _SCIPY_OK and (distance_transform_edt is not None)

    def compute(self, grid_int8: np.ndarray, res: float, stamp: float) -> np.ndarray:
        grid = np.asarray(grid_int8, dtype=np.int8)
        if grid.ndim != 2:
            raise ValueError("grid must be 2D (HxW)")

        H, W = grid.shape
        res = float(res)
        key = (float(stamp), H, W, res, self.method, self.max_dist)

        if self.use_cache and self._cache_key == key and self._cache_dist is not None:
            return self._cache_dist

        m = self.method
        if m in ("bfs_cuda", "bfs", "cuda"):
            dist = self._compute_bfs_cuda(grid, res, self.max_dist)
        elif m in ("bf_cuda", "bf", "brute", "bruteforce"):
            dist = self._compute_bf_cuda(grid, res, self.max_dist)
        elif m in ("cpu", "edt"):
            dist = self._compute_cpu(grid, res, self.max_dist)
        else:
            raise ValueError(f"unknown distmap method: {self.method}")

        if self.use_cache:
            self._cache_key = key
            self._cache_dist = dist
        return dist

    def _compute_bfs_cuda(self, grid: np.ndarray, res: float, max_dist: float) -> np.ndarray:
        if not self.cuda_available:
            raise RuntimeError("CUDA not available")

        H, W = grid.shape
        obstacle = (grid > 0).astype(np.float32)
        dist_map = np.where(obstacle > 0, 0.0, np.inf).astype(np.float32)

        obstacle_gpu = cuda.mem_alloc(obstacle.nbytes)
        dist_gpu = cuda.mem_alloc(dist_map.nbytes)
        changed_gpu = cuda.mem_alloc(np.int32().nbytes)

        cuda.memcpy_htod(obstacle_gpu, obstacle)
        cuda.memcpy_htod(dist_gpu, dist_map)

        block = (16, 16, 1)
        grid_dim = ((W + 15) // 16, (H + 15) // 16)

        iteration = 0
        while True:
            changed = np.zeros(1, dtype=np.int32)
            cuda.memcpy_htod(changed_gpu, changed)

            self._fn_bfs(
                dist_gpu, obstacle_gpu,
                np.int32(W), np.int32(H),
                np.float32(res), np.float32(max_dist),
                changed_gpu,
                block=block, grid=grid_dim
            )

            cuda.memcpy_dtoh(changed, changed_gpu)
            iteration += 1
            if changed[0] == 0 or iteration > 1000:
                break

        cuda.memcpy_dtoh(dist_map, dist_gpu)
        dist_map[np.isinf(dist_map)] = max_dist
        return np.clip(dist_map, 0.0, max_dist).astype(np.float32)

    def _compute_bf_cuda(self, grid: np.ndarray, res: float, max_dist: float) -> np.ndarray:
        if not self.cuda_available:
            raise RuntimeError("CUDA not available")

        H, W = grid.shape
        obs = np.argwhere(grid > 0).astype(np.int32)  # (n,2) as (i,j)
        n_obs = obs.shape[0]
        if n_obs == 0:
            return np.full((H, W), max_dist, dtype=np.float32)

        dist_map = np.full((H, W), max_dist, dtype=np.float32)

        dist_gpu = cuda.mem_alloc(dist_map.nbytes)
        obs_gpu = cuda.mem_alloc(obs.nbytes)

        cuda.memcpy_htod(dist_gpu, dist_map)
        cuda.memcpy_htod(obs_gpu, obs)

        block = (16, 16, 1)
        grid_dim = ((W + 15) // 16, (H + 15) // 16)

        self._fn_bf(
            dist_gpu, obs_gpu,
            np.int32(n_obs),
            np.int32(W), np.int32(H),
            np.float32(res), np.float32(max_dist),
            block=block, grid=grid_dim
        )

        cuda.memcpy_dtoh(dist_map, dist_gpu)
        return np.clip(dist_map, 0.0, max_dist).astype(np.float32)

    def _compute_cpu(self, grid: np.ndarray, res: float, max_dist: float) -> np.ndarray:
        if not self.scipy_available:
            raise RuntimeError("SciPy not available")

        obs = (grid.astype(np.int16) > 0)
        free = ~obs
        dist_cells = distance_transform_edt(free)
        dist_m = (dist_cells * float(res)).astype(np.float32)
        return np.clip(dist_m, 0.0, max_dist).astype(np.float32)


# ==============================================================================
# Output record
# ==============================================================================

@dataclass(frozen=True)
class DwaRouteRecord:
    t_monotonic: float

    # GNSS input goal (robot/body frame, m)
    dx_in: float = 0.0
    dy_in: float = 0.0
    heading_calibrated: bool = False
    reached_goal: bool = False

    # chosen local goal (robot/body frame, m)
    dx_dwa: float = float("nan")
    dy_dwa: float = float("nan")
    best_clearance_m: float = float("nan")

    # output command
    vx_cmd: float = 0.0
    vyaw_cmd: float = 0.0

    # state / debug
    mode: str = "IDLE"          # "DWA" | "STOP" | "IDLE"
    stop_reason: str = "none"
    dist_method: str = ""
    occ_timestamp: float = 0.0


# ==============================================================================
# DwaRouteProvider
# ==============================================================================

class DwaRouteProvider:
    """
    Legacy DWACommandNode 로직을 Provider로 옮긴 버전 (distmap builder 내장).

    Inputs:
      - GnssRouteProvider.get_record(): dx,dy (+ heading_calibrated, reached_goal)
      - BEVOccupancyGridProvider.data["occupancy_grid"]: grid + meta

    Output:
      - DwaRouteRecord (vx_cmd, vyaw_cmd, etc.)

    NOTE:
      - GNSS의 vx/vyaw는 중복 제어 방지를 위해 소비하지 않음 (dx,dy만 입력으로 사용).
    """

    def __init__(
        self,
        *,
        gnss_route_provider: GnssRouteProvider,
        bev_provider: Optional[BEVOccupancyGridProvider] = None,
        # loop timing — control_rate_hz takes precedence over timer_dt
        timer_dt: float = 0.1,
        control_rate_hz: Optional[float] = None,
        # distmap
        dist_method: str = "bfs_cuda",
        dist_max_m: float = 3.0,
        # logging (reserved, not yet wired)
        log_csv_path: Optional[str] = None,
        # DWA cost params
        penalty: float = 13.0,
        margin: float = 1.2,
        w_goal: float = 1.0,
        w_clear: float = 1.2,
        y_bias: float = -0.5,
        obstacle_cost: float = 1e9,
        # person stop
        person_stop_dist: float = 1.2,
        person_stop_y_width: float = 0.5,
        # window
        ahead_m: float = 2.0,
        half_width_m: float = 1.2,
        stride: int = 1,
        unknown_is_obstacle: bool = False,
        # speed
        kv: float = 0.6,  # reserved — currently unused (vx_cmd uses vx_fixed directly)
        kyaw: float = 1.0,
        v_max: float = 0.9,
        w_max: float = 0.75,
        v_min: float = 0.0,
        vx_fixed: float = 0.8,
        # motion
        safety_slowdown: bool = True,
        enable_turn_in_place: bool = True,
        theta_turn_deg: float = 40.0,
        allow_backward: bool = False,
    ) -> None:
        self._gnss = gnss_route_provider
        self._bev = bev_provider if bev_provider is not None else BEVOccupancyGridProvider()

        # loop
        if control_rate_hz is not None:
            self.timer_dt = 1.0 / max(1e-3, float(control_rate_hz))
        else:
            self.timer_dt = float(timer_dt)

        # distmap builder
        self.dist_method = (dist_method or "bfs_cuda").lower()
        self.dist_max_m = float(dist_max_m)
        self._dist_builder = DistMapBuilder(method=self.dist_method, max_dist=self.dist_max_m, use_cache=True)

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

        # ---- robot anchor in grid (legacy) ----
        # NOTE: BEV 쪽 dx=-0.34 보정을 이미 했다면, 여기 -0.34는 중복일 수 있음.
        self.robot_x_offset_m = -0.34
        self.robot_y_offset_m = 0.0

        # shared record
        self._lock = threading.Lock()
        self._latest: Optional[DwaRouteRecord] = None

        # thread control
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="DwaRouteCtrl")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._set_latest(DwaRouteRecord(
            t_monotonic=time.monotonic(),
            mode="STOP",
            stop_reason="provider_stop()",
            dist_method=self.dist_method,
        ))
        self._stop_evt.set()
        self._thread.join(timeout=5.0)
        if not self._thread.is_alive():
            self._thread = None

    # ---------------- outputs ----------------

    def get_record(self) -> Optional[DwaRouteRecord]:
        with self._lock:
            return self._latest

    def get(self) -> Optional[Dict[str, Any]]:
        rec = self.get_record()
        if rec is None:
            return None
        return {
            "t_monotonic": rec.t_monotonic,
            "dx_in": rec.dx_in,
            "dy_in": rec.dy_in,
            "heading_calibrated": rec.heading_calibrated,
            "reached_goal": rec.reached_goal,
            "dx_dwa": rec.dx_dwa,
            "dy_dwa": rec.dy_dwa,
            "best_clearance_m": rec.best_clearance_m,
            "vx_cmd": rec.vx_cmd,
            "vyaw_cmd": rec.vyaw_cmd,
            "mode": rec.mode,
            "stop_reason": rec.stop_reason,
            "dist_method": rec.dist_method,
            "occ_timestamp": rec.occ_timestamp,
        }

    def _set_latest(self, rec: DwaRouteRecord) -> None:
        with self._lock:
            self._latest = rec

    # ---------------- core loop ----------------

    def _run(self) -> None:
        logger.info(
            "DwaRouteProvider: started dt=%.3f dist_method=%s dist_max=%.2f",
            self.timer_dt, self.dist_method, self.dist_max_m
        )

        while not self._stop_evt.is_set():
            t0 = time.monotonic()

            # 1) GNSS goal
            g = self._gnss.get_record()
            if g is None:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    mode="IDLE",
                    stop_reason="no_gnss_record",
                    dist_method=self.dist_method,
                ))
                self._sleep(t0)
                continue

            dx_in = float(getattr(g, "dx", 0.0))
            dy_in = float(getattr(g, "dy", 0.0))
            heading_ok = bool(getattr(g, "heading_calibrated", False))
            reached = bool(getattr(g, "reached_goal", False))

            if reached:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=heading_ok,
                    reached_goal=True,
                    mode="STOP",
                    stop_reason="gnss_reached_goal",
                    dist_method=self.dist_method,
                ))
                self._sleep(t0)
                continue

            if not heading_ok:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=False,
                    reached_goal=False,
                    mode="STOP",
                    stop_reason="gnss_heading_not_calibrated",
                    dist_method=self.dist_method,
                ))
                self._sleep(t0)
                continue

            # 2) BEV occupancy
            bev = self._bev.data
            if not bev or "occupancy_grid" not in bev:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=True,
                    mode="IDLE",
                    stop_reason="no_bev_occupancy",
                    dist_method=self.dist_method,
                ))
                self._sleep(t0)
                continue

            occ = bev["occupancy_grid"]
            occ_ts = float(bev.get("timestamp", 0.0))

            try:
                res = float(occ["resolution"])
                W = int(occ["width"])
                H = int(occ["height"])
                x0 = float(occ["origin_x"])
                y0 = float(occ["origin_y"])
                grid = np.asarray(occ["data"], dtype=np.int8).reshape(H, W)
            except Exception:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=True,
                    mode="IDLE",
                    stop_reason="bad_occupancy_format",
                    dist_method=self.dist_method,
                    occ_timestamp=occ_ts,
                ))
                self._sleep(t0)
                continue

            # 3) distmap
            try:
                # builder 내부 캐시 사용(occ_ts 기준)
                dist = self._dist_builder.compute(grid, res, occ_ts)
            except Exception as e:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=True,
                    mode="IDLE",
                    stop_reason=f"no_distmap:{type(e).__name__}",
                    dist_method=self.dist_method,
                    occ_timestamp=occ_ts,
                ))
                self._sleep(t0)
                continue

            # 4) window indices (legacy)
            j0 = int(((self.robot_x_offset_m) - x0) / res)
            i0 = int(((self.robot_y_offset_m) - y0) / res)

            j_start = max(0, j0)
            j_end = min(W, j0 + int(self.ahead_m / res) + 1)
            i_start = max(0, i0 - int(self.half_width_m / res))
            i_end = min(H, i0 + int(self.half_width_m / res) + 1)

            if j_start >= j_end or i_start >= i_end:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=True,
                    mode="IDLE",
                    stop_reason="window_out_of_map",
                    dist_method=self.dist_method,
                    occ_timestamp=occ_ts,
                ))
                self._sleep(t0)
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
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=True,
                    mode="STOP",
                    stop_reason=f"person_occ88_within_{self.person_stop_dist:.2f}m",
                    dist_method=self.dist_method,
                    occ_timestamp=occ_ts,
                ))
                self._sleep(t0)
                continue

            # 6) cost minimization (legacy)
            best: Optional[Tuple[float, int, int, float, float, float]] = None
            m = max(1e-6, float(self.margin))

            for i in range(i_start, i_end, step):
                y = i * res + y0
                desired_y = dy_in + float(self.y_bias)
                base_y = (y - desired_y) ** 2

                for j in range(j_start, j_end, step):
                    occ_ij = int(grid[i, j])

                    x = j * res + x0
                    base = (x - dx_in) ** 2 + base_y

                    d = float(dist[i, j])
                    obs_soft = float(self.penalty) * (1.0 - d / m) ** 2 if d < m else 0.0
                    is_obstacle = occ_ij >= 100 or (self.unknown_is_obstacle and occ_ij < 0)
                    obs_hard = float(self.obstacle_cost) if is_obstacle else 0.0

                    cost = float(self.w_goal) * base + float(self.w_clear) * obs_soft + obs_hard

                    if best is None or cost < best[0]:
                        best = (cost, i, j, x, y, d)

            if best is None:
                self._set_latest(DwaRouteRecord(
                    t_monotonic=time.monotonic(),
                    dx_in=dx_in, dy_in=dy_in,
                    heading_calibrated=True,
                    mode="STOP",
                    stop_reason="no_cell_in_window",
                    dist_method=self.dist_method,
                    occ_timestamp=occ_ts,
                ))
                self._sleep(t0)
                continue

            _, bi, bj, bx, by, bd = best
            dx_dwa, dy_dwa = float(bx), float(by)

            # 7) speed generation
            theta = math.atan2(dy_dwa, dx_dwa)

            vyaw_cmd = max(-float(self.w_max), min(float(self.w_max), float(self.kyaw) * theta))

            vx_cmd = float(self.vx_fixed)

            if not self.allow_backward_target and dx_dwa < 0.0:
                vx_cmd = 0.0

            if self.enable_turn_in_place and abs(theta) > float(self.theta_turn):
                vx_cmd = 0.0

            if self.safety_slowdown and bd < m:
                scale = max(0.0, min(1.0, bd / m))
                vx_cmd *= scale

            vx_cmd = max(float(self.v_min), min(float(self.v_max), vx_cmd))

            self._set_latest(DwaRouteRecord(
                t_monotonic=time.monotonic(),
                dx_in=dx_in, dy_in=dy_in,
                heading_calibrated=True,
                reached_goal=False,
                dx_dwa=dx_dwa,
                dy_dwa=dy_dwa,
                best_clearance_m=float(bd),
                vx_cmd=float(vx_cmd),
                vyaw_cmd=float(vyaw_cmd),
                mode="DWA",
                stop_reason="none",
                dist_method=self.dist_method,
                occ_timestamp=occ_ts,
            ))

            self._sleep(t0)

        logger.info("DwaRouteProvider: stopped")

    def _sleep(self, t0: float) -> None:
        remain = self.timer_dt - (time.monotonic() - t0)
        if remain > 0:
            self._stop_evt.wait(remain)