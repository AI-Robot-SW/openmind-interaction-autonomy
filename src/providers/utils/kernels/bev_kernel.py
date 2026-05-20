# bev_kernel.py

"""
BEVKernel — BEV occupancy grid 전체 파이프라인을 GPU에서 실행한다.

입력:
    points       : (N, 4) float32  — X_right, Y, Z_fwd, RGB_packed (AoS)
    semantic_map : (H_cam, W_cam) uint8 — segmentation class map

출력:
    (H_grid, W_grid) int8  — -1=unknown, 0=free, 70=avoid/curb, 88=person

파이프라인 (GPU 전용, Python 쪽에 CPU 연산 없음):
    1.  pc_to_grids       pointcloud → free_ep / avoid / curb / person 그리드
    2.  make_blocked      avoid|curb|person → blocked
    3.  blocker_bottom    semantic_map → 열별 장애물 하단 행
    4.  ground_proj       semantic_map + blocker_bottom → ground_free
    5.  dilate_3x3        ground_free → ground_dil
    6.  max_grids         max(free_ep, ground_dil) → free_cand
    7.  raycast           Bresenham (한 thread = 한 grid cell)
    8.  dilate_3x3        ray_grid → temp
    9.  dilate_5x5        temp → ray_grid       ─┐ morphological
    10. erode_5x5         ray_grid → temp       ─┘ close(5×5)
    11. merge_final       temp + obstacles → grid_np int8

GPUWorker를 통해 GPU thread 안에서만 인스턴스화·호출해야 한다.
"""

import logging
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_KERNEL_PATH = Path(__file__).resolve().parent / "bev.cu"

_T1D = 256       # 1-D kernel thread count
_T2D = 16        # 2-D kernel thread count (T2D × T2D per block)

# 사전 할당 상한 — 프레임마다 재할당 없음
_MAX_POINTS = 200_000
_MAX_SEM_H  = 480
_MAX_SEM_W  = 640


class BEVKernel:
    """
    bev.cu 의 전체 파이프라인 래퍼.
    GPU 버퍼는 __init__ 에서 최대 크기로 한 번만 할당된다.
    모든 메서드는 GPU thread (GPUWorker) 안에서만 호출해야 한다.
    """

    def __init__(
        self,
        width: int,
        height: int,
        origin_x: float,
        origin_y: float,
        dx: float,
        dy: float,
        inv_res: float,
        camera_height_m: float,
        ground_projection_stride: int,
        sensor_i: int,
        sensor_j: int,
        occlusion_margin: int = 4,
        blocker_dilation_radius: int = 2,
        ray_gap_max_cells: int = 50,
        ray_gap_min_free_after: int = 5,
    ) -> None:
        src = _KERNEL_PATH.read_text()
        mod = SourceModule(src)

        self._fn_pc_to_grids    = mod.get_function("pc_to_grids_kernel")
        self._fn_make_blocked   = mod.get_function("make_blocked_kernel")
        self._fn_blocker_bottom = mod.get_function("blocker_bottom_kernel")
        self._fn_ground_proj    = mod.get_function("ground_proj_kernel")
        self._fn_dilate3        = mod.get_function("dilate_3x3_kernel")
        self._fn_dilate5        = mod.get_function("dilate_5x5_kernel")
        self._fn_erode5         = mod.get_function("erode_5x5_kernel")
        self._fn_max_grids      = mod.get_function("max_grids_kernel")
        self._fn_raycast        = mod.get_function("raycast_grid_kernel")
        self._fn_merge_final    = mod.get_function("merge_final_kernel")
        self._fn_ray_gap_fill   = mod.get_function("ray_gap_fill_final_kernel")

        self._W      = width
        self._H      = height
        self._N_grid = width * height

        # 스칼라 파라미터 — np.float32/int32 로 미리 박싱
        self._inv_res          = np.float32(inv_res)
        self._dx_minus_ox      = np.float32(dx - origin_x)
        self._dy_minus_oy      = np.float32(dy - origin_y)
        self._dx               = np.float32(dx)
        self._dy               = np.float32(dy)
        self._origin_x         = np.float32(origin_x)
        self._origin_y         = np.float32(origin_y)
        self._camera_height_m  = np.float32(camera_height_m)
        self._proj_stride      = np.int32(ground_projection_stride)
        self._sensor_i         = np.int32(sensor_i)
        self._sensor_j         = np.int32(sensor_j)
        self._occlusion_margin = np.int32(occlusion_margin)
        self._blocker_dil_rad       = np.int32(blocker_dilation_radius)
        self._ray_gap_max_i32       = np.int32(ray_gap_max_cells)
        self._ray_gap_min_free_i32  = np.int32(ray_gap_min_free_after)
        self._W_i32            = np.int32(width)
        self._H_i32            = np.int32(height)
        self._N_grid_i32       = np.int32(self._N_grid)

        # ── GPU 버퍼 (1회 할당) ───────────────────────────────────────────
        g = self._N_grid
        self._d_points       = cuda.mem_alloc(_MAX_POINTS * 4 * 4)   # float32
        self._d_sem_map      = cuda.mem_alloc(_MAX_SEM_H * _MAX_SEM_W)
        self._d_blocker_bot  = cuda.mem_alloc(_MAX_SEM_W * 4)         # int32

        self._d_free_ep      = cuda.mem_alloc(g)
        self._d_avoid        = cuda.mem_alloc(g)
        self._d_curb         = cuda.mem_alloc(g)
        self._d_person       = cuda.mem_alloc(g)
        self._d_blocked      = cuda.mem_alloc(g)
        self._d_ground_free  = cuda.mem_alloc(g)
        self._d_ground_dil   = cuda.mem_alloc(g)
        self._d_free_cand    = cuda.mem_alloc(g)
        self._d_ray_grid     = cuda.mem_alloc(g)
        self._d_temp         = cuda.mem_alloc(g)
        self._d_grid_np      = cuda.mem_alloc(g)      # int8

        # ── Pinned host 버퍼 (DMA 직전송) ────────────────────────────────
        self._h_points  = cuda.pagelocked_empty(_MAX_POINTS * 4, dtype=np.float32)
        self._h_sem_map = cuda.pagelocked_empty(_MAX_SEM_H * _MAX_SEM_W, dtype=np.uint8)
        self._h_grid_np = cuda.pagelocked_empty(self._N_grid, dtype=np.int8)

        logging.info(
            "BEVKernel: compiled, GPU buffers allocated "
            "(grid=%dx%d, max_points=%d, max_sem=%dx%d)",
            width, height, _MAX_POINTS, _MAX_SEM_H, _MAX_SEM_W,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        points: np.ndarray,        # (N, 4) float32
        semantic_map: np.ndarray,  # (H_cam, W_cam) uint8
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """
        전체 파이프라인 실행 후 (H_grid, W_grid) int8 를 반환.
        GPU thread 안에서만 호출.
        """
        N = min(len(points), _MAX_POINTS)
        H_cam, W_cam = semantic_map.shape
        assert H_cam <= _MAX_SEM_H and W_cam <= _MAX_SEM_W, (
            f"semantic_map {semantic_map.shape} exceeds max ({_MAX_SEM_H},{_MAX_SEM_W})"
        )

        W, H, N_grid = self._W, self._H, self._N_grid

        # 서브샘플 그리드 크기 (ground projection 용)
        stride   = int(self._proj_stride)
        W_samp   = (W_cam + stride - 1) // stride
        H_samp   = (H_cam + stride - 1) // stride

        # ── H2D 전송 ──────────────────────────────────────────────────────
        n_f = N * 4
        self._h_points[:n_f] = points[:N].ravel()
        cuda.memcpy_htod(self._d_points,  self._h_points[:n_f])

        sem_n = H_cam * W_cam
        self._h_sem_map[:sem_n] = semantic_map.ravel()
        cuda.memcpy_htod(self._d_sem_map, self._h_sem_map[:sem_n])

        # ── 프레임별 GPU 버퍼 초기화 ──────────────────────────────────────
        for buf in (
            self._d_free_ep, self._d_avoid, self._d_curb, self._d_person,
            self._d_blocked, self._d_ground_free, self._d_ground_dil,
            self._d_free_cand, self._d_ray_grid, self._d_temp,
        ):
            cuda.memset_d8(buf, 0, N_grid)
        # blocker_bottom_by_col 을 -1 로 초기화 (0xFFFFFFFF = int32 -1)
        cuda.memset_d32(self._d_blocker_bot, 0xFFFFFFFF, W_cam)

        # ── 공통 grid 블록 설정 ────────────────────────────────────────────
        bx_g = (W + _T2D - 1) // _T2D
        by_g = (H + _T2D - 1) // _T2D
        b2d  = (_T2D, _T2D, 1)
        b1d  = (_T1D, 1, 1)
        bg   = ((N_grid + _T1D - 1) // _T1D, 1, 1)

        # ── 1. pointcloud → {free_ep, avoid, curb, person} ───────────────
        self._fn_pc_to_grids(
            self._d_points, np.int32(N),
            self._inv_res, self._dx_minus_ox, self._dy_minus_oy,
            self._W_i32, self._H_i32,
            self._d_free_ep, self._d_avoid, self._d_curb, self._d_person,
            block=b1d,
            grid=((N + _T1D - 1) // _T1D, 1, 1),
        )

        # ── 2. blocked = avoid | curb | person ───────────────────────────
        self._fn_make_blocked(
            self._d_avoid, self._d_curb, self._d_person, self._d_blocked,
            self._N_grid_i32,
            block=b1d, grid=bg,
        )

        # ── 3. 열별 장애물 하단 행 ─────────────────────────────────────────
        bx_s = (W_cam + _T2D - 1) // _T2D
        by_s = (H_cam + _T2D - 1) // _T2D
        self._fn_blocker_bottom(
            self._d_sem_map, np.int32(H_cam), np.int32(W_cam),
            self._blocker_dil_rad, self._d_blocker_bot,
            block=b2d, grid=(bx_s, by_s, 1),
        )

        # ── 4. Ground projection ──────────────────────────────────────────
        bx_p = (W_samp + _T2D - 1) // _T2D
        by_p = (H_samp + _T2D - 1) // _T2D
        self._fn_ground_proj(
            self._d_sem_map, self._d_blocker_bot,
            np.int32(H_cam), np.int32(W_cam),
            np.float32(fx), np.float32(fy),
            np.float32(cx), np.float32(cy),
            self._camera_height_m,
            self._dx, self._dy,
            self._origin_x, self._origin_y,
            self._inv_res,
            self._W_i32, self._H_i32,
            self._proj_stride, self._occlusion_margin,
            self._d_ground_free,
            block=b2d, grid=(bx_p, by_p, 1),
        )

        # ── 5. dilate_3x3(ground_free → ground_dil) ──────────────────────
        self._fn_dilate3(
            self._d_ground_free, self._d_ground_dil,
            self._W_i32, self._H_i32,
            block=b2d, grid=(bx_g, by_g, 1),
        )

        # ── 6. free_cand = max(free_ep, ground_dil) ──────────────────────
        self._fn_max_grids(
            self._d_free_ep, self._d_ground_dil, self._d_free_cand,
            self._N_grid_i32,
            block=b1d, grid=bg,
        )

        # ── 7. Raycast ────────────────────────────────────────────────────
        self._fn_raycast(
            self._d_free_cand, self._d_blocked, self._d_ray_grid,
            self._sensor_i, self._sensor_j,
            self._W_i32, self._H_i32,
            block=b2d, grid=(bx_g, by_g, 1),
        )

        # ── 8–10. Morphological close(5×5): dilate3 → dilate5 → erode5 ──
        # step 8: dilate_3x3(ray_grid → temp)
        self._fn_dilate3(
            self._d_ray_grid, self._d_temp,
            self._W_i32, self._H_i32,
            block=b2d, grid=(bx_g, by_g, 1),
        )
        # step 9: dilate_5x5(temp → ray_grid)   [ray_grid 재활용]
        self._fn_dilate5(
            self._d_temp, self._d_ray_grid,
            self._W_i32, self._H_i32,
            block=b2d, grid=(bx_g, by_g, 1),
        )
        # step 10: erode_5x5(ray_grid → temp)   [temp = closed free_grid]
        self._fn_erode5(
            self._d_ray_grid, self._d_temp,
            self._W_i32, self._H_i32,
            block=b2d, grid=(bx_g, by_g, 1),
        )

        # ── 11. 최종 그리드 조립 ──────────────────────────────────────────
        # obstacle 이 free 를 덮어쓰는 우선순위는 merge_final 내부에서 처리
        self._fn_merge_final(
            self._d_temp,
            self._d_avoid, self._d_curb, self._d_person,
            self._d_grid_np, self._N_grid_i32,
            block=b1d, grid=bg,
        )

        # ── 12. Ray gap fill — 카메라 시점 기반 노이즈 채우기 ────────────────
        # sensor→경계 방향 Bresenham 광선마다 thread 1개
        # free(0)→avoid/curb(70, ≤max_gap)→free(0, ≥min_free_after) 패턴 감지 시 채움
        n_boundary = 2 * (self._W + self._H - 2)
        b_gap = (min(256, n_boundary), 1, 1)
        g_gap = ((n_boundary + b_gap[0] - 1) // b_gap[0], 1, 1)
        self._fn_ray_gap_fill(
            self._d_grid_np,
            self._sensor_i, self._sensor_j,
            self._W_i32, self._H_i32,
            self._ray_gap_max_i32, self._ray_gap_min_free_i32,
            block=b_gap, grid=g_gap,
        )

        # ── D2H: 결과만 전송 ──────────────────────────────────────────────
        cuda.memcpy_dtoh(self._h_grid_np, self._d_grid_np)
        return self._h_grid_np.reshape(H, W).copy()

    # ─────────────────────────────────────────────────────────────────────────
    def free(self) -> None:
        """GPU 메모리 해제. GPU thread 안에서만 호출."""
        _bufs = [
            "_d_points", "_d_sem_map", "_d_blocker_bot",
            "_d_free_ep", "_d_avoid", "_d_curb", "_d_person",
            "_d_blocked", "_d_ground_free", "_d_ground_dil",
            "_d_free_cand", "_d_ray_grid", "_d_temp", "_d_grid_np",
        ]
        for attr in _bufs:
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, attr, None)
