# bev_occupancy_grid_provider.py

import cv2
import time
import logging
import threading
import numpy as np

from dataclasses import dataclass
from typing import Optional

from .singleton import singleton

from providers.utils.gpu_worker import GPUWorker
from providers.utils.kernels.bev_kernel import BEVKernel
from .pointcloud_provider import PointCloudProvider


@dataclass(frozen=True)
class OccupancyGrid:
    """
    resolution : grid 해상도 (meters/pixel)
    width      : (pixels)
    height     : (pixels)
    origin_x   : (meters)
    origin_y   : (meters)
    data       : (height, width) int8 — 0=free, 70=avoid, 88=person, 100=occupied/unknown
    """
    resolution: float
    width:      int
    height:     int
    origin_x:   float
    origin_y:   float
    data:       np.ndarray


@dataclass(frozen=True)
class BEVFrame:
    """
    t_monotonic    : 원본 PointCloudFrame의 t_monotonic (동기화 기준)
    bev_image      : (height, width, 3) uint8 BGR
    occupancy_grid : OccupancyGrid
    latency_s      : BEV 처리 시간 (초)
    bev_fps        : 1 / latency_s
    frame_cnt      : 대응하는 PointCloudFrame.frame_cnt — 동기화 기준
    """
    t_monotonic:    float
    bev_image:      np.ndarray
    occupancy_grid: OccupancyGrid
    latency_s:      float
    bev_fps:        float
    frame_cnt:      int


@singleton
class BEVOccupancyGridProvider:
    """
    background thread에서 PointCloudFrame을 읽어 CUDA 커널로
    BEV 이미지와 occupancy grid를 생성하고 최신 결과를 data 프로퍼티로 노출.
    GPU 연산은 GPUWorker를 통해 실행
    """

    def __init__(
        self,
        res: float = 0.05,
        width: int = 50,
        height: int = 60,
        origin_x: float = 0.0,
        origin_y: float = -1.5,
        dx: float = -0.34,
        dy: float = 0.0,
        closing_kernel_size: int = 1,
    ):
        self.res = res
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.dx = dx
        self.dy = dy
        self.closing_kernel_size = closing_kernel_size

        self.pointcloud_provider = PointCloudProvider()

        self._gpu_worker: Optional[GPUWorker] = None
        self._kernel: Optional[BEVKernel] = None

        self._data: Optional[BEVFrame] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None

        # 매 프레임 사용되는 상수 사전 계산
        self._grid_shape = (self.height, self.width)
        self._inv_res = 1.0 / self.res
        self._grid_x_offset = self.dx - self.origin_x
        self._grid_y_offset = self.dy - self.origin_y
        self._apply_closing = self.closing_kernel_size > 1
        self._closing_kernel = (
            np.ones((closing_kernel_size, closing_kernel_size), dtype=np.uint8)
            if self._apply_closing
            else None
        )

        self._last_cnt: int = -1

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.warning("BEVOccupancyGridProvider already running")
            return

        self._gpu_worker = GPUWorker()
        self._kernel = self._gpu_worker.submit(lambda: BEVKernel(self.width, self.height)).result()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # 첫 프레임 도착까지 대기 — 반환 후 data가 항상 BEVFrame을 보장
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("BEVOccupancyGridProvider: timed out waiting for first frame")

        logging.info("BEVOccupancyGridProvider started")

    def stop(self) -> None:
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._kernel is not None:
            self._gpu_worker.submit(self._kernel.free).result()
            self._kernel = None

        self._last_cnt = -1
        with self._lock:
            self._data = None

        logging.info("BEVOccupancyGridProvider stopped")

    def _process_frame(self, pc_frame) -> Optional[BEVFrame]:
        # BEV 이미지: GPU thread 실행 / occupancy grid: CPU numpy
        t0 = time.monotonic()
        
        bev = self._gpu_worker.submit(
            lambda: self._kernel.run(pc_frame.points, self.res)
        ).result()

        occ = self._build_occupancy_grid(pc_frame.points)

        latency_s = float(time.monotonic() - t0)
        return BEVFrame(
            t_monotonic=float(pc_frame.t_monotonic),
            bev_image=bev,
            occupancy_grid=occ,
            latency_s=latency_s,
            bev_fps=1.0 / latency_s if latency_s > 0.0 else 0.0,
            frame_cnt=pc_frame.frame_cnt,
        )

    def _build_occupancy_grid(self, points: np.ndarray) -> Optional[OccupancyGrid]:
        # points (N, 4) float32 → OccupancyGrid (nav_msgs/OccupancyGrid 호환)
        try:
            x = points[:, 0]
            z = points[:, 2]
            rgb_int = points[:, 3].view(np.uint32)
            r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
            g = ((rgb_int >>  8) & 0xFF).astype(np.uint8)
            b = ( rgb_int        & 0xFF).astype(np.uint8)

            grid_np = np.full(self._grid_shape, 100, dtype=np.int8)

            # 좌표 변환: 카메라 x/z → 그리드 행/열 인덱스
            j_grid = ((z + self._grid_x_offset) * self._inv_res).astype(np.int32)
            i_grid = ((-x + self._grid_y_offset) * self._inv_res).astype(np.int32)

            valid = (
                (i_grid >= 0)
                & (i_grid < self.height)
                & (j_grid >= 0)
                & (j_grid < self.width)
            )

            if np.any(valid):
                valid_i = i_grid[valid]
                valid_j = j_grid[valid]
                valid_r = r[valid]
                valid_g = g[valid]
                valid_b = b[valid]

                mask_person = (valid_b > 100) & (valid_r < 80) & (valid_g < 80)
                mask_avoid  = (valid_r > 200) & (valid_g > 200) & (valid_b > 200)
                mask_free   = (valid_g > 100) & (valid_r < 80)  & (valid_b < 80)

                grid_np[valid_i[mask_person], valid_j[mask_person]] = 88
                grid_np[valid_i[mask_avoid],  valid_j[mask_avoid]]  = 70
                grid_np[valid_i[mask_free],   valid_j[mask_free]]   = 0

            if self._apply_closing and self._closing_kernel is not None:
                occ_mask = (grid_np == 100).astype(np.uint8)
                occ_mask = cv2.morphologyEx(
                    occ_mask,
                    cv2.MORPH_CLOSE,
                    self._closing_kernel,
                    iterations=3,
                )
                grid_np[occ_mask > 0] = 100

            return OccupancyGrid(
                resolution=self.res,
                width=self.width,
                height=self.height,
                origin_x=self.origin_x,
                origin_y=self.origin_y,
                data=grid_np,
            )
        except Exception as e:
            logging.error(f"BEVOccupancyGridProvider: occupancy grid build failed: {e}")
            return None


    @property
    def data(self) -> Optional[BEVFrame]:
        """최신 BEVFrame. 첫 프레임 처리 전에는 None."""
        with self._lock:
            return self._data
        
    def _run(self) -> None:
        while self.running:
            try:
                pc_frame = self.pointcloud_provider.data
                if (
                    pc_frame is not None
                    and pc_frame.frame_cnt != self._last_cnt
                ):
                    self._last_cnt = pc_frame.frame_cnt
                    frame = self._process_frame(pc_frame)
                    with self._lock:
                        self._data = frame
                else:
                    time.sleep(0.001)
            except Exception as e:
                logging.error(f"BEVOccupancyGridProvider: run loop error: {e}")
                with self._lock:
                    self._data = None