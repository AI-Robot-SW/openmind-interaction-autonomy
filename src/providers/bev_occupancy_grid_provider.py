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
from .pointcloud_provider import PointCloudProvider, _SEMANTIC_COLORS


def _pack_rgb(bgr: np.ndarray) -> np.uint32:
    """BGR uint8 배열 → CUDA 커널 출력 포맷 (R<<16 | G<<8 | B) uint32."""
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return np.uint32((r << 16) | (g << 8) | b)


# pointcloud 내 색상 — _SEMANTIC_COLORS(BGR) 기준으로 사전 계산
_COLOR_UNKNOWN   = _pack_rgb(_SEMANTIC_COLORS[0])  # gray
_COLOR_DRIVEABLE = _pack_rgb(_SEMANTIC_COLORS[1])  # green
_COLOR_PERSON    = _pack_rgb(_SEMANTIC_COLORS[2])  # blue (BGR) → B channel dominant
_COLOR_AVOID     = _pack_rgb(_SEMANTIC_COLORS[3])  # red (BGR)  → R channel dominant
_COLOR_CURB      = _pack_rgb(_SEMANTIC_COLORS[4])  # white

@dataclass(frozen=True)
class OccupancyGrid:
    """
    resolution : grid 해상도 (meters/pixel)
    width      : (pixels)
    height     : (pixels)
    origin_z   : (meters)
    origin_x  : (meters)
    data       : (height, width) int8 — 0=free, 70=avoid, 88=person, 100=occupied/unknown
    """
    resolution: float
    width:      int
    height:     int
    origin_z:   float
    origin_x:  float
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
        res: float = 0.05,              # cell 1칸이 나타내는 실제 거리 (m/cell)
        width: int = 60,                # X(좌우) 방향 grid cell 수, 실제 커버 길이는 res * width  (m)
        height: int = 50,               # Z(전방) 방향 grid cell 수, 실제 커버 길이는 res * height (m)
        origin_z: float = 0.0,          # 행 인덱스 0에 대응하는 Z 좌표 기준값 (m)
        origin_x: float = -1.5,         # 열 인덱스 0에 대응하는 X 좌표 기준값 (m)
        dz: float = -0.34,              # 카메라 마운트 위치 보정 — Z(전방) 방향 오프셋 (m)
        dx: float = 0.0,                # 카메라 마운트 위치 보정 — X(좌우) 방향 오프셋 (m)
        closing_kernel_size: int = 1,
    ):
        self.res = res
        self.width = width
        self.height = height
        self.origin_z = origin_z
        self.origin_x = origin_x
        self.dz = dz
        self.dx = dx
        self.closing_kernel_size = closing_kernel_size

        self.pointcloud_provider = PointCloudProvider()

        self._gpu_worker: Optional[GPUWorker] = None
        self._kernel: Optional[BEVKernel] = None

        self._data: Optional[BEVFrame] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None

        # 매 프레임 사용되는 상수 사전 계산
        self._inv_res = 1.0 / self.res
        self._offset_z = self.dz - self.origin_z
        self._offset_x = self.dx - self.origin_x
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
        # points (N, 4) float32 → OccupancyGrid
        try:
            x       = points[:, 0]
            z       = points[:, 2]
            rgb_int = points[:, 3].view(np.uint32)

            grid_np = np.full((self.height, self.width), 100, dtype=np.int8)

            # 카메라 좌표계 기준, X 우측 / Z 전방 / Y 하단
            # row(i) : i 증가는 Z 증가를 의미 / col(j) : j 증가는 X 증가를 의미
            i_grid = ((z + self._offset_z) * self._inv_res).astype(np.int32)
            j_grid = ((x + self._offset_x) * self._inv_res).astype(np.int32)

            valid = (
                (i_grid >= 0)
                & (i_grid < self.height)
                & (j_grid >= 0)
                & (j_grid < self.width)
            )

            if np.any(valid):
                valid_i   = i_grid[valid]
                valid_j   = j_grid[valid]
                valid_rgb = rgb_int[valid]

                mask_unknown   = (valid_rgb == _COLOR_UNKNOWN)
                mask_driveable = (valid_rgb == _COLOR_DRIVEABLE)
                mask_person    = (valid_rgb == _COLOR_PERSON)
                mask_avoid     = (valid_rgb == _COLOR_AVOID)
                mask_curb      = (valid_rgb == _COLOR_CURB)

                grid_np[valid_i[mask_unknown],   valid_j[mask_unknown]]   = 100
                grid_np[valid_i[mask_driveable], valid_j[mask_driveable]] = 0
                grid_np[valid_i[mask_person],    valid_j[mask_person]]    = 88
                grid_np[valid_i[mask_avoid],     valid_j[mask_avoid]]     = 100
                grid_np[valid_i[mask_curb],      valid_j[mask_curb]]      = 70

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
                origin_z=self.origin_z,
                origin_x=self.origin_x,
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
