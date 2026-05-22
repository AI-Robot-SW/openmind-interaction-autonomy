# bev_occupancy_grid_provider.py

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
    data       : (height, width) int8 — -1=unknown, 0=free, 70=avoid, 80=curb, 88=person
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
    전체 파이프라인 (pc→grid, ground projection, ray cast, morphology) 이
    단일 BEVKernel 안에서 GPU로 처리된다.
    """

    def __init__(
        self,
        res: float = 0.05,
        width: int = 140,
        height: int = 120,
        origin_x: float = -0.5,
        origin_y: float = -3.0,
        dx: float = -0.34,
        dy: float = 0.0,
        closing_kernel_size: int = 3,
        camera_height_m: float = 0.413,
        ground_projection_stride: int = 4,
    ):
        self.res      = res
        self.width    = width
        self.height   = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.dx       = dx
        self.dy       = dy
        self.camera_height_m          = float(camera_height_m)
        self.ground_projection_stride = max(1, int(ground_projection_stride))

        self.pointcloud_provider = PointCloudProvider()

        self._gpu_worker:       Optional[GPUWorker]         = None
        self._pipeline_kernel:  Optional[BEVKernel] = None

        self._data: Optional[BEVFrame] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None

        # 매 프레임 사용되는 상수 사전 계산
        self._grid_shape = (self.height, self.width)
        self._inv_res    = 1.0 / self.res
        self._last_cnt:  int = -1
        self._last_frame_t: float = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.warning("BEVOccupancyGridProvider already running")
            return

        inv_res   = self._inv_res
        sensor_i  = int(np.clip(round((self.dy - self.origin_y) * inv_res), 0, self.height - 1))
        sensor_j  = int(np.clip(round((self.dx - self.origin_x) * inv_res), 0, self.width  - 1))

        self._gpu_worker = GPUWorker()
        self._pipeline_kernel = self._gpu_worker.submit(
            lambda: BEVKernel(
                width=self.width,
                height=self.height,
                origin_x=self.origin_x,
                origin_y=self.origin_y,
                dx=self.dx,
                dy=self.dy,
                inv_res=inv_res,
                camera_height_m=self.camera_height_m,
                ground_projection_stride=self.ground_projection_stride,
                sensor_i=sensor_i,
                sensor_j=sensor_j,
            )
        ).result()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # 첫 프레임 도착까지 대기
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

        if self._pipeline_kernel is not None:
            self._gpu_worker.submit(self._pipeline_kernel.free).result()
            self._pipeline_kernel = None

        if self._gpu_worker is not None:
            self._gpu_worker.stop()
            self._gpu_worker = None

        self._last_cnt = -1
        self._last_frame_t = 0.0
        with self._lock:
            self._data = None

        logging.info("BEVOccupancyGridProvider stopped")

    # ─────────────────────────────────────────────────────────────────────────

    def _process_frame(self, pc_frame) -> Optional[BEVFrame]:
        t0 = time.monotonic()

        occ = self._build_occupancy_grid(pc_frame)
        if occ is None:
            return None

        bev       = self._render_occupancy_grid_debug_image(occ.data)
        latency_s = float(time.monotonic() - t0)

        now = time.monotonic()
        bev_fps = 1.0 / (now - self._last_frame_t) if self._last_frame_t > 0.0 else 0.0
        self._last_frame_t = now

        return BEVFrame(
            t_monotonic=float(pc_frame.t_monotonic),
            bev_image=bev,
            occupancy_grid=occ,
            latency_s=latency_s,
            bev_fps=bev_fps,
            frame_cnt=pc_frame.frame_cnt,
        )

    def _build_occupancy_grid(self, pc_frame) -> Optional[OccupancyGrid]:
        try:
            points = pc_frame.points
            if points is None or len(points) == 0:
                grid_np = np.full(self._grid_shape, -1, dtype=np.int8)
                return OccupancyGrid(
                    resolution=self.res,
                    width=self.width,
                    height=self.height,
                    origin_x=self.origin_x,
                    origin_y=self.origin_y,
                    data=grid_np,
                )

            _pts = np.ascontiguousarray(points, dtype=np.float32)
            _sem = pc_frame.semantic_map
            _fx  = float(pc_frame.fx)
            _fy  = float(pc_frame.fy)
            _cx  = float(pc_frame.cx)
            _cy  = float(pc_frame.cy)

            grid_np = self._gpu_worker.submit(
                lambda: self._pipeline_kernel.run(_pts, _sem, _fx, _fy, _cx, _cy)
            ).result()

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

    def _render_occupancy_grid_debug_image(self, grid_np: np.ndarray) -> np.ndarray:
        display_grid = np.flipud(np.fliplr(grid_np.T))

        image = np.zeros(
            (display_grid.shape[0], display_grid.shape[1], 3),
            dtype=np.uint8,
        )
        image[display_grid == -1] = ( 40,  40,  40)   # unknown: dark gray
        image[display_grid ==  0] = (  0, 255,   0)   # free: green
        image[display_grid == 70] = (  0,   0, 255)   # avoid: red
        image[display_grid == 80] = (  0, 165, 255)   # curb: orange
        image[display_grid == 88] = (255,   0,   0)   # person: blue

        return image

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def data(self) -> Optional[BEVFrame]:
        """최신 BEVFrame. 첫 프레임 처리 전에는 None."""
        with self._lock:
            return self._data

    def _run(self) -> None:
        while self.running:
            try:
                signaled = self.pointcloud_provider.frame_event.wait(timeout=0.1)
                if not signaled:
                    continue
                self.pointcloud_provider.frame_event.clear()
                pc_frame = self.pointcloud_provider.data
                if pc_frame is not None and pc_frame.frame_cnt != self._last_cnt:
                    self._last_cnt = pc_frame.frame_cnt
                    frame = self._process_frame(pc_frame)
                    with self._lock:
                        self._data = frame
            except Exception as e:
                logging.error(f"BEVOccupancyGridProvider: run loop error: {e}")
                with self._lock:
                    self._data = None
