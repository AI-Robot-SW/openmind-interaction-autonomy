# pointcloud_provider.py

import time
import logging
import threading
import numpy as np

from dataclasses import dataclass
from typing import Optional

from .singleton import singleton

from providers.utils.gpu_worker import GPUWorker
from providers.utils.kernels.pointcloud_kernel import PointCloudKernel
from providers.realsense_camera_provider import RealSenseCameraProvider, CameraFrame
from providers.segmentation_provider import SegmentationProvider


# BGR colormap: category index → (B, G, R)
_SEMANTIC_COLORS = np.array([
    [128, 128, 128],  # 0: unknown   — gray
    [  0, 255,   0],  # 1: driveable — green
    [255,   0,   0],  # 2: person    — blue (BGR)
    [  0,   0, 255],  # 3: avoid     — red
    [255, 255, 255],  # 4: curb      — white
], dtype=np.uint8)


@dataclass(frozen=True)
class PointCloudFrame:
    """
    t_monotonic    : 원본 CameraFrame의 t_monotonic (동기화 기준)
    points         : (N, 4) float32 — X, Y, Z, RGB-packed-as-float
    latency_s      : pointcloud 처리 시간 (초)
    pointcloud_fps : 1 / latency_s
    frame_cnt      : 대응하는 CameraFrame.frame_cnt — 동기화 기준
    """
    t_monotonic:    float
    points:         np.ndarray
    latency_s:      float
    pointcloud_fps: float
    frame_cnt:      int


@singleton
class PointCloudProvider:
    """
    background thread에서 realSense cameraFrame을 읽어 CUDA 커널로
    XYZ RGB point cloud를 생성하고 최신 결과를 data 프로퍼티로 노출.
    GPU 연산은 GPUWorker를 통해 실행
    """

    def __init__(
        self,
        range_max: Optional[float] = None,
        stride: int = 1
    ):
        self.range_max = range_max
        self.stride = int(stride)

        self.camera_provider = RealSenseCameraProvider()
        self.segmentation_provider = SegmentationProvider()

        self._gpu_worker: Optional[GPUWorker] = None
        self._kernel: Optional[PointCloudKernel] = None

        self._data: Optional[PointCloudFrame] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        self._last_cnt: int = -1

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.warning("PointCloudProvider already running")
            return

        self._gpu_worker = GPUWorker()
        self._kernel = self._gpu_worker.submit(lambda: PointCloudKernel()).result()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # 첫 프레임 도착까지 대기 — 반환 후 data가 항상 PointCloudFrame을 보장
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("PointCloudProvider: timed out waiting for first frame")

        logging.info("PointCloudProvider started")

    def stop(self) -> None:
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._kernel is not None and self._gpu_worker is not None:
            self._gpu_worker.submit(self._kernel.free).result()
            self._kernel = None

        self._last_cnt = -1
        with self._lock:
            self._data = None
            
        logging.info("PointCloudProvider stopped")


    def _process_frame(self, cam_frame: CameraFrame, seg_frame) -> PointCloudFrame:
        t0     = time.monotonic()
        stride = max(1, self.stride)
        depth  = cam_frame.depth[::stride, ::stride]
        color  = _SEMANTIC_COLORS[seg_frame.semantic_map][::stride, ::stride]
        intr   = cam_frame.intrinsics

        flat = self._gpu_worker.submit(
            lambda: self._kernel.run(depth, color, intr.fx, intr.fy, intr.cx, intr.cy, self.range_max)
        ).result()

        if self.range_max is not None:
            mask = (flat[:, 2] > 0.0) & (flat[:, 2] <= self.range_max) & np.isfinite(flat[:, 2])
        else:
            mask = (flat[:, 2] > 0.0) & np.isfinite(flat[:, 2])

        latency_s = float(time.monotonic() - t0)
        return PointCloudFrame(
            t_monotonic=float(cam_frame.t_monotonic),
            points=flat[mask],
            latency_s=latency_s,
            pointcloud_fps=1.0 / latency_s if latency_s > 0 else 0.0,
            frame_cnt=cam_frame.frame_cnt,
        )

    @property
    def data(self) -> Optional[PointCloudFrame]:
        """최신 PointCloudFrame. 첫 프레임 처리 전에는 None."""
        with self._lock:
            return self._data

    def _run(self) -> None:
        while self.running:
            try:
                cam_frame = self.camera_provider.data
                seg_frame = self.segmentation_provider.data
                if (
                    cam_frame is not None
                    and seg_frame is not None
                    and cam_frame.frame_cnt != self._last_cnt
                    and cam_frame.frame_cnt == seg_frame.frame_cnt
                ):
                    self._last_cnt = cam_frame.frame_cnt
                    with self._lock:
                        self._data = self._process_frame(cam_frame, seg_frame)
                else:
                    time.sleep(0.001)
            except Exception as e:
                logging.error(f"PointCloudProvider: run loop error: {e}")
                with self._lock:
                    self._data = None
