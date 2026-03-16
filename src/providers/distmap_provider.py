# distmap_provider.py

import time
import logging
import threading
import numpy as np

from dataclasses import dataclass
from typing import Optional

from .singleton import singleton

from providers.utils.gpu_worker import GPUWorker
from providers.utils.kernels.distmap_kernel import DistMapKernel
from .bev_occupancy_grid_provider import BEVOccupancyGridProvider


@dataclass(frozen=True)
class DistMapFrame:
    """
    t_monotonic : 원본 BEVFrame의 t_monotonic (동기화 기준)
    dist        : (H, W) float32 — 장애물까지 거리 (미터, [0, max_dist])
    latency_s   : distmap 처리 시간 (초)
    distmap_fps : 1 / latency_s
    frame_cnt   : 대응하는 BEVFrame.frame_cnt — 동기화 기준
    """
    t_monotonic: float
    dist:        np.ndarray
    latency_s:    float
    distmap_fps:  float
    frame_cnt:    int


@singleton
class DistMapProvider:
    """
    background thread에서 BEVFrame을 읽어 CUDA 커널로
    장애물 distmap을 생성하고 최신 결과를 data 프로퍼티로 노출.
    GPU 연산은 GPUWorker를 통해 실행.
    """

    def __init__(
        self,
        max_dist: float = 5.0,
    ) -> None:
        self.max_dist = float(max_dist)

        self.bev_provider = BEVOccupancyGridProvider()

        self._gpu_worker: Optional[GPUWorker] = None
        self._kernel: Optional[DistMapKernel] = None

        self._data: Optional[DistMapFrame] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None

        self._last_cnt: int = -1

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.warning("DistMapProvider already running")
            return

        self._gpu_worker = GPUWorker()
        bev = self.bev_provider
        self._kernel = self._gpu_worker.submit(lambda: DistMapKernel(bev.width, bev.height, self.max_dist, bev.res)).result()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # 첫 프레임 도착까지 대기 — 반환 후 data가 항상 DistmapFrame을 보장
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("DistMapProvider: timed out waiting for first frame")

        logging.info("DistMapProvider started")

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

        logging.info("DistMapProvider stopped")

    @property
    def data(self) -> Optional[DistMapFrame]:
        """최신 DistMapFrame. 첫 프레임 처리 전에는 None."""
        with self._lock:
            return self._data

    def _process_frame(self, bev_frame) -> DistMapFrame:
        t0 = time.monotonic()

        dist = self._gpu_worker.submit(
            lambda: self._kernel.run(bev_frame.occupancy_grid.data)
        ).result()

        latency_s = float(time.monotonic() - t0)
        return DistMapFrame(
            t_monotonic=float(bev_frame.t_monotonic),
            dist=dist,
            latency_s=latency_s,
            distmap_fps=1.0 / latency_s if latency_s > 0.0 else 0.0,
            frame_cnt=bev_frame.frame_cnt,
        )


    def _run(self) -> None:
        while self.running:
            try:
                bev_frame = self.bev_provider.data
                if (
                    bev_frame is not None
                    and bev_frame.frame_cnt != self._last_cnt
                ):
                    self._last_cnt = bev_frame.frame_cnt
                    frame = self._process_frame(bev_frame)
                    with self._lock:
                        self._data = frame
                else:
                    time.sleep(0.001)
            except Exception as e:
                logging.error(f"DistMapProvider: frame processing failed: {e}")
                with self._lock:
                    self._data = None