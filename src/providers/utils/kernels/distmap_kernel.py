# distmap_kernel.py

"""
DistMapKernel — BEV 점유 격자 (H×W int8) → 장애물까지 거리맵 (H×W float32) CUDA 커널 래퍼.

GPUWorker를 통해 GPU thread 안에서만 인스턴스화·호출해야 한다.

    kernel = gpu_worker.submit(lambda: DistMapKernel(width, height, max_dist, res)).result()
    dist   = gpu_worker.submit(lambda: kernel.run(grid_int8)).result()
    gpu_worker.submit(kernel.free).result()

알고리즘:
  - BFS wave propagation을 ceil(max_dist / res)회 고정 반복
  - 수렴 체크 없이 커널만 반복 실행 → CPU-GPU 동기화는 마지막 D2H 한 번뿐
  - 장애물(grid > 0) 셀은 거리 0, 자유 셀은 max_dist로 초기화 후 전파
"""

import math
import logging
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_KERNEL_PATH = Path(__file__).resolve().parent / "distmap.cu"

_BLOCK = (16, 16, 1)


class DistMapKernel:
    """
    distmap.cu 커널의 Python 래퍼.
    컴파일·메모리 할당·실행·해제를 캡슐화한다.
    모든 메서드는 GPU thread(GPUWorker) 안에서만 호출해야 한다.
    """

    def __init__(self, width: int, height: int, max_dist: float, res: float) -> None:
        mod = SourceModule(_KERNEL_PATH.read_text())
        self._fn = mod.get_function("distmap_bfs")

        self._width = width
        self._height = height
        self._max_dist = float(max_dist)
        self._res = float(res)

        # 수렴 보장을 위한 반복 횟수: max_dist까지 전파하려면 최대 ceil(max_dist/res)회
        self._max_iter = int(math.ceil(max_dist / res))

        # 고정 크기 GPU 버퍼 1회 할당
        n_bytes = width * height * np.dtype(np.float32).itemsize
        self._gpu_obstacle: cuda.DeviceAllocation = cuda.mem_alloc(n_bytes)
        self._gpu_dist: cuda.DeviceAllocation = cuda.mem_alloc(n_bytes)

        self._grid_dim = ((width + 15) // 16, (height + 15) // 16)

        logging.info("DistMapKernel: CUDA kernel compiled")

    def run(self, grid: np.ndarray) -> np.ndarray:
        """
        BEV 점유 격자 → 장애물까지 거리맵.

        Parameters
        ----------
        grid : np.ndarray
            (H, W) int8 — BEVOccupancyGridProvider.data.occupancy_grid.data.
            양수(> 0) 셀이 장애물.

        Returns
        -------
        np.ndarray
            (H, W) float32 — 미터 단위 거리, [0, max_dist]로 클리핑.

        GPU thread 안에서만 호출.
        """
        obstacle = (grid > 0).astype(np.float32)
        dist_init = np.where(obstacle > 0, 0.0, self._max_dist).astype(np.float32)

        cuda.memcpy_htod(self._gpu_obstacle, obstacle)
        cuda.memcpy_htod(self._gpu_dist, dist_init)

        # 고정 횟수 반복 — 커널 launch는 비동기, CPU는 즉시 다음 launch로 진행
        for _ in range(self._max_iter):
            self._fn(
                self._gpu_dist,
                self._gpu_obstacle,
                np.int32(self._width),
                np.int32(self._height),
                np.float32(self._res),
                np.float32(self._max_dist),
                block=_BLOCK,
                grid=self._grid_dim,
            )

        # 여기서만 동기화 (GPU 완료 대기)
        out = np.empty(self._height * self._width, dtype=np.float32)
        cuda.memcpy_dtoh(out, self._gpu_dist)
        return out.reshape(self._height, self._width)

    def free(self) -> None:
        """GPU 메모리 해제. GPU thread 안에서만 호출."""
        for attr in ("_gpu_obstacle", "_gpu_dist"):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, attr, None)
