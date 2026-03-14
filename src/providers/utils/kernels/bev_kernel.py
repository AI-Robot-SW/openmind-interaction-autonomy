# bev_kernel.py

"""
BEVKernel — XYZ-RGB 포인트 클라우드 (N×4 float32) → BEV BGR 이미지 CUDA 커널 래퍼.

GPUWorker를 통해 GPU thread 안에서만 인스턴스화·호출해야 한다.

    kernel = gpu_worker.submit(lambda: BEVKernel(width, height)).result()
    bev    = gpu_worker.submit(lambda: kernel.run(points, res)).result()
    gpu_worker.submit(kernel.free).result()
"""

import logging
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_KERNEL_PATH = Path(__file__).resolve().parent / "bev.cu"

_THREADS = 256


class BEVKernel:
    """
    bev.cu 커널의 Python 래퍼.
    컴파일·메모리 할당·실행·해제를 캡슐화한다.
    모든 메서드는 GPU thread(GPUWorker) 안에서만 호출해야 한다.
    """

    def __init__(self, width: int, height: int) -> None:
        mod = SourceModule(_KERNEL_PATH.read_text())
        self._fn = mod.get_function("bev_kernel")

        self._width = width
        self._height = height

        # 출력 버퍼는 고정 크기 (height × width × 3 bytes) → 1회 할당
        self._gpu_bev: cuda.DeviceAllocation = cuda.mem_alloc(height * width * 3)

        # 입력 버퍼는 포인트 수(N)에 따라 캐싱
        self._gpu_points: cuda.DeviceAllocation | None = None
        self._cached_n: int = 0

        logging.info("BEVKernel: CUDA kernel compiled")

    def run(self, points: np.ndarray, res: float) -> np.ndarray:
        """
        포인트 클라우드 → BEV BGR 이미지.

        Parameters
        ----------
        points : np.ndarray
            (N, 4) float32 — [X, Y, Z, RGB-packed-as-float].
            PointCloudFrame.points와 동일한 레이아웃.
        res : float
            그리드 해상도 (meters/pixel).

        Returns
        -------
        np.ndarray
            (height, width, 3) uint8 BGR 이미지.

        GPU thread 안에서만 호출.
        """
        points_flat = np.ascontiguousarray(points, dtype=np.float32).ravel()
        n = len(points)

        if n != self._cached_n:
            self._free_input_buffer()
            self._gpu_points = cuda.mem_alloc(points_flat.nbytes)
            self._cached_n = n

        # 출력 버퍼 초기화
        cuda.memset_d8(self._gpu_bev, 0, self._height * self._width * 3)

        cuda.memcpy_htod(self._gpu_points, points_flat)

        blocks = (n + _THREADS - 1) // _THREADS
        self._fn(
            self._gpu_points,
            np.int32(n),
            np.int32(self._width),
            np.int32(self._height),
            np.float32(res),
            self._gpu_bev,
            block=(_THREADS, 1, 1),
            grid=(blocks, 1, 1),
        )

        out = np.empty(self._height * self._width * 3, dtype=np.uint8)
        cuda.memcpy_dtoh(out, self._gpu_bev)
        return out.reshape(self._height, self._width, 3)

    def free(self) -> None:
        """GPU 메모리 해제. GPU thread 안에서만 호출."""
        self._free_input_buffer()
        if self._gpu_bev is not None:
            try:
                self._gpu_bev.free()
            except Exception:
                pass
            self._gpu_bev = None

    def _free_input_buffer(self) -> None:
        if self._gpu_points is not None:
            try:
                self._gpu_points.free()
            except Exception:
                pass
            self._gpu_points = None
        self._cached_n = 0
