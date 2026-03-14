# pointcloud_kernel.py

"""
PointCloudKernel — depth + RGB → XYZ RGB 포인트 클라우드 CUDA 커널 래퍼.

GPUWorker를 통해 GPU thread 안에서만 인스턴스화·호출해야 한다.

    kernel = gpu_worker.submit(lambda: PointCloudKernel()).result()
    points = gpu_worker.submit(lambda: kernel.run(depth, rgb, fx, fy, cx, cy, range_max)).result()
    gpu_worker.submit(kernel.free).result()
"""

import logging
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


_KERNEL_PATH = Path(__file__).resolve().parent / "pointcloud.cu"

# BGR 인코딩 오프셋
_BGR_STEP  = 3
_BGR_RED   = 2
_BGR_GREEN = 1
_BGR_BLUE  = 0


class PointCloudKernel:
    """
    pointcloud.cu 커널의 Python 래퍼.
    컴파일·메모리 할당·실행·해제를 캡슐화한다.
    모든 메서드는 GPU thread(GPUWorker) 안에서만 호출해야 한다.
    """

    def __init__(self) -> None:
        mod = SourceModule(_KERNEL_PATH.read_text())
        self._fn = mod.get_function("depth_rgb_to_xyzrgb_kernel")

        self._gpu_depth: cuda.DeviceAllocation | None = None
        self._gpu_rgb:   cuda.DeviceAllocation | None = None
        self._gpu_cloud: cuda.DeviceAllocation | None = None
        self._cached_pixels: int = 0

        logging.info("PointCloudKernel: CUDA kernel compiled")

    def run(
        self,
        depth_np: np.ndarray,
        rgb_np:   np.ndarray,
        fx: float, fy: float,
        cx: float, cy: float,
        range_max: float,
    ) -> np.ndarray:
        """
        depth(H,W) float32 + rgb(H,W,3) uint8 BGR → points(H*W, 4) float32
        col 0=X, col 1=Y, col 2=Z, col 3=RGB-packed-as-float

        GPU thread 안에서만 호출.
        """
        depth_flat = np.ascontiguousarray(depth_np, dtype=np.float32).ravel()
        rgb_flat   = np.ascontiguousarray(rgb_np,   dtype=np.uint8).ravel()
        num_pixels = depth_np.shape[0] * depth_np.shape[1]

        if self._cached_pixels != num_pixels:
            self._free_buffers()
            self._gpu_depth = cuda.mem_alloc(depth_flat.nbytes)
            self._gpu_rgb   = cuda.mem_alloc(rgb_flat.nbytes)
            self._gpu_cloud = cuda.mem_alloc(num_pixels * 4 * np.float32().nbytes)
            self._cached_pixels = num_pixels

        cuda.memcpy_htod(self._gpu_depth, depth_flat)
        cuda.memcpy_htod(self._gpu_rgb,   rgb_flat)

        threads = 256
        blocks  = (num_pixels + threads - 1) // threads
        self._fn(
            self._gpu_depth, self._gpu_rgb, self._gpu_cloud,
            np.int32(depth_np.shape[1]), np.int32(depth_np.shape[0]),
            np.float32(fx), np.float32(fy),
            np.float32(cx), np.float32(cy),
            np.int32(_BGR_STEP), np.int32(_BGR_RED),
            np.int32(_BGR_GREEN), np.int32(_BGR_BLUE),
            np.float32(range_max),
            block=(threads, 1, 1), grid=(blocks, 1, 1),
        )

        out = np.empty(num_pixels * 4, dtype=np.float32)
        cuda.memcpy_dtoh(out, self._gpu_cloud)
        return out.reshape(num_pixels, 4)

    def free(self) -> None:
        """GPU 메모리 해제. GPU thread 안에서만 호출."""
        self._free_buffers()

    def _free_buffers(self) -> None:
        for attr in ("_gpu_depth", "_gpu_rgb", "_gpu_cloud"):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    buf.free()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._cached_pixels = 0
