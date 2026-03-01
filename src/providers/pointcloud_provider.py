import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .singleton import singleton

try:
    import pycuda.autoinit  # noqa: F401 - initialize CUDA context
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    _PYCUDA_AVAILABLE = True
except ImportError:
    cuda = None
    SourceModule = None
    _PYCUDA_AVAILABLE = False

# CUDA kernel (legacy-compatible)
_KERNEL_CODE = r"""
#include <math.h>

extern "C"
__global__ void depth_rgb_to_xyzrgb_kernel(
    const float* depth,
    const unsigned char* rgb,
    float* out,
    int width,
    int height,
    float fx, float fy,
    float cx, float cy,
    int rgb_step,
    int red_offset,
    int green_offset,
    int blue_offset,
    float range_max
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = width * height;
    if (idx >= num_pixels) return;

    float d = depth[idx];
    float X, Y, Z;

    if (d <= 0.0f || (range_max > 0.0f && d > range_max)) {
        X = NAN;
        Y = NAN;
        Z = NAN;
    } else {
        int u = idx % width;
        int v = idx / width;
        Z = d;
        X = ((float)u - cx) * d / fx;
        Y = ((float)v - cy) * d / fy;
    }

    int rgb_idx = idx * rgb_step;
    unsigned char r = rgb[rgb_idx + red_offset];
    unsigned char g = rgb[rgb_idx + green_offset];
    unsigned char b = rgb[rgb_idx + blue_offset];

    unsigned int rgb_packed =
        ((unsigned int)r << 16) |
        ((unsigned int)g << 8)  |
        (unsigned int)b;

    float rgb_f = __int_as_float(rgb_packed);

    int base = idx * 4;
    out[base + 0] = X;
    out[base + 1] = Y;
    out[base + 2] = Z;
    out[base + 3] = rgb_f;
}
"""


@singleton
class PointCloudProvider:
    """
    PointCloud Provider.

    This class implements a singleton pattern to manage:
        * PointCloud generation from Depth and RGB images using CUDA

    Parameters
    ----------
    range_max : float, optional
        Maximum range for point cloud filtering (default: 0.0, no limit)
    stride : int, optional
        Downsampling stride (default: 1, no downsampling)
    sync_slop : float, optional
        Time synchronization tolerance in seconds (default: 0.05)
    """

    def __init__(
        self,
        range_max: float = 0.0,
        stride: int = 1,
        sync_slop: float = 0.05,
    ):
        """
        Initialize the PointCloud Provider.
        Parameters
        ----------
        range_max : float
            Maximum range for point cloud filtering.
        stride : int
            Downsampling stride.
        sync_slop : float
            Time synchronization tolerance in seconds.
        """
        logging.info(
            f"PointCloudProvider booting with range_max={range_max}, stride={stride}, sync_slop={sync_slop}"
        )

        self.range_max = range_max
        self.stride = stride
        self.sync_slop = sync_slop

        # Internal state variables
        self._pointcloud_data: Optional[dict] = None
        self._data: Optional[dict] = None
        self._lock = threading.Lock()

        # Thread control
        self.running = False
        self._thread: Optional[threading.Thread] = None

        # Message synchronization state
        self._last_depth_data: Optional[dict] = None
        self._last_rgb_data: Optional[dict] = None
        self._last_camera_info: Optional[dict] = None
        self._last_sync_time: Optional[float] = None

        # External providers (lazy)
        self._realsense_provider = None
        self._segmentation_provider = None

        # CUDA initialization (placeholder)
        self._gpu_kernel = None
        self._gpu_mod = None
        self._gpu_cached_pixels = 0
        self._gpu_depth = None
        self._gpu_rgb = None
        self._gpu_cloud = None
        self._gpu_log_once = False
        self._cuda_ctx = None

        logging.info(f"PointCloudProvider CUDA available: {_PYCUDA_AVAILABLE}")

        logging.info("PointCloudProvider initialized")

    def start(self):
        """
        Start the PointCloud Provider.
        Creates and starts a background thread if not already running.
        """
        if self._thread and self._thread.is_alive():
            logging.warning("PointCloudProvider already running")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("PointCloudProvider started")

    def _run(self):
        """
        Main loop for the PointCloud Provider.
        Continuously synchronizes and processes depth/rgb/camera_info to generate point clouds.
        """
        try:
            while self.running:
                try:
                    if self._realsense_provider is None:
                        from .realsense_camera_provider import RealSenseCameraProvider

                        self._realsense_provider = RealSenseCameraProvider()
                    if self._segmentation_provider is None:
                        try:
                            from .segmentation_provider import SegmentationProvider

                            self._segmentation_provider = SegmentationProvider()
                        except Exception as e:
                            logging.warning(
                                f"SegmentationProvider import/init failed; using RGB only: {e}"
                            )
                            self._segmentation_provider = None

                    self._update_data()

                except Exception as e:
                    logging.error(f"Error in PointCloudProvider loop: {e}")
                    self._data = None
                    self._pointcloud_data = None

                time.sleep(0.033)  # ~30 FPS
        finally:
            self._cleanup_cuda_ctx()

    def _select_intrinsics(self, camera_info: dict) -> Optional[Tuple[float, float, float, float]]:
        if not camera_info:
            return None

        intr = camera_info.get("color") or camera_info.get("depth")
        if not intr:
            return None

        try:
            fx = float(intr["fx"])
            fy = float(intr["fy"])
            cx = float(intr["cx"])
            cy = float(intr["cy"])
        except Exception:
            return None

        if fx == 0.0 or fy == 0.0:
            return None
        return fx, fy, cx, cy

    def _pack_cloud4(self, arr: np.ndarray) -> Optional[dict]:
        if arr.ndim != 2 or arr.shape[1] != 4:
            return None

        n_points = int(arr.shape[0])
        if n_points <= 0:
            return None

        point_step = 20
        buf = np.zeros((n_points, point_step), dtype=np.uint8)
        arr32 = np.ascontiguousarray(arr, dtype=np.float32)
        buf[:, 0:16] = arr32.view(np.uint8).reshape(n_points, 16)

        return {
            "data": buf.ravel(),
            "point_step": int(point_step),
            "width": int(n_points),
            "height": 1,
            "is_dense": True,
        }

    def _encoding_to_offsets(self, encoding: str, channels: int):
        enc = (encoding or "").lower()
        red_offset = 0
        green_offset = 1
        blue_offset = 2
        rgb_step = channels

        if enc == "rgb8":
            red_offset, green_offset, blue_offset, rgb_step = 0, 1, 2, 3
        elif enc == "rgba8":
            red_offset, green_offset, blue_offset, rgb_step = 0, 1, 2, 4
        elif enc == "bgr8":
            red_offset, green_offset, blue_offset, rgb_step = 2, 1, 0, 3
        elif enc == "bgra8":
            red_offset, green_offset, blue_offset, rgb_step = 2, 1, 0, 4
        elif enc == "mono8":
            red_offset = green_offset = blue_offset = 0
            rgb_step = 1
        else:
            red_offset, green_offset, blue_offset, rgb_step = 0, 1, 2, channels

        return rgb_step, red_offset, green_offset, blue_offset

    def _ensure_gpu_kernel(self) -> bool:
        if not _PYCUDA_AVAILABLE or SourceModule is None or cuda is None:
            return False
        if self._gpu_kernel is not None:
            return True
        try:
            if cuda.Context.get_current() is None:
                self._cuda_ctx = cuda.Device(0).make_context()
            self._gpu_mod = SourceModule(_KERNEL_CODE)
            self._gpu_kernel = self._gpu_mod.get_function("depth_rgb_to_xyzrgb_kernel")
            logging.info("PointCloudProvider CUDA kernel loaded successfully")
            return True
        except Exception as e:
            logging.error(f"PointCloudProvider CUDA kernel load failed: {e}")
            self._gpu_mod = None
            self._gpu_kernel = None
            return False

    def _cleanup_cuda_ctx(self) -> None:
        if self._cuda_ctx is not None:
            try:
                if self._gpu_cloud is not None:
                    try:
                        self._gpu_cloud.free()
                    except Exception:
                        pass
                if self._gpu_rgb is not None:
                    try:
                        self._gpu_rgb.free()
                    except Exception:
                        pass
                if self._gpu_depth is not None:
                    try:
                        self._gpu_depth.free()
                    except Exception:
                        pass
                self._gpu_cloud = None
                self._gpu_rgb = None
                self._gpu_depth = None
                self._gpu_cached_pixels = 0
                self._gpu_mod = None
                self._gpu_kernel = None
                self._cuda_ctx.pop()
            except Exception as e:
                logging.warning(f"PointCloudProvider CUDA context cleanup failed: {e}")
            self._cuda_ctx = None

    def _depth_rgb_to_xyzrgb_gpu(
        self,
        depth_np: np.ndarray,
        rgb_np: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        encoding: str,
        range_max: float,
    ) -> Optional[np.ndarray]:
        if not self._ensure_gpu_kernel():
            return None

        depth_flat = np.ascontiguousarray(depth_np, dtype=np.float32).reshape(-1)
        if rgb_np.ndim == 2:
            rgb_np = rgb_np[:, :, None]
        channels = int(rgb_np.shape[2])
        rgb_flat = np.ascontiguousarray(rgb_np, dtype=np.uint8).reshape(-1)

        rgb_step, red_offset, green_offset, blue_offset = self._encoding_to_offsets(
            encoding, channels
        )

        num_pixels = int(depth_np.shape[0] * depth_np.shape[1])
        if self._gpu_cached_pixels != num_pixels or self._gpu_depth is None:
            self._gpu_cached_pixels = num_pixels
            self._gpu_depth = cuda.mem_alloc(depth_flat.nbytes)
            self._gpu_rgb = cuda.mem_alloc(rgb_flat.nbytes)
            self._gpu_cloud = cuda.mem_alloc(num_pixels * 4 * np.float32().nbytes)

        cuda.memcpy_htod(self._gpu_depth, depth_flat)
        cuda.memcpy_htod(self._gpu_rgb, rgb_flat)

        threads_per_block = 256
        blocks = (num_pixels + threads_per_block - 1) // threads_per_block

        self._gpu_kernel(
            self._gpu_depth,
            self._gpu_rgb,
            self._gpu_cloud,
            np.int32(depth_np.shape[1]),
            np.int32(depth_np.shape[0]),
            np.float32(fx),
            np.float32(fy),
            np.float32(cx),
            np.float32(cy),
            np.int32(rgb_step),
            np.int32(red_offset),
            np.int32(green_offset),
            np.int32(blue_offset),
            np.float32(range_max),
            block=(threads_per_block, 1, 1),
            grid=(blocks, 1, 1),
        )

        cloud4_flat = np.empty(num_pixels * 4, dtype=np.float32)
        cuda.memcpy_dtoh(cloud4_flat, self._gpu_cloud)
        return cloud4_flat.reshape(depth_np.shape[0], depth_np.shape[1], 4)

    def _process_triplet(self, depth_msg: dict, rgb_msg: dict, info_msg: dict) -> None:
        depth_img = depth_msg.get("image")
        if depth_img is None:
            self._pointcloud_data = None
            return

        depth_enc = str(depth_msg.get("encoding") or "32FC1")
        depth_np = np.asarray(depth_img)
        if depth_enc in ["16UC1", "mono16"]:
            scale = float(depth_msg.get("depth_scale") or 0.001)
            depth_np = depth_np.astype(np.float32) * scale
        elif depth_enc == "32FC1":
            depth_np = depth_np.astype(np.float32, copy=False)
        else:
            depth_np = depth_np.astype(np.float32, copy=False)

        color_img = rgb_msg.get("image")
        if color_img is None:
            self._pointcloud_data = None
            return
        color_np = np.asarray(color_img)

        intr = self._select_intrinsics(info_msg.get("camera_info") or {})
        if intr is None:
            self._pointcloud_data = None
            return

        fx, fy, cx, cy = intr

        if depth_np.ndim != 2 or color_np.ndim != 3 or color_np.shape[2] != 3:
            self._pointcloud_data = None
            return
        if depth_np.shape[0] != color_np.shape[0] or depth_np.shape[1] != color_np.shape[1]:
            self._pointcloud_data = None
            return

        stride = max(1, int(self.stride))
        depth_s = depth_np[::stride, ::stride].astype(np.float32, copy=False)
        color_s = color_np[::stride, ::stride].astype(np.uint8, copy=False)

        max_range = float(self.range_max)
        use_gpu = _PYCUDA_AVAILABLE and self._ensure_gpu_kernel()
        if use_gpu:
            if not self._gpu_log_once:
                logging.info("PointCloudProvider using CUDA path")
                self._gpu_log_once = True
            cloud4 = self._depth_rgb_to_xyzrgb_gpu(
                depth_s,
                color_s,
                fx,
                fy,
                cx,
                cy,
                encoding=str(rgb_msg.get("encoding") or "bgr8"),
                range_max=max_range,
            )
            if cloud4 is None:
                use_gpu = False

        if not use_gpu:
            h_s, w_s = depth_s.shape
            v_idx, u_idx = np.indices((h_s, w_s), dtype=np.float32)
            u = (u_idx * stride).astype(np.float32)
            v = (v_idx * stride).astype(np.float32)

            z = depth_s
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            r = color_s[:, :, 2].astype(np.uint32)
            g = color_s[:, :, 1].astype(np.uint32)
            b = color_s[:, :, 0].astype(np.uint32)
            rgb_uint32 = (r << 16) | (g << 8) | b
            rgb_float = rgb_uint32.view(np.float32)

            cloud4 = np.stack([x, y, z, rgb_float], axis=-1).astype(np.float32, copy=False)

        flat = cloud4.reshape(-1, 4)
        if max_range > 0.0:
            mask = (flat[:, 2] > 0.0) & (flat[:, 2] <= max_range) & np.isfinite(flat[:, 2])
        else:
            mask = (flat[:, 2] > 0.0) & np.isfinite(flat[:, 2])

        kept = flat[mask]
        if kept.shape[0] == 0:
            self._pointcloud_data = None
            return

        self._pointcloud_data = self._pack_cloud4(kept)
        if not self.running:
            self._cleanup_cuda_ctx()

    def _try_sync(self) -> None:
        if (
            self._last_depth_data is None
            or self._last_rgb_data is None
            or self._last_camera_info is None
        ):
            return

        t_d = self._last_depth_data.get("timestamp")
        t_r = self._last_rgb_data.get("timestamp")
        t_i = self._last_camera_info.get("timestamp")
        if t_d is None or t_r is None or t_i is None:
            return

        t_d = float(t_d)
        t_r = float(t_r)
        t_i = float(t_i)

        t_min = min(t_d, t_r, t_i)
        t_max = max(t_d, t_r, t_i)

        if self._last_sync_time is not None and t_max < self._last_sync_time - 1e-3:
            logging.warning(
                "PointCloudProvider time jumped backwards (last_sync_time=%.3f, new=%.3f), "
                "reset sync buffers.",
                self._last_sync_time,
                t_max,
            )
            self._last_depth_data = None
            self._last_rgb_data = None
            self._last_camera_info = None
            self._last_sync_time = None
            return

        if t_max - t_min <= float(self.sync_slop):
            self._last_sync_time = t_max
            depth_msg = self._last_depth_data
            rgb_msg = self._last_rgb_data
            info_msg = self._last_camera_info

            self._last_depth_data = None
            self._last_rgb_data = None
            self._last_camera_info = None

            self._process_triplet(depth_msg, rgb_msg, info_msg)

    def _update_data(self):
        """
        Update internal data from point cloud processing.
        This method processes synchronized depth/rgb/camera_info and generates point clouds.
        """
        ts = time.time()

        rs = self._realsense_provider.data if self._realsense_provider is not None else None
        seg = (
            self._segmentation_provider.data
            if self._segmentation_provider is not None
            else None
        )

        if not rs:
            self._pointcloud_data = None
            with self._lock:
                self._data = {"pointcloud": None, "timestamp": ts}
            return

        depth_data = rs.get("depth") if isinstance(rs, dict) else None
        camera_info = rs.get("camera_info") if isinstance(rs, dict) else None
        rgb_data = rs.get("rgb") if isinstance(rs, dict) else None

        rs_ts = rs.get("timestamp") if isinstance(rs, dict) else None
        seg_ts = seg.get("timestamp") if isinstance(seg, dict) else None

        depth_img = depth_data.get("image") if depth_data else None
        rgb_img = rgb_data.get("image") if rgb_data else None
        segmented = seg.get("segmented_image") if isinstance(seg, dict) else None

        if depth_img is not None and rs_ts is not None:
            self._last_depth_data = {
                "image": depth_img,
                "encoding": depth_data.get("encoding") if depth_data else None,
                "depth_scale": depth_data.get("depth_scale") if depth_data else None,
                "timestamp": float(rs_ts),
            }

        if camera_info is not None and rs_ts is not None:
            self._last_camera_info = {
                "camera_info": camera_info,
                "timestamp": float(rs_ts),
            }

        color_img = None
        color_enc = None
        if isinstance(segmented, np.ndarray):
            color_img = segmented
            color_enc = "bgr8"
            ts_rgb = seg_ts if seg_ts is not None else rs_ts
        elif isinstance(rgb_img, np.ndarray):
            color_img = rgb_img
            color_enc = rgb_data.get("encoding") if rgb_data else None
            ts_rgb = rs_ts
        else:
            ts_rgb = None

        if color_img is not None and ts_rgb is not None:
            self._last_rgb_data = {
                "image": color_img,
                "encoding": color_enc,
                "timestamp": float(ts_rgb),
            }

        self._try_sync()

        with self._lock:
            self._data = {
                "pointcloud": self._pointcloud_data,
                "timestamp": ts,
            }

    def stop(self):
        """
        Stop the PointCloud Provider.
        Stops the background thread and cleans up resources.
        """
        self.running = False
        if self._thread:
            logging.info("Stopping PointCloudProvider")
            self._thread.join(timeout=5)

        logging.info("PointCloudProvider stopped")

    @property
    def data(self) -> Optional[dict]:
        """
        Get the current point cloud data.
        Returns
        -------
        Optional[dict]
            Dictionary containing point cloud data,
            or None if not available.
        """
        with self._lock:
            return self._data
