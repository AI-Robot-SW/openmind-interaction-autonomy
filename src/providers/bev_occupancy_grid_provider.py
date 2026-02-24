"""
BEV Occupancy Grid Provider.

Generates Bird's-Eye-View image and occupancy grid from PointCloud using CUDA,
following the contract with PointCloudProvider (data["pointcloud"]).
"""

import logging
import threading
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    import pycuda.autoinit  # noqa: F401 - initialize CUDA context before kernel compile
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule

    _PYCUDA_AVAILABLE = True
except ImportError:
    gpuarray = None
    SourceModule = None
    _PYCUDA_AVAILABLE = False

from .pointcloud_provider import PointCloudProvider
from .singleton import singleton

# CUDA kernel (vendor-compatible: x,z -> grid, BGR output)
KERNEL_CODE = """
__global__ void bev_kernel(
    float *x, float *y, float *z,
    unsigned char *r, unsigned char *g, unsigned char *b,
    int num_points, int width, int height, float res,
    unsigned char *bev_img)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) return;

    int gx = (int)(x[idx] / res + width / 2);
    int gz = (int)(height - z[idx] / res);

    if (gx >= 0 && gx < width && gz >= 0 && gz < height) {
        int offset = (gz * width + gx) * 3;
        bev_img[offset + 0] = b[idx];
        bev_img[offset + 1] = g[idx];
        bev_img[offset + 2] = r[idx];
    }
}
"""


@singleton
class BEVOccupancyGridProvider:
    """
    BEV Occupancy Grid Provider.

    Singleton that produces Bird's-Eye-View image and occupancy grid from
    PointCloud using CUDA. Reads PointCloud from PointCloudProvider.data
    (contract: data["pointcloud"] with pc["data"] and pc["point_step"],
    offsets x 0:4, y 4:8, z 8:12, rgb 12:16).

    Parameters
    ----------
    res : float, optional
        Resolution of the grid in meters per pixel (default: 0.05).
    width : int, optional
        Width of the grid in pixels (default: 50).
    height : int, optional
        Height of the grid in pixels (default: 60).
    origin_x : float, optional
        X origin of the grid in meters (default: 0.0).
    origin_y : float, optional
        Y origin of the grid in meters (default: -1.5).
    dx : float, optional
        X offset for coordinate transformation (default: -0.34).
    dy : float, optional
        Y offset for coordinate transformation (default: 0.0).
    closing_kernel_size : int, optional
        Size of morphological closing kernel (default: 1).
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
        """
        Initialize the BEV Occupancy Grid Provider.

        Parameters
        ----------
        res : float
            Resolution of the grid in meters per pixel.
        width : int
            Width of the grid in pixels.
        height : int
            Height of the grid in pixels.
        origin_x : float
            X origin of the grid in meters.
        origin_y : float
            Y origin of the grid in meters.
        dx : float
            X offset for coordinate transformation.
        dy : float
            Y offset for coordinate transformation.
        closing_kernel_size : int
            Size of morphological closing kernel.
        """
        logging.info(
            f"BEVOccupancyGridProvider booting with res={res}, size=({width},{height}), "
            f"origin=({origin_x},{origin_y}), dx={dx}, dy={dy}"
        )

        self.res = res
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.dx = dx
        self.dy = dy
        self.closing_kernel_size = closing_kernel_size

        # Internal state
        self._bev_image: Optional[np.ndarray] = None
        self._occupancy_grid: Optional[dict] = None
        self._data: Optional[dict] = None

        # Thread control
        self.running = False
        self._thread: Optional[threading.Thread] = None

        # Morphological closing kernel (height x width)
        self._closing_kernel = np.ones(
            (closing_kernel_size, closing_kernel_size), dtype=np.uint8
        )

        # CUDA: load kernel (vendor-compatible)
        self._cuda_mod: Optional[Any] = None
        self._cuda_kernel: Optional[Any] = None
        if _PYCUDA_AVAILABLE and SourceModule is not None:
            try:
                self._cuda_mod = SourceModule(KERNEL_CODE)
                self._cuda_kernel = self._cuda_mod.get_function("bev_kernel")
            except Exception as e:
                logging.error(f"BEVOccupancyGridProvider CUDA kernel load failed: {e}")
                self._cuda_mod = None
                self._cuda_kernel = None
        else:
            logging.warning(
                "BEVOccupancyGridProvider: pycuda not available; BEV output will be disabled"
            )

        logging.info("BEVOccupancyGridProvider initialized")

    def start(self) -> None:
        """
        Start the BEV Occupancy Grid Provider.

        Creates and starts a background thread if not already running.
        """
        if self._thread and self._thread.is_alive():
            logging.warning("BEVOccupancyGridProvider already running")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("BEVOccupancyGridProvider started")

    def _run(self) -> None:
        """
        Main loop: read PointCloud from PointCloudProvider, run CUDA BEV,
        build occupancy grid, update _data.
        """
        while self.running:
            try:
                provider = PointCloudProvider()
                raw = provider.data
                if not raw or "pointcloud" not in raw:
                    time.sleep(0.033)
                    continue

                pc = raw["pointcloud"]
                if pc is None:
                    time.sleep(0.033)
                    continue

                parsed = self._parse_pointcloud(pc)
                if parsed is None:
                    time.sleep(0.033)
                    continue

                x, y, z, r, g, b = parsed
                num_points = len(x)
                if num_points == 0:
                    time.sleep(0.033)
                    continue

                # CUDA BEV image
                if self._cuda_kernel is None:
                    self._data = None
                    self._bev_image = None
                    self._occupancy_grid = None
                    time.sleep(0.033)
                    continue

                bev = self._run_bev_kernel(x, y, z, r, g, b, num_points)
                if bev is None:
                    logging.error("BEVOccupancyGridProvider: BEV kernel failed")
                    self._data = None
                    self._bev_image = None
                    self._occupancy_grid = None
                    time.sleep(0.033)
                    continue

                # Occupancy grid from same x,y,z,r,g,b
                occ = self._build_occupancy_grid(x, y, z, r, g, b)
                if occ is None:
                    self._data = None
                    self._bev_image = None
                    self._occupancy_grid = None
                    time.sleep(0.033)
                    continue

                self._bev_image = bev
                self._occupancy_grid = occ
                self._data = {
                    "bev_image": self._bev_image,
                    "occupancy_grid": self._occupancy_grid,
                    "timestamp": time.time(),
                }

            except Exception as e:
                logging.error(f"Error in BEVOccupancyGridProvider loop: {e}")
                self._data = None
                self._bev_image = None
                self._occupancy_grid = None

            time.sleep(0.033)  # ~30 FPS

    def _parse_pointcloud(self, pc: dict) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Parse pointcloud dict to (x, y, z, r, g, b) numpy arrays.

        Expects A안 only: pc["data"] (bytes or np.uint8) and pc["point_step"].
        Uses offsets: x 0:4, y 4:8, z 8:12, rgb 12:16.

        Returns
        -------
        Optional[Tuple[np.ndarray, ...]]
            (x, y, z, r, g, b) or None if parsing fails.
        """
        if "data" not in pc or "point_step" not in pc:
            return None

        data = pc["data"]
        point_step = int(pc["point_step"])
        if point_step < 20:
            return None

        if isinstance(data, bytes):
            buf = np.frombuffer(data, dtype=np.uint8)
        else:
            buf = np.asarray(data, dtype=np.uint8).ravel()

        n_points = buf.size // point_step
        if n_points == 0:
            return None

        cloud_arr = buf.reshape(-1, point_step)

        x = np.frombuffer(cloud_arr[:, 0:4].tobytes(), dtype=np.float32)
        y = np.frombuffer(cloud_arr[:, 4:8].tobytes(), dtype=np.float32)
        z = np.frombuffer(cloud_arr[:, 8:12].tobytes(), dtype=np.float32)
        rgb_float = np.frombuffer(cloud_arr[:, 12:16].tobytes(), dtype=np.float32)
        rgb_int = rgb_float.view(np.uint32)
        r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_int >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_int & 0xFF).astype(np.uint8)

        if len(x) != n_points:
            return None
        return (x, y, z, r, g, b)

    def _run_bev_kernel(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
        num_points: int,
    ) -> Optional[np.ndarray]:
        """Run CUDA BEV kernel and return (height, width, 3) BGR uint8 array."""
        if self._cuda_kernel is None:
            return None
        try:
            x_gpu = gpuarray.to_gpu(x)
            y_gpu = gpuarray.to_gpu(y)
            z_gpu = gpuarray.to_gpu(z)
            r_gpu = gpuarray.to_gpu(r)
            g_gpu = gpuarray.to_gpu(g)
            b_gpu = gpuarray.to_gpu(b)
            bev_gpu = gpuarray.zeros((self.height * self.width * 3), dtype=np.uint8)

            block = (256, 1, 1)
            grid = ((num_points + 255) // 256, 1)

            self._cuda_kernel(
                x_gpu,
                y_gpu,
                z_gpu,
                r_gpu,
                g_gpu,
                b_gpu,
                np.int32(num_points),
                np.int32(self.width),
                np.int32(self.height),
                np.float32(self.res),
                bev_gpu,
                block=block,
                grid=grid,
            )

            bev = bev_gpu.get().reshape((self.height, self.width, 3))
            return bev
        except Exception as e:
            logging.error(f"BEVOccupancyGridProvider CUDA kernel error: {e}")
            return None

    def _build_occupancy_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> Optional[dict]:
        """
        Build occupancy grid from point colors and coordinates.
        Values: 0=free, 70=avoid, 88=person, 100=occupied (nav_msgs/OccupancyGrid compatible).
        """
        try:
            grid_np = np.full((self.height, self.width), 100, dtype=np.int8)

            # Color-based masks (vendor logic)
            mask_obstacle = (r > 100) & (g < 80) & (b < 80)
            mask_person = (b > 100) & (r < 80) & (g < 80)
            mask_free = (g > 100) & (r < 80) & (b < 80)
            mask_avoid = (r > 200) & (g > 200) & (b > 200)

            # Coordinate transform: x_fp = z+dx, y_fp = -x+dy -> grid indices
            x_fp = z + self.dx
            y_fp = -x + self.dy
            j_grid = ((x_fp - self.origin_x) / self.res).astype(np.int32)
            i_grid = ((y_fp - self.origin_y) / self.res).astype(np.int32)

            valid = (
                (i_grid >= 0)
                & (i_grid < self.height)
                & (j_grid >= 0)
                & (j_grid < self.width)
            )

            grid_np[i_grid[mask_obstacle & valid], j_grid[mask_obstacle & valid]] = 100
            grid_np[i_grid[mask_person & valid], j_grid[mask_person & valid]] = 88
            grid_np[i_grid[mask_avoid & valid], j_grid[mask_avoid & valid]] = 70
            grid_np[i_grid[mask_free & valid], j_grid[mask_free & valid]] = 0

            # Morphological closing on obstacle mask
            occ_mask = (grid_np == 100).astype(np.uint8)
            occ_mask = cv2.morphologyEx(
                occ_mask,
                cv2.MORPH_CLOSE,
                self._closing_kernel,
                iterations=3,
            )
            grid_np[occ_mask > 0] = 100

            return {
                "resolution": self.res,
                "width": self.width,
                "height": self.height,
                "origin_x": self.origin_x,
                "origin_y": self.origin_y,
                "data": grid_np,
            }
        except Exception as e:
            logging.error(f"BEVOccupancyGridProvider occupancy grid build failed: {e}")
            return None

    def stop(self) -> None:
        """
        Stop the BEV Occupancy Grid Provider.

        Stops the background thread and releases CUDA resources.
        """
        self.running = False
        if self._thread:
            logging.info("Stopping BEVOccupancyGridProvider")
            self._thread.join(timeout=5)
            self._thread = None

        self._cuda_kernel = None
        self._cuda_mod = None

        logging.info("BEVOccupancyGridProvider stopped")

    @property
    def data(self) -> Optional[dict]:
        """
        Get the current BEV occupancy grid data.

        Returns
        -------
        Optional[dict]
            Dictionary with "bev_image", "occupancy_grid", "timestamp",
            or None if not available.
        """
        return self._data
