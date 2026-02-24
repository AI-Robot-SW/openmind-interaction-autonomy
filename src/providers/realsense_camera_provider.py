from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None
    _rs_import_error = e

from .singleton import singleton


@dataclass(frozen=True)
class FrameBundle:
    """Single frame bundle: rgb, depth, camera_info, timestamp."""

    rgb: Dict[str, Any]
    depth: Dict[str, Any]
    camera_info: Dict[str, Any]
    timestamp: float


@singleton
class RealSenseCameraProvider:
    """
    RealSense camera Provider.

    Manages RealSense camera pipeline and exposes rgb/depth/camera_info via
    the `data` property. Pipeline open/close and frame read logic are
    integrated in this class.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_depth_to_color: bool = True,
    ):
        logging.info(f"RealSenseCameraProvider booting at camera_index: {camera_index}")

        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.align_depth_to_color = bool(align_depth_to_color)

        # tests expect these
        self._rgb_data: Optional[dict] = None
        self._depth_data: Optional[dict] = None
        self._camera_info: Optional[dict] = None
        self._data: Optional[dict] = None

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Pipeline state (formerly in RealSenseSDK)
        self._pipeline: Optional["rs.pipeline"] = None
        self._config: Optional["rs.config"] = None
        self._profile: Optional["rs.pipeline_profile"] = None
        self._align: Optional["rs.align"] = None
        self._depth_scale: Optional[float] = None

        logging.info("RealSenseCameraProvider initialized")

    def start(self):
        """Start the provider."""
        if self._thread and self._thread.is_alive():
            logging.warning("RealSenseCameraProvider already running")
            return

        self._start_sdk()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("RealSenseCameraProvider started")

    def stop(self):
        """Stop the provider and release pipeline."""
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None

        self._stop_sdk()

        logging.info("RealSenseCameraProvider stopped")

    def _run(self):
        period = 1.0 / max(1, self.fps)

        while self.running:
            t0 = time.time()
            try:
                self._update_data()
            except Exception as e:
                logging.error(f"Error in RealSenseCameraProvider loop: {e}")
                with self._lock:
                    self._data = None
                    self._rgb_data = None
                    self._depth_data = None
                    self._camera_info = None

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))

    def _start_sdk(self):
        if self._pipeline is not None:
            return
        if rs is None:
            raise ImportError(f"pyrealsense2 import failed: {_rs_import_error}")

        self._pipeline = rs.pipeline()
        self._config = rs.config()

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found.")
        if self.camera_index < 0 or self.camera_index >= len(devices):
            raise RuntimeError(
                f"camera_index out of range: {self.camera_index} / {len(devices)}"
            )

        dev = devices[self.camera_index]
        serial = dev.get_info(rs.camera_info.serial_number)
        self._config.enable_device(serial)

        self._config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self._config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )

        self._profile = self._pipeline.start(self._config)

        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        self._align = (
            rs.align(rs.stream.color) if self.align_depth_to_color else None
        )

        for _ in range(10):
            self._pipeline.wait_for_frames()

    def _stop_sdk(self):
        if self._pipeline is None:
            return
        try:
            self._pipeline.stop()
        finally:
            self._pipeline = None
            self._config = None
            self._profile = None
            self._align = None
            self._depth_scale = None

    def _read_frames(self, timestamp: float) -> FrameBundle:
        if self._pipeline is None:
            raise RuntimeError("SDK not started. Call start() first.")

        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Missing color or depth frame.")

        color_img = np.asanyarray(color_frame.get_data())
        h, w, _ = color_img.shape
        rgb = {
            "image": color_img,
            "encoding": "bgr8",
            "width": int(w),
            "height": int(h),
            "stride": int(w * 3),
        }

        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        scale = float(self._depth_scale or 0.001)
        depth_m = depth_raw * scale
        dh, dw = depth_m.shape
        depth = {
            "image": depth_m,
            "encoding": "32FC1",
            "width": int(dw),
            "height": int(dh),
            "depth_scale": scale,
            "aligned_to": "color" if self._align is not None else "depth",
        }

        camera_info = self._read_camera_info()

        return FrameBundle(
            rgb=rgb,
            depth=depth,
            camera_info=camera_info,
            timestamp=timestamp,
        )

    def _read_camera_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if self._profile is None:
            return info

        try:
            color_stream = (
                self._profile.get_stream(rs.stream.color).as_video_stream_profile()
            )
            ci = color_stream.get_intrinsics()
            info["color"] = {
                "width": int(ci.width),
                "height": int(ci.height),
                "fx": float(ci.fx),
                "fy": float(ci.fy),
                "cx": float(ci.ppx),
                "cy": float(ci.ppy),
                "model": str(ci.model),
                "coeffs": [float(x) for x in ci.coeffs],
            }
        except Exception:
            pass

        try:
            depth_stream = (
                self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
            )
            di = depth_stream.get_intrinsics()
            info["depth"] = {
                "width": int(di.width),
                "height": int(di.height),
                "fx": float(di.fx),
                "fy": float(di.fy),
                "cx": float(di.ppx),
                "cy": float(di.ppy),
                "model": str(di.model),
                "coeffs": [float(x) for x in di.coeffs],
            }
        except Exception:
            pass

        return info

    def _update_data(self):
        ts = time.time()
        bundle = self._read_frames(timestamp=ts)

        self._rgb_data = bundle.rgb
        self._depth_data = bundle.depth
        self._camera_info = bundle.camera_info

        with self._lock:
            self._data = {
                "rgb": self._rgb_data,
                "depth": self._depth_data,
                "camera_info": self._camera_info,
                "timestamp": bundle.timestamp,
            }

    @property
    def data(self) -> Optional[dict]:
        """
        Get the current provider data (rgb, depth, camera_info, timestamp).

        Returns
        -------
        Optional[dict]
            Current frame data, or None if not available.
        """
        with self._lock:
            return self._data
