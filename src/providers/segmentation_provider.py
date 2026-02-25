#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import atexit
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, List

import numpy as np
import cv2
import torch

from .singleton import singleton

from providers.utils.trt_utils.trt_engine import TRTEngine  # type: ignore
from transformers import AutoImageProcessor  # type: ignore

from providers.realsense_camera_provider import RealSenseCameraProvider  # type: ignore
import pycuda.autoinit


def _default_engine_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "providers"
        / "engines"
        / "trt"
        / "ddrnet23_fp16_kist-v1-80k_1x480x640.engine"
    )

@dataclass
class SegmentationProviderConfig:
    engine_path: str = _default_engine_path()

    hf_model_id: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    hf_cache_dir: Optional[str] = "/home/nvidia/.cache/huggingface"
    hf_local_files_only: bool = True

    input_h: int = 480
    input_w: int = 640

    rate_hz: float = 30.0
    alpha: float = 0.9

    driveable_ids: Optional[Set[int]] = None
    person_ids: Optional[Set[int]] = None
    avoid_ids: Optional[Set[int]] = None
    curb_ids: Optional[Set[int]] = None


@singleton
class SegmentationProvider:
    """
    OpenMind-style Segmentation Provider:
    - singleton
    - start/stop background thread
    - reads latest frame from RealSenseCameraProvider().data
    - runs TensorRT inference (Mask2Former)
    - updates self.data (thread-safe)
    """

    def __init__(
        self,
        cfg: Optional[SegmentationProviderConfig] = None,
        # compatibility: 기존 테스트/호출 형태를 살리고 싶으면 engine_path만 넣어도 되게
        engine_path: str = _default_engine_path(),
        camera_provider: Optional[RealSenseCameraProvider] = None,
        processor_factory: Optional[Callable[[], Any]] = None,
        engine_factory: Optional[Callable[[], Any]] = None,
        auto_start_camera: bool = False,
    ):
        if cfg is None:
            if not engine_path:
                engine_path = ""
            cfg = SegmentationProviderConfig(engine_path=engine_path)

        self.cfg = cfg
        self.engine_path = cfg.engine_path  
        self.hf_cache_dir = cfg.hf_cache_dir

        self.cam = camera_provider if camera_provider is not None else RealSenseCameraProvider()
        self.auto_start_camera = bool(auto_start_camera)

        # dependency injection for clean tests
        self._processor_factory = processor_factory
        self._engine_factory = engine_factory

        # heavy objects (lazy-loaded)
        self._processor = None
        self._engine = None
        self._cuda_ctx = None
        self._atexit_registered = False

        # internal states 
        self._segmented_image: Optional[np.ndarray] = None
        self._classes: Optional[List[int]] = None
        self._data: Optional[Dict[str, Any]] = None

        # thread control
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running = False

        # semantic grouping sets 
        self._driveable = set(cfg.driveable_ids) if cfg.driveable_ids is not None else {15, 23, 7, 11, 41, 8, 28}
            #sidewalk,lane,bikelane,pedestrian area,manhole,crosswalk,snow
        self._person = set(cfg.person_ids) if cfg.person_ids is not None else {19} 
            # person
        self._avoid = set(cfg.avoid_ids) if cfg.avoid_ids is not None else {13, 6, 29, 30, 55, 59, 57, 54, 52}
            #road,wall,terrain,vegetation,car,other vehicle,motorcycle,bus,bicycle
        self._curb = set(cfg.curb_ids) if cfg.curb_ids is not None else {2, 24}
            #lane mk-general,curb     
        self._period_s = 1.0 / max(1e-6, float(cfg.rate_hz))
            # 나머지
        self._register_atexit()
        logging.info(f"SegmentationProvider initialized (engine_path={self.engine_path})")

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        self._atexit_registered = True
        atexit.register(self._safe_shutdown)

    def _safe_shutdown(self) -> None:
        try:
            if self.running:
                self.stop()
        except Exception:
            logging.exception("SegmentationProvider shutdown failed")

    # ---------------- public API ----------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.warning("SegmentationProvider already running")
            return

        if self.auto_start_camera:
            try:
                self.cam.start()
            except Exception as e:
                logging.warning(f"Camera start failed (ignored): {e}")

        self._stop_evt.clear()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logging.info("SegmentationProvider started")

    def stop(self) -> None:
        self.running = False
        self._stop_evt.set()

        if self._thread:
            self._thread.join(timeout=5.0)

        if self.auto_start_camera:
            try:
                self.cam.stop()
            except Exception as e:
                logging.warning(f"Camera stop failed (ignored): {e}")

        logging.info("SegmentationProvider stopped")

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data

    # ---------------- internal helpers ----------------
    def _set_data(self, d: Optional[Dict[str, Any]]) -> None:
        with self._lock:
            self._data = d

    def _ensure_loaded(self) -> None:
        """
        Load heavy objects only once.
        - HF processor (resize to engine input)
        - TRT engine
        """
        if self._processor is None:
            processor_kwargs = {
                "use_fast": True,
                "do_resize": True,
                "size": {"height": int(self.cfg.input_h), "width": int(self.cfg.input_w)},
                "do_rescale": True,
            }
            if self.hf_cache_dir:
                processor_kwargs["cache_dir"] = self.hf_cache_dir
                os.environ.setdefault("HF_HOME", self.hf_cache_dir)
            processor_kwargs["local_files_only"] = bool(self.cfg.hf_local_files_only)

            if self._processor_factory is not None:
                self._processor = self._processor_factory()
            else:
                self._processor = AutoImageProcessor.from_pretrained(
                    self.cfg.hf_model_id,
                    **processor_kwargs,
                )


        if self._engine is None:
            if self._engine_factory is not None:
                self._engine = self._engine_factory()
            else:
                try:
                    # Ensure CUDA context exists in this thread before TRT allocs.
                    import pycuda.driver as cuda
                    cuda.init()
                    if cuda.Context.get_current() is None:
                        self._cuda_ctx = cuda.Device(0).make_context()
                except Exception as e:
                    logging.warning(f"CUDA context init failed or unavailable: {e}")
                if not self.cfg.engine_path:
                    raise ValueError("engine_path is empty. Provide cfg.engine_path")
                self._engine = TRTEngine(self.cfg.engine_path)

    def _get_frame_bgr(self) -> tuple[Optional[np.ndarray], Optional[float]]:
        """
        RealSenseCameraProvider.data schema:
        {
          "rgb": {"image": np.ndarray, "encoding": "bgr8", ...},
          "depth": {...},
          "camera_info": {...},
          "timestamp": float
        }
        """
        d = getattr(self.cam, "data", None)
        if not d:
            return None, None

        rgb = d.get("rgb")
        if not rgb:
            return None, None

        frame = rgb.get("image")
        if frame is None:
            return None, None

        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return None, None
        ts = d.get("timestamp")
        try:
            ts = float(ts) if ts is not None else None
        except Exception:
            ts = None
        return frame, ts

    def _build_semantic_map(self, class_map: np.ndarray) -> np.ndarray:
        """
        Convert class-id map -> compact semantic map:
        0: unknown/other
        1: driveable
        2: person
        3: avoid
        4: curb
        """
        sem = np.zeros(class_map.shape, dtype=np.uint8)
        uniq = np.unique(class_map)
        for cid in uniq.tolist():
            if cid in self._driveable:
                sem[class_map == cid] = 1
            elif cid in self._person:
                sem[class_map == cid] = 2
            elif cid in self._avoid:
                sem[class_map == cid] = 3
            elif cid in self._curb:
                sem[class_map == cid] = 4
            else:
                sem[class_map == cid] = 0
        return sem

    def _colorize_semantic(self, sem: np.ndarray) -> np.ndarray:
        """
        Create BGR overlay image from semantic map.
        """
        color_map = {
            0: (128, 128, 128),  # gray
            1: (0, 255, 0),      # green
            2: (255, 0, 0),      # blue (BGR) - person
            3: (0, 0, 255),      # red - avoid
            4: (255, 255, 255),  # white
        }
        colored = np.zeros((sem.shape[0], sem.shape[1], 3), dtype=np.uint8)
        for k, c in color_map.items():
            colored[sem == k] = c
        return colored

    # ---------------- core processing ----------------
    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Pure-ish processing for one frame.
        (Still uses self._processor, self._engine)
        """
        t0 = time.time()

        # BGR -> RGB for HF processor
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inputs = self._processor(images=img_rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].detach().cpu().numpy().copy()

        outputs = self._engine.infer(pixel_values)
        try:
            out0 = outputs[0] if isinstance(outputs, (list, tuple)) and len(outputs) > 0 else None
            logging.debug(
                "Segmentation TRT output[0] shape=%s dtype=%s",
                getattr(out0, "shape", None),
                getattr(out0, "dtype", None),
            )
        except Exception:
            logging.exception("Failed to log TRT output shape")

        # expected: outputs[0] = logits [1, C, H, W]
        logits = torch.from_numpy(outputs[0])
        predicted_small = logits.argmax(dim=1).cpu().numpy()[0].astype(np.int32)

        # resize back to original frame size (nearest for labels)
        predicted_map = cv2.resize(
            predicted_small,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

        sem = self._build_semantic_map(predicted_map)
        colored = self._colorize_semantic(sem)

        alpha = float(self.cfg.alpha)
        overlay = cv2.addWeighted(frame_bgr, 1.0 - alpha, colored, alpha, 0)

        uniq = np.unique(predicted_map).astype(int).tolist()

        return {
            "segmented_image": overlay,      # BGR uint8
            "semantic_map": sem,             # uint8
            "predicted_map": predicted_map,  # int32
            "classes": uniq,                 # list[int]
            "latency_s": float(time.time() - t0),
            "timestamp": float(time.time()),
        }

    def _update_data(self) -> None:
        """
        Update internal data.

        - If camera frame exists: run segmentation and update _data/_segmented_image/_classes
        - If no frame: still update _data with placeholder keys (for tests / contract)
        """
        frame, frame_ts = self._get_frame_bgr()

        # always set timestamp for heartbeat / tests
        ts = float(time.time())

        if frame is None:
            # placeholder update (keeps contract)
            self._segmented_image = None
            self._classes = None
            self._set_data({
                "segmented_image": None,
                "classes": None,
                "timestamp": ts,
            })
            return

        out = self.process_frame(frame)

        # Keep legacy fields
        self._segmented_image = out.get("segmented_image")
        self._classes = out.get("classes")

        if frame_ts is not None:
            out["timestamp"] = frame_ts
        elif "timestamp" not in out:
            out["timestamp"] = ts

        self._set_data(out)

    def _cleanup_cuda_ctx(self) -> None:
        if self._cuda_ctx is not None:
            try:
                self._cuda_ctx.pop()
            except Exception as e:
                logging.warning(f"CUDA context cleanup failed: {e}")
            self._cuda_ctx = None

    def _run(self) -> None:
        # init heavy objects
        try:
            self._ensure_loaded()
        except Exception:
            logging.exception("SegmentationProvider init failed")
            self._set_data(None)
            self.running = False
            self._cleanup_cuda_ctx()
            return

        try:
            while self.running and not self._stop_evt.is_set():
                t_loop = time.time()
                try:
                    self._update_data()
                except Exception as e:
                    logging.error(f"Error in SegmentationProvider loop: {e}")
                    self._set_data(None)
                    self._segmented_image = None
                    self._classes = None

                dt = time.time() - t_loop
                sleep_t = self._period_s - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)
        finally:
            self._cleanup_cuda_ctx()
