# segmentation_provider.py

import cv2
import time
import yaml
import logging
import threading
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from .singleton import singleton

from providers.utils.gpu_worker import GPUWorker
from providers.utils.trt_utils.trt_engine import TRTEngine
from providers.realsense_camera_provider import RealSenseCameraProvider, CameraFrame


_BASE_DIR = Path(__file__).resolve().parent

_DEFAULT_ENGINE_PATH = _BASE_DIR / "engines" / "trt" / "ddrnet23_fp16_kist-v1-l8-indoor-concat-aug3-720k-pre-320k_1x480x640.engine"
_DEFAULT_LABELS_PATH = _BASE_DIR / "engines" / "labels" / "mapillary_vistas_ddrnet.yaml"


# DDRNet ImageNet 정규화 파라미터
_DDRNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DDRNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# BGR colormap: category index → (B, G, R)
# _SEMANTIC_COLORS = np.array([
#     [128, 128, 128],  # 0: unknown   — gray
#     [  0, 255,   0],  # 1: driveable — green
#     [255,   0,   0],  # 2: person    — blue (BGR)
#     [  0,   0, 255],  # 3: avoid     — red
#     [255, 255, 255],  # 4: curb      — white
# ], dtype=np.uint8)


@dataclass(frozen=True)
class SegmentationFrame:
    """
    t_monotonic      : 처리된 카메라 프레임의 t_monotonic (동기화 기준)
    semantic_map     : (H, W) uint8 — category 값 (0:unknown 1:driveable 2:person 3:avoid 4:curb)
    classes          : 이번 프레임에서 검출된 class ID 목록 (모델 출력 해상도 기준)
    latency_s        : 전처리 + 추론 + 후처리 소요 시간 (초)
    segmentation_fps : 1 / latency_s
    frame_cnt        : 대응하는 CameraFrame.frame_cnt — 동기화 기준
    """
    t_monotonic:      float
    semantic_map:     np.ndarray
    classes:          List[int]
    latency_s:        float
    segmentation_fps: float
    frame_cnt:        int


@singleton
class SegmentationProvider:
    """
    background thread에서 realsense camera frame을 읽어 DDRNet TensorRT 추론을 수행하고
    최신 결과를 data 프로퍼티로 노출. GPU 연산은 GPUWorker를 통해 실행
    """

    def __init__(self):
        self.camera_provider = RealSenseCameraProvider()

        self._engine: Optional[TRTEngine] = None
        self._class_lut: Optional[np.ndarray] = None
        self._gpu_worker: Optional[GPUWorker] = None

        self._data: Optional[SegmentationFrame] = None
        self._lock = threading.Lock()
        self.frame_event = threading.Event()  # 새 프레임 처리 완료 신호

        self.running = False
        self._thread: Optional[threading.Thread] = None

        self._last_cnt: int = -1
        self._last_frame_t: float = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.warning("SegmentationProvider already running")
            return

        self._gpu_worker = GPUWorker()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # 첫 프레임 도착까지 대기 — 반환 후 data가 항상 SegmentationFrame을 보장
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("SegmentationProvider: timed out waiting for first frame")

        logging.info("SegmentationProvider started")

    def stop(self) -> None:
        self.running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._engine is not None and self._gpu_worker is not None:
            self._gpu_worker.submit(self._engine.free).result()
            self._engine = None

        self._last_cnt = -1
        with self._lock:
            self._data = None

        logging.info("SegmentationProvider stopped")

    def _load_engine(self) -> None:
        """TRT 엔진과 LUT를 GPUWorker를 통해 최초 1회 로드"""
        if self._engine is not None:
            return

        self._engine = self._gpu_worker.submit(
            lambda: TRTEngine(_DEFAULT_ENGINE_PATH)
        ).result()

        with open(_DEFAULT_LABELS_PATH) as f:
            _labels = yaml.safe_load(f)
        self._class_lut = np.zeros(215, dtype=np.uint8)  # Mapillary Vistas 65 + indoor 150 classes
        for cid in _labels.get("driveable", []): self._class_lut[cid] = 1  # driveable
        for cid in _labels.get("person",    []): self._class_lut[cid] = 2  # person
        for cid in _labels.get("avoid",     []): self._class_lut[cid] = 3  # avoid
        for cid in _labels.get("curb",      []): self._class_lut[cid] = 4  # curb

    def _preprocess(self, frame: CameraFrame) -> np.ndarray:
        """
        DDRNet 전처리: BGR → RGB → resize → normalize → 1CHW float32
        """
        img = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - _DDRNET_MEAN) / _DDRNET_STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis])  # HWC → 1CHW, C-contiguous

    def _process_frame(self, frame: CameraFrame) -> SegmentationFrame:
        """CameraFrame 1개에 대해 segmentation 수행 후 결과 반환"""
        if self._engine is None:
            raise RuntimeError("Engine not loaded. Call start() first.")
        t0 = time.monotonic()

        pixel_values = self._preprocess(frame)
        outputs = self._gpu_worker.submit(
            lambda: self._engine.infer(pixel_values)
        ).result()

        # logits: [1, C, H, W] → argmax (모델 출력 크기, uint8)
        # full-size int32 predicted_map을 만들지 않고, LUT를 소 크기에서 먼저 적용
        predicted_class_map = np.argmax(outputs[0], axis=1)[0].astype(np.uint8)
        semantic_map_small = self._class_lut[predicted_class_map]

        # uint8 semantic map을 원본 해상도로 resize (int32보다 4배 가벼움)
        h, w = frame.bgr.shape[:2]
        sem = cv2.resize(semantic_map_small, (w, h), interpolation=cv2.INTER_NEAREST)

        latency_s = float(time.monotonic() - t0)

        now = time.monotonic()
        seg_fps = 1.0 / (now - self._last_frame_t) if self._last_frame_t > 0.0 else 0.0
        self._last_frame_t = now

        return SegmentationFrame(
            t_monotonic=float(frame.t_monotonic),
            semantic_map=sem,
            classes=np.unique(predicted_class_map).astype(int).tolist(),
            latency_s=latency_s,
            segmentation_fps=seg_fps,
            frame_cnt=frame.frame_cnt,
        )

    @property
    def data(self) -> Optional[SegmentationFrame]:
        """최신 SegmentationFrame. 첫 프레임 처리 전에는 None."""
        with self._lock:
            return self._data
        
    def _run(self) -> None:
        try:
            self._load_engine()
        except Exception:
            logging.exception("SegmentationProvider init failed")
            self.running = False
            self._engine = None
            return

        while self.running:
            try:
                signaled = self.camera_provider.frame_event.wait(timeout=0.1)
                if not signaled:
                    continue
                self.camera_provider.frame_event.clear()
                cam_frame = self.camera_provider.data
                if cam_frame is not None and cam_frame.frame_cnt != self._last_cnt:
                    self._last_cnt = cam_frame.frame_cnt
                    with self._lock:
                        self._data = self._process_frame(cam_frame)
                    self.frame_event.set()
            except Exception as e:
                logging.error(f"SegmentationProvider: run loop error: {e}")
                with self._lock:
                    self._data = None