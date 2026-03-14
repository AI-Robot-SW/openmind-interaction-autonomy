# test_segmentation_provider.py

"""
SegmentationProvider — Hardware test script.

Tested APIs:
  start, stop, data (property)

Prerequisites:
  USB 연결을 통해 realsense camera를 연결
  TensorRT 엔진 파일이 engines/trt/ 에 존재

Usage:
  python system_hw_test/providers/test_segmentation_provider.py

Controls (visualization window에서):
  q / ESC  — quit
"""

from __future__ import annotations

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import cv2
import numpy as np

from providers.realsense_camera_provider import RealSenseCameraProvider
from providers.segmentation_provider import SegmentationProvider, SegmentationFrame

# SegmentationFrame.semantic_map category → BGR color (segmentation_provider.py의 _SEMANTIC_COLORS와 동일)
_SEMANTIC_COLORS = np.array([
    [128, 128, 128],  # 0: unknown   — gray
    [  0, 255,   0],  # 1: driveable — green
    [255,   0,   0],  # 2: person    — blue (BGR)
    [  0,   0, 255],  # 3: avoid     — red
    [255, 255, 255],  # 4: curb      — white
], dtype=np.uint8)


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────

def semantic_to_bgr(semantic_map: np.ndarray) -> np.ndarray:
    """(H, W) uint8 category map → (H, W, 3) BGR 컬러맵."""
    return _SEMANTIC_COLORS[np.clip(semantic_map, 0, len(_SEMANTIC_COLORS) - 1)]


def draw_overlay(
    color_img: np.ndarray,
    seg_frame: SegmentationFrame,
    alpha: float = 0.5,
) -> np.ndarray:
    """컬러 이미지와 semantic overlay를 가로로 이어붙인 이미지를 반환."""
    semantic_bgr = semantic_to_bgr(seg_frame.semantic_map).astype(np.uint8)
    blended = cv2.addWeighted(color_img, 1 - alpha, semantic_bgr, alpha, 0)

    class_names = {0: "unknown", 1: "driveable", 2: "person", 3: "avoid", 4: "curb"}
    detected = ", ".join(class_names.get(c, str(c)) for c in sorted(seg_frame.classes))

    lines = [
        f"latency: {seg_frame.latency_s * 1000:.1f} ms",
        f"seg FPS: {seg_frame.segmentation_fps:.1f}",
        f"classes: {detected}",
        f"frame:   {seg_frame.frame_cnt}",
    ]

    def put(img: np.ndarray, text: str, y: int) -> None:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),     3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0),   1, cv2.LINE_AA)

    color_disp = color_img.copy()
    for i, line in enumerate(lines):
        put(color_disp, line, 24 + i * 24)

    return np.hstack([color_disp, blended])


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    RealSenseCameraProvider.reset()  # type: ignore[attr-defined]
    SegmentationProvider.reset()     # type: ignore[attr-defined]

    camera_provider  = RealSenseCameraProvider()
    seg_provider     = SegmentationProvider()
    print("  Providers created")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 1: Start
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 1: Start\n{'='*60}")
    try:
        camera_provider.start()
    except RuntimeError as e:
        print(f"  FAIL (camera): {e}")
        return 1
    print("  camera_provider.start()")

    try:
        seg_provider.start()
    except RuntimeError as e:
        print(f"  FAIL (segmentation): {e}")
        camera_provider.stop()
        return 1
    print("  seg_provider.start()\n  OK")

    # -------------------------------------------------------------------------
    # Phase 2: Frame verification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: Frame verification\n{'='*60}")
    frame = seg_provider.data

    h, w = frame.semantic_map.shape[:2]
    cam_frame = camera_provider.data
    assert cam_frame is not None
    cam_h, cam_w = cam_frame.bgr.shape[:2]

    print(f"  semantic_map shape={frame.semantic_map.shape}  dtype={frame.semantic_map.dtype}")
    print(f"  camera bgr   shape=({cam_h}, {cam_w})")

    if (h, w) != (cam_h, cam_w):
        print(f"  FAIL: semantic_map 해상도({h}x{w})가 카메라({cam_h}x{cam_w})와 다름")
        seg_provider.stop()
        camera_provider.stop()
        return 1

    unique_vals = sorted(np.unique(frame.semantic_map).tolist())
    if not all(0 <= v <= 4 for v in unique_vals):
        print(f"  FAIL: semantic_map에 범위 밖 값 포함: {unique_vals}")
        seg_provider.stop()
        camera_provider.stop()
        return 1

    print(f"  classes detected: {frame.classes}")
    print(f"  latency:  {frame.latency_s * 1000:.2f} ms")
    print(f"  seg FPS:  {frame.segmentation_fps:.1f}")
    print(f"  frame_cnt: seg={frame.frame_cnt}  cam={cam_frame.frame_cnt}  (diff={cam_frame.frame_cnt - frame.frame_cnt}, 정상=1)")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 3: Latency stats (3초간 수집)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 3: Latency stats (3s)\n{'='*60}")
    latencies: list[float] = []
    last_cnt = -1
    t_end = time.monotonic() + 3.0

    while time.monotonic() < t_end:
        f = seg_provider.data
        if f is not None and f.frame_cnt != last_cnt:
            last_cnt = f.frame_cnt
            latencies.append(f.latency_s * 1000.0)
        time.sleep(0.005)

    if latencies:
        arr = np.array(latencies)
        print(f"  samples : {len(arr)}")
        print(f"  mean    : {arr.mean():.2f} ms")
        print(f"  p95     : {np.percentile(arr, 95):.2f} ms")
        print(f"  max     : {arr.max():.2f} ms")
    else:
        print("  WARN: latency 샘플 수집 실패")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 4: Live visualization
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Live visualization\n{'='*60}")
    print("  Press 'q' or ESC in the window to quit.")

    window = "SegmentationProvider HW Test  |  color + semantic overlay  (q/ESC to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_cnt = -1
    while True:
        cam_f  = camera_provider.data
        seg_f  = seg_provider.data

        if cam_f is None or seg_f is None:
            time.sleep(0.01)
            continue

        if seg_f.frame_cnt != last_cnt:
            last_cnt = seg_f.frame_cnt
            combined = draw_overlay(cam_f.bgr, seg_f)
            cv2.imshow(window, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    cv2.destroyAllWindows()
    print("  Visualization closed.")

    # -------------------------------------------------------------------------
    # Phase 5: Teardown
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 5: Teardown\n{'='*60}")
    seg_provider.stop()
    print("  seg_provider.stop()")
    camera_provider.stop()
    print("  camera_provider.stop()\n  OK")

    print("\n  All phases complete. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
