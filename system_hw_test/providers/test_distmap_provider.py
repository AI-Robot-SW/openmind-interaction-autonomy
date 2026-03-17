# test_distmap_provider.py

"""
DistMapProvider — Hardware test script.

Tested APIs:
  start, stop, data (property)

Prerequisites:
  USB 연결을 통해 RealSense camera 연결
  TensorRT 엔진 파일이 engines/trt/ 에 존재

Usage:
  python system_hw_test/providers/test_distmap_provider.py

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
from providers.segmentation_provider import SegmentationProvider
from providers.pointcloud_provider import PointCloudProvider
from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider
from providers.distmap_provider import DistMapProvider


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────

def occupancy_to_bgr(grid: np.ndarray) -> np.ndarray:
    """Occupancy grid values → BGR visualization."""
    lut = np.zeros((101, 3), dtype=np.uint8)
    lut[0]   = (0, 255, 0)      # free       — green
    lut[70]  = (255, 255, 255)  # avoid      — white
    lut[88]  = (255, 0, 0)      # person     — blue (BGR)
    lut[100] = (0, 0, 255)      # occupied   — red
    return lut[np.clip(grid.astype(np.int16), 0, 100)]


def distmap_to_bgr(dist: np.ndarray, max_dist: float) -> np.ndarray:
    """Distance map (float32, meters) → BGR heatmap.
    빨강=장애물(위험, 거리 0), 파랑=열린공간(안전, 거리 max_dist).
    """
    norm = np.clip(dist / max_dist, 0.0, 1.0)
    gray = 255 - (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def draw_panel(bev_frame, dist_frame, max_dist: float, scale: int = 8) -> np.ndarray:
    bev_img = bev_frame.bev_image
    occ_vis = occupancy_to_bgr(np.rot90(np.flipud(bev_frame.occupancy_grid.data)))
    dist_vis = distmap_to_bgr(np.rot90(np.flipud(dist_frame.dist)), max_dist)

    def upscale(img):
        return cv2.resize(
            img,
            (bev_img.shape[1] * scale, bev_img.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    combined = np.hstack([upscale(bev_img), upscale(occ_vis), upscale(dist_vis)])

    lines = [
        f"frame       : {dist_frame.frame_cnt}",
        f"dist latency: {dist_frame.latency_s * 1000.0:.1f} ms",
        f"dist FPS    : {dist_frame.distmap_fps:.1f}",
        f"bev FPS     : {bev_frame.bev_fps:.1f}",
    ]

    def put(text: str, y: int) -> None:
        cv2.putText(combined, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        put(line, 24 + i * 24)

    return combined


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    MAX_DIST = 5.0

    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    RealSenseCameraProvider.reset()   # type: ignore[attr-defined]
    SegmentationProvider.reset()      # type: ignore[attr-defined]
    PointCloudProvider.reset()        # type: ignore[attr-defined]
    BEVOccupancyGridProvider.reset()  # type: ignore[attr-defined]
    DistMapProvider.reset()           # type: ignore[attr-defined]

    camera_provider = RealSenseCameraProvider()
    seg_provider    = SegmentationProvider()
    pc_provider     = PointCloudProvider()
    bev_provider    = BEVOccupancyGridProvider()
    dist_provider   = DistMapProvider(max_dist=MAX_DIST)
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
    print("  seg_provider.start()")

    try:
        pc_provider.start()
    except RuntimeError as e:
        print(f"  FAIL (pointcloud): {e}")
        seg_provider.stop()
        camera_provider.stop()
        return 1
    print("  pc_provider.start()")

    try:
        bev_provider.start()
    except RuntimeError as e:
        print(f"  FAIL (bev): {e}")
        pc_provider.stop()
        seg_provider.stop()
        camera_provider.stop()
        return 1
    print("  bev_provider.start()")

    try:
        dist_provider.start()
        print("  dist_provider.start()")
    except RuntimeError as e:
        print(f"  FAIL (distmap): {e}")
        bev_provider.stop()
        pc_provider.stop()
        seg_provider.stop()
        camera_provider.stop()
        return 1
    print("  dist_provider.start()\n  OK")

    # -------------------------------------------------------------------------
    # Phase 2: Frame verification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: Frame verification\n{'='*60}")
    frame = dist_provider.data

    dist = frame.dist
    print(f"  dist shape    : {dist.shape} dtype={dist.dtype}")
    print(f"  dist range    : [{dist.min():.3f}, {dist.max():.3f}] m")
    print(f"  latency       : {frame.latency_s * 1000.0:.2f} ms")
    print(f"  distmap FPS   : {frame.distmap_fps:.1f}")
    print(f"  frame_cnt     : {frame.frame_cnt}")

    expected_shape = (bev_provider.height, bev_provider.width)
    if dist.shape != expected_shape:
        print(f"  FAIL: dist shape mismatch: {dist.shape} != {expected_shape}")
        return 1

    if dist.dtype != np.float32:
        print(f"  FAIL: dist dtype should be float32, got {dist.dtype}")
        return 1

    if dist.min() < 0.0 or dist.max() > MAX_DIST + 1e-4:
        print(f"  FAIL: dist values out of range [0, {MAX_DIST}]")
        return 1

    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 3: Latency stats (3초간 수집)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 3: Latency stats (3s)\n{'='*60}")
    latencies: list[float] = []
    last_cnt = -1
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        data = dist_provider.data
        if data is not None and data.frame_cnt != last_cnt:
            last_cnt = data.frame_cnt
            latencies.append(data.latency_s * 1000.0)
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
    # Phase 4: Live visualization (BEV | Occupancy | DistMap heatmap)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Live visualization\n{'='*60}")
    print("  Press 'q' or ESC in the window to quit.")
    window = "DistMapProvider HW Test  |  BEV | Occupancy | DistMap  (q/ESC to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_cnt = -1
    while True:
        dist_frame = dist_provider.data
        bev_frame  = bev_provider.data
        if (
            dist_frame is not None
            and bev_frame is not None
            and dist_frame.frame_cnt != last_cnt
        ):
            last_cnt = dist_frame.frame_cnt
            cv2.imshow(window, draw_panel(bev_frame, dist_frame, MAX_DIST))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print("  Visualization closed.")

    # -------------------------------------------------------------------------
    # Phase 5: Teardown
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 5: Teardown\n{'='*60}")
    dist_provider.stop()
    print("  dist_provider.stop()")
    bev_provider.stop()
    print("  bev_provider.stop()")
    pc_provider.stop()
    print("  pc_provider.stop()")
    seg_provider.stop()
    print("  seg_provider.stop()")
    camera_provider.stop()
    print("  camera_provider.stop()\n  OK")

    print("\n  All phases complete. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
