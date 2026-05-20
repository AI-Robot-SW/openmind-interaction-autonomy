# test_bev_occupancy_grid_provider.py

"""
BEVOccupancyGridProvider — Hardware test script.

Tested APIs:
  start, stop, data (property)

Prerequisites:
  USB 연결을 통해 RealSense camera 연결
  TensorRT 엔진 파일이 engines/trt/ 에 존재

Usage:
  python system_hw_test/providers/test_bev_occupancy_grid_provider.py

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


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────

def occupancy_to_bgr(grid: np.ndarray) -> np.ndarray:
    """Occupancy grid values → BGR visualization.
    -1=unknown(dark gray), 0=free(green), 70=avoid/curb(red), 88=person(blue)
    """
    img = np.full((*grid.shape, 3), (40, 40, 40), dtype=np.uint8)
    img[grid ==  0] = (0,   255,   0)
    img[grid == 70] = (0,     0, 255)
    img[grid == 88] = (255,   0,   0)
    return img


def draw_panel(data: dict, scale: int = 8) -> np.ndarray:
    bev_img = data.bev_image
    occ_grid = data.occupancy_grid.data
    occ_vis = occupancy_to_bgr(np.rot90(np.flipud(occ_grid)))

    bev_vis = cv2.resize(
        bev_img,
        (bev_img.shape[1] * scale, bev_img.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    occ_vis = cv2.resize(
        occ_vis,
        (bev_img.shape[1] * scale, bev_img.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )

    combined = np.hstack([bev_vis, occ_vis])
    lines = [
        f"frame   : {data.frame_cnt}",
        f"latency : {data.latency_s * 1000.0:.1f} ms",
        f"bev FPS : {data.bev_fps:.1f}",
    ]

    def put(text: str, y: int) -> None:
        cv2.putText(combined, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        put(line, 24 + i * 24)

    return combined


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    RealSenseCameraProvider.reset()      # type: ignore[attr-defined]
    SegmentationProvider.reset()         # type: ignore[attr-defined]
    PointCloudProvider.reset()           # type: ignore[attr-defined]
    BEVOccupancyGridProvider.reset()     # type: ignore[attr-defined]

    camera_provider = RealSenseCameraProvider()
    seg_provider = SegmentationProvider()
    pc_provider = PointCloudProvider()
    bev_provider = BEVOccupancyGridProvider()
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
    print("  bev_provider.start()\n  OK")

    # -------------------------------------------------------------------------
    # Phase 2: Frame verification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: Frame verification\n{'='*60}")
    frame = bev_provider.data

    bev_img = frame.bev_image
    occ = frame.occupancy_grid
    occ_data = occ.data
    print(f"  bev_image shape : {bev_img.shape} dtype={bev_img.dtype}")
    print(f"  occupancy shape : {occ_data.shape} dtype={occ_data.dtype}")
    print(f"  latency         : {frame.latency_s * 1000.0:.2f} ms")
    print(f"  bev FPS         : {frame.bev_fps:.1f}")
    print(f"  frame_cnt       : {frame.frame_cnt}")

    if bev_img.shape != (bev_provider.width, bev_provider.height, 3):
        print(f"  FAIL: bev_image shape mismatch: {bev_img.shape}")
        return 1

    if occ_data.shape != (bev_provider.height, bev_provider.width):
        print(f"  FAIL: occupancy shape mismatch: {occ_data.shape}")
        return 1

    valid_values = np.isin(occ_data, [-1, 0, 70, 88]).all()
    if not valid_values:
        print(f"  FAIL: unexpected occupancy values: {np.unique(occ_data)}")
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
        data = bev_provider.data
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
    # Phase 4: Live visualization (BEV)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Live visualization\n{'='*60}")
    print("  Press 'q' or ESC in the window to quit.")
    window = "BEVOccupancyGridProvider HW Test  |  BEV + Occupancy  (q/ESC to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_cnt = -1
    while True:
        data = bev_provider.data
        if data is not None and data.frame_cnt != last_cnt:
            last_cnt = data.frame_cnt
            cv2.imshow(window, draw_panel(data))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print("  Visualization closed.")

    # -------------------------------------------------------------------------
    # Phase 5: Teardown
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 5: Teardown\n{'='*60}")
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
