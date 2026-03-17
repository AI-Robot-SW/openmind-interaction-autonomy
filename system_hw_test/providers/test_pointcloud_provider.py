# test_pointcloud_provider.py

"""
PointCloudProvider — Hardware test script.

Tested APIs:
  start, stop, data (property)

Prerequisites:
  USB 연결을 통해 RealSense camera 연결
  TensorRT 엔진 파일이 engines/trt/ 에 존재

Usage:
  python system_hw_test/providers/test_pointcloud_provider.py

Controls (visualization window에서):
  q / ESC  — quit
"""

from __future__ import annotations

import sys
import time
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import cv2
import numpy as np

from providers.realsense_camera_provider import RealSenseCameraProvider
from providers.segmentation_provider import SegmentationProvider
from providers.pointcloud_provider import PointCloudProvider, PointCloudFrame


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────


@dataclass
class CounterStats:
    first_cnt: Optional[int] = None
    last_cnt: Optional[int] = None
    observed_frames: int = 0
    missed_by_poll: int = 0

    def update(self, frame_cnt: Optional[int]) -> None:
        if frame_cnt is None:
            return

        if self.first_cnt is None:
            self.first_cnt = frame_cnt
            self.last_cnt = frame_cnt
            self.observed_frames = 1
            return

        assert self.last_cnt is not None
        if frame_cnt == self.last_cnt:
            return

        if frame_cnt > self.last_cnt + 1:
            self.missed_by_poll += frame_cnt - self.last_cnt - 1

        self.last_cnt = frame_cnt
        self.observed_frames += 1


def draw_bev(
    frame: PointCloudFrame,
    canvas_size: int = 600,
    range_m: float = 5.0,
) -> np.ndarray:
    """
    포인트 클라우드를 위에서 내려다본 BEV(Bird-Eye View) 이미지로 렌더링.
    X=오른쪽, Z=앞쪽 기준으로 canvas 중앙 하단을 원점으로 사용.
    """
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    pts = frame.points  # (N, 4): X, Y, Z, rgb_packed

    if len(pts) == 0:
        return img

    x = pts[:, 0]
    z = pts[:, 2]
    rgb_packed = pts[:, 3].view(np.uint32)
    r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_packed >>  8) & 0xFF).astype(np.uint8)
    b = ( rgb_packed        & 0xFF).astype(np.uint8)

    # canvas 좌표 변환: 원점 = 하단 중앙
    cx = (x / range_m * (canvas_size / 2) + canvas_size / 2).astype(int)
    cy = (canvas_size - z / range_m * canvas_size).astype(int)

    mask = (cx >= 0) & (cx < canvas_size) & (cy >= 0) & (cy < canvas_size)
    cx, cy = cx[mask], cy[mask]
    r, g, b = r[mask], g[mask], b[mask]

    img[cy, cx] = np.stack([b, g, r], axis=1)

    lines = [
        f"points : {len(pts):,}",
        f"latency: {frame.latency_s * 1000:.1f} ms",
        f"pc FPS : {frame.pointcloud_fps:.1f}",
        f"frame  : {frame.frame_cnt}",
    ]

    def put(text: str, y: int) -> None:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    for i, line in enumerate(lines):
        put(line, 24 + i * 24)

    return img


def safe_frame_cnt(frame: object) -> Optional[int]:
    return None if frame is None else int(frame.frame_cnt)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    RealSenseCameraProvider.reset()  # type: ignore[attr-defined]
    SegmentationProvider.reset()     # type: ignore[attr-defined]
    PointCloudProvider.reset()       # type: ignore[attr-defined]

    camera_provider = RealSenseCameraProvider()
    seg_provider    = SegmentationProvider()
    pc_provider     = PointCloudProvider()
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
    print("  pc_provider.start()\n  OK")

    # -------------------------------------------------------------------------
    # Phase 2: Frame verification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: Frame verification\n{'='*60}")
    frame = pc_provider.data
    assert frame is not None

    print(f"  points shape : {frame.points.shape}  dtype={frame.points.dtype}")
    print(f"  point count  : {len(frame.points):,}")
    print(f"  latency      : {frame.latency_s * 1000:.2f} ms")
    print(f"  pc FPS       : {frame.pointcloud_fps:.1f}")
    print(f"  frame_cnt    : {frame.frame_cnt}")

    if frame.points.ndim != 2 or frame.points.shape[1] != 4:
        print(f"  FAIL: points shape이 (N, 4)가 아님: {frame.points.shape}")
        pc_provider.stop()
        seg_provider.stop()
        camera_provider.stop()
        return 1

    if len(frame.points) == 0:
        print("  WARN: point count가 0 — range_max 또는 depth 확인 필요")

    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 3: Latency stats (3초간 수집)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 3: Latency + sync stats (3s)\n{'='*60}")
    latencies: list[float] = []
    cam_stats = CounterStats()
    seg_stats = CounterStats()
    pc_stats = CounterStats()
    last_pc_cnt = -1
    sync_match_polls = 0
    sync_mismatch_polls = 0
    t_end = time.monotonic() + 3.0

    while time.monotonic() < t_end:
        cam_frame = camera_provider.data
        seg_frame = seg_provider.data
        pc_frame = pc_provider.data

        cam_cnt = safe_frame_cnt(cam_frame)
        seg_cnt = safe_frame_cnt(seg_frame)
        pc_cnt = safe_frame_cnt(pc_frame)

        cam_stats.update(cam_cnt)
        seg_stats.update(seg_cnt)
        pc_stats.update(pc_cnt)

        if cam_cnt is not None and seg_cnt is not None:
            if cam_cnt == seg_cnt:
                sync_match_polls += 1
            else:
                sync_mismatch_polls += 1

        if pc_frame is not None and pc_frame.frame_cnt != last_pc_cnt:
            last_pc_cnt = pc_frame.frame_cnt
            latencies.append(pc_frame.latency_s * 1000.0)
        time.sleep(0.005)

    if latencies:
        arr = np.array(latencies)
        print(f"  samples : {len(arr)}")
        print(f"  mean    : {arr.mean():.2f} ms")
        print(f"  p95     : {np.percentile(arr, 95):.2f} ms")
        print(f"  max     : {arr.max():.2f} ms")
    else:
        print("  WARN: latency 샘플 수집 실패")

    cam_pc_end_lag = (
        cam_stats.last_cnt - pc_stats.last_cnt
        if cam_stats.last_cnt is not None and pc_stats.last_cnt is not None
        else None
    )
    seg_pc_end_lag = (
        seg_stats.last_cnt - pc_stats.last_cnt
        if seg_stats.last_cnt is not None and pc_stats.last_cnt is not None
        else None
    )

    print(
        f"  camera observed      : {cam_stats.observed_frames}"
        f" (poll-missed gaps: {cam_stats.missed_by_poll})"
    )
    print(
        f"  segmentation observed: {seg_stats.observed_frames}"
        f" (poll-missed gaps: {seg_stats.missed_by_poll})"
    )
    print(
        f"  pointcloud observed  : {pc_stats.observed_frames}"
        f" (poll-missed gaps: {pc_stats.missed_by_poll})"
    )
    print(
        f"  cam/seg sync polls   : match={sync_match_polls} mismatch={sync_mismatch_polls}"
    )
    print(
        f"  end lag              : cam-pc={cam_pc_end_lag} seg-pc={seg_pc_end_lag}"
    )
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 4: Live visualization (BEV)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Live visualization (BEV)\n{'='*60}")
    print("  Press 'q' or ESC in the window to quit.")

    window = "PointCloudProvider HW Test  |  BEV  (q/ESC to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    last_cnt = -1
    while True:
        f = pc_provider.data
        if f is not None and f.frame_cnt != last_cnt:
            last_cnt = f.frame_cnt
            bev = draw_bev(f)
            cv2.imshow(window, bev)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()
    print("  Visualization closed.")

    # -------------------------------------------------------------------------
    # Phase 5: Teardown
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 5: Teardown\n{'='*60}")
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
