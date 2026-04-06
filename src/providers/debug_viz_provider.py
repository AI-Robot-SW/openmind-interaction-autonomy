# debug_viz_provider.py

import cv2
import time
import logging
import threading
import numpy as np

from .singleton import singleton
from typing import Optional

from .rtk_provider import RtkProvider
from .uwb_provider import UwbProvider
from .bev_occupancy_grid_provider import BEVOccupancyGridProvider
from .dwa_route_provider import DwaRouteProvider
from .gnss_route_provider import GnssRouteProvider
from .navigation_provider import NavigationProvider
from .realsense_camera_provider import RealSenseCameraProvider
from .segmentation_provider import SegmentationProvider


_DWA_M2PX = 55

_RTK_FIX_LABEL = {0: "none", 1: "float", 2: "fix"}
_MODE_COLOR = {
    "DWA":  (0, 220, 80),     # green
    "STOP": (60, 60, 220),    # red
    "IDLE": (120, 120, 120),  # gray
}
_SEG_COLORS: list[tuple[int, int, int]] = [
    (40,  40,  40),
    (40, 180,  40),
    (200,  80,  40),
    (40,  40, 200),
    (40, 200, 200),
]


@singleton
class DebugVizProvider:
    """
    HW 디버그용 실시간 시각화 Provider.

    run.py와 같은 프로세스에서 동작하며, 이미 실행 중인 singleton provider들을
    읽어 6-패널 OpenCV 창을 표시한다. ESC로 창을 닫을 수 있다.

      ┌─────────────────┬─────────────────┬─────────────────┐
      │ 1: Camera RGB   │                 │ 4: DWA Local    │
      │                 │  2: BEV Grid    │                 │
      ├─────────────────┤  (tall panel)   ├─────────────────┤
      │ 3: Segmentation │                 │ 5: Status Text  │
      └─────────────────┴─────────────────┴─────────────────┘
    """

    WIN_TITLE: str = "HW Debug Visualizer"

    def __init__(
        self,
        panel_w: int = 320,
        panel_h: int = 240,
        target_fps: float = 30.0,
    ) -> None:
        self.panel_w = panel_w
        self.panel_h = panel_h
        self.target_fps = target_fps

        self.running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("DebugVizProvider already running")
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="DebugVizRunner")
        self._thread.start()

        logging.info("DebugVizProvider started")

    def stop(self) -> None:
        self.running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

        try:
            cv2.destroyWindow(self.WIN_TITLE)
        except Exception:
            pass

        logging.info("DebugVizProvider stopped")

    # ── main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        interval = 1.0 / self.target_fps
        try:
            cv2.namedWindow(self.WIN_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WIN_TITLE, self.panel_w * 3, self.panel_h * 2)
        except Exception as e:
            logging.warning("DebugVizProvider: failed to create OpenCV window: %s", e)
            self.running = False
            return

        while self.running:
            t0 = time.monotonic()
            try:
                frame = self._build_composite()
                cv2.imshow(self.WIN_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC — close window but keep running
                    logging.info("DebugVizProvider: window closed by ESC")
                    break
            except Exception as e:
                logging.warning("DebugVizProvider: render error: %s", e)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

        try:
            cv2.destroyWindow(self.WIN_TITLE)
        except Exception:
            pass
        self.running = False

    # ── composite ─────────────────────────────────────────────────────────────

    def _build_composite(self) -> np.ndarray:
        W, H = self.panel_w, self.panel_h
        col_left = np.vstack([
            self._panel_camera_rgb(W, H),
            self._panel_segmentation(W, H),
        ])
        col_mid = self._panel_bev(W, H * 2)
        col_right = np.vstack([
            self._panel_dwa_local(W, H),
            self._panel_status(W, H),
        ])
        return np.hstack([col_left, col_mid, col_right])

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _blank(W: int, H: int, color: tuple = (30, 30, 30)) -> np.ndarray:
        return np.full((H, W, 3), color, dtype=np.uint8)

    @staticmethod
    def _put(img: np.ndarray, text: str, pos: tuple, color=(200, 200, 200),
             scale: float = 0.40, thick: int = 1) -> None:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _title(self, img: np.ndarray, text: str, color=(200, 200, 200)) -> None:
        self._put(img, text, (4, 14), color)

    # ── Panel 1: Camera RGB ───────────────────────────────────────────────────

    def _panel_camera_rgb(self, W: int, H: int) -> np.ndarray:
        frame = RealSenseCameraProvider().data
        if frame is None:
            p = self._blank(W, H)
            self._title(p, "Camera RGB  [no data]")
            return p
        p = cv2.resize(frame.bgr, (W, H))
        self._put(p, f"RGB  {frame.camera_fps:.0f}fps",
                  (4, 14), color=(220, 220, 50))
        return p

    # ── Panel 2: BEV Grid ─────────────────────────────────────────────────────

    def _panel_bev(self, W: int, H: int) -> np.ndarray:
        bev_frame = BEVOccupancyGridProvider().data
        if bev_frame is None:
            p = self._blank(W, H)
            self._title(p, "BEV Grid  [no data]")
            return p

        p = cv2.resize(bev_frame.bev_image, (W, H))
        grid = bev_frame.occupancy_grid
        dwa_rec = DwaRouteProvider().data
        gnss_rec = GnssRouteProvider().data

        if grid.resolution > 0:
            scale_x = W / grid.width
            scale_y = H / grid.height
            px_per_m_x = scale_x / grid.resolution
            px_per_m_y = scale_y / grid.resolution
            rx, ry = W // 2, H - 1

            if gnss_rec is not None:
                gx = max(0, min(W - 1, int(rx - gnss_rec.dy * px_per_m_x)))
                gy = max(0, min(H - 1, int(ry - gnss_rec.dx * px_per_m_y)))
                cv2.line(p, (rx, ry), (gx, gy), (0, 160, 255), 2, cv2.LINE_AA)

            if dwa_rec is not None and dwa_rec.mode == "DWA":
                dwa_x = max(0, min(W - 1, int(rx - dwa_rec.dy_dwa * px_per_m_x)))
                dwa_y = max(0, min(H - 1, int(ry - dwa_rec.dx_dwa * px_per_m_y)))
                cv2.arrowedLine(p, (rx, ry), (dwa_x, dwa_y), (0, 255, 0), 2, tipLength=0.15, line_type=cv2.LINE_AA)

        self._put(p, f"BEV  {bev_frame.bev_fps:.0f}fps",
                  (4, 14), color=(220, 220, 50))
        return p

    # ── Panel 3: Segmentation ─────────────────────────────────────────────────

    def _panel_segmentation(self, W: int, H: int) -> np.ndarray:
        seg_frame = SegmentationProvider().data
        cam_frame = RealSenseCameraProvider().data

        if seg_frame is None:
            p = self._blank(W, H)
            self._title(p, "Segmentation  [no data]")
            return p

        lut = np.array(_SEG_COLORS, dtype=np.uint8)
        idx = np.clip(seg_frame.semantic_map.astype(np.int32), 0, len(lut) - 1)
        seg_colored = lut[idx]

        if cam_frame is not None:
            cam_resized = cv2.resize(cam_frame.bgr,
                                     (seg_colored.shape[1], seg_colored.shape[0]))
            blended = cv2.addWeighted(cam_resized, 0.45, seg_colored, 0.55, 0)
            p = cv2.resize(blended, (W, H))
        else:
            p = cv2.resize(seg_colored, (W, H))

        self._put(p, f"Seg  {seg_frame.segmentation_fps:.0f}fps",
                  (4, 14), color=(220, 220, 50))
        return p

    # ── Panel 4: DWA Local View ───────────────────────────────────────────────

    def _panel_dwa_local(self, W: int, H: int) -> np.ndarray:
        dwa_rec = DwaRouteProvider().data
        gnss_rec = GnssRouteProvider().data

        p = self._blank(W, H, (15, 15, 25))
        cx, cy = W // 2, H // 2

        AXIS = (55, 55, 55)
        for dist_m in (1, 2, 3):
            r = dist_m * _DWA_M2PX
            cv2.circle(p, (cx, cy), r, AXIS, 1)
            self._put(p, f"{dist_m}m",  (cx - r - 22, cy), color=AXIS, scale=0.30)
            self._put(p, f"-{dist_m}m", (cx + r + 2,  cy), color=AXIS, scale=0.30)
        cv2.line(p, (cx, 0), (cx, H), AXIS, 1)
        cv2.line(p, (0, cy), (W, cy), AXIS, 1)

        mode = dwa_rec.mode if dwa_rec is not None else "IDLE"
        body_color = _MODE_COLOR.get(mode, (120, 120, 120))
        cv2.circle(p, (cx, cy), 13, body_color, 2)

        if gnss_rec is not None:
            gx = int(cx - gnss_rec.dy * _DWA_M2PX)
            gy = int(cy - gnss_rec.dx * _DWA_M2PX)
            cv2.arrowedLine(p, (cx, cy), (gx, gy), (0, 160, 255), 1, tipLength=0.25)

        if dwa_rec is not None:
            dx = int(cx - dwa_rec.dy_dwa * _DWA_M2PX)
            dy = int(cy - dwa_rec.dx_dwa * _DWA_M2PX)
            cv2.arrowedLine(p, (cx, cy), (dx, dy), (0, 235, 80), 2, tipLength=0.25)

        lines: list[tuple[str, tuple]] = [
            ("DWA Local  (body-frame)", (180, 180, 180)),
            (f"mode: {mode}", body_color),
        ]
        if dwa_rec is not None:
            lines += [
                (f"dwa goal : ({dwa_rec.dx_dwa:+.2f}, {dwa_rec.dy_dwa:+.2f}) m", (200, 200, 200)),
                (f"clearance: {dwa_rec.best_clearance_m:.2f} m", (200, 200, 200)),
                (f"vx={dwa_rec.vx_cmd:.2f}  vyaw={dwa_rec.vyaw_cmd:.2f}", (200, 200, 200)),
            ]
            if dwa_rec.stop_reason != "none":
                lines.append((f"stop: {dwa_rec.stop_reason}", (80, 80, 220)))
        if gnss_rec is not None:
            lines.append(
                (f"gnss goal: ({gnss_rec.dx:+.2f}, {gnss_rec.dy:+.2f}) m", (80, 160, 255))
            )

        y_start = H - len(lines) * 14 - 4
        for i, (text, color) in enumerate(lines):
            self._put(p, text, (4, y_start + i * 14), color=color, scale=0.36)

        return p

    # ── Panel 5: Navigation Status ────────────────────────────────────────────

    def _panel_status(self, W: int, H: int) -> np.ndarray:
        p = self._blank(W, H, (10, 10, 15))

        nav_prov = NavigationProvider()
        rtk_rec = RtkProvider().data
        uwb_rec = UwbProvider().data
        gnss_prov = GnssRouteProvider()

        SECTION = (100, 200, 255)
        DIM = (80, 80, 80)
        OK = (80, 220, 80)
        WARN = (80, 80, 220)

        rows: list[tuple[str, tuple | None]] = []

        rows.append(("[ NAV ]", SECTION))
        if nav_prov.running:
            try:
                state = nav_prov.get_state()
                goal = nav_prov.get_active_goal() or "—"
                dist = nav_prov.get_remaining_distance()
                rows += [
                    (f" mode : {state.mode}", None),
                    (f" goal : {goal}", None),
                    (f" dist : {dist:.1f} m", None),
                    (f" v : vx={state.vx:.2f} vy={state.vy:.2f} vyaw={state.vyaw:.2f}", None),
                    (f" calib: {'YES' if state.heading_calibrated else 'NO (calibrating...)'}",
                     OK if state.heading_calibrated else WARN),
                ]
                if state.reached_goal:
                    rows.append(("GOAL REACHED", OK))
            except Exception as e:
                rows.append((f" err: {e}", WARN))
        else:
            rows.append((" not running", DIM))

        rows.append(("", None))

        rows.append(("[ PATH ]", SECTION))
        tracker = gnss_prov.tracker
        if tracker is not None:
            idx, total = tracker.progress
            rows.append((f" node {idx} / {total}", None))
            cur = tracker.current_node
            if cur is not None:
                node = cur.node
                if hasattr(node, "lat") and node.lat is not None:
                    rows.append((f" cur : {node.lat:.6f}, {node.lon:.6f}", None))
                elif hasattr(node, "x") and node.x is not None:
                    rows.append((f" cur : x={node.x:.2f} y={node.y:.2f} m", None))
        else:
            rows.append((" no active tracker", DIM))

        rows.append(("", None))

        rows.append(("[ RTK ]", SECTION))
        if rtk_rec is not None and rtk_rec.lat is not None:
            fix = int(rtk_rec.carrSoln or 0)
            fix_label = _RTK_FIX_LABEL.get(fix, str(fix))
            fix_color = {0: WARN, 1: (80, 200, 220), 2: OK}.get(fix, None)
            rows += [
                (f" {rtk_rec.lat:.7f}, {rtk_rec.lon:.7f}", None),
                (f" hAcc={rtk_rec.hAcc_m:.2f}m  RTK={fix_label}", fix_color),
            ]
        else:
            rows.append((" no data", DIM))

        rows.append(("", None))

        rows.append(("[ UWB ]", SECTION))
        if uwb_rec is not None and uwb_rec.x_m is not None:
            rows += [
                (f" x={uwb_rec.x_m:.2f} y={uwb_rec.y_m:.2f} z={uwb_rec.z_m:.2f} m", None),
                (f" quality={uwb_rec.quality_factor}", None),
            ]
        else:
            rows.append((" no data", DIM))

        y = 14
        LINE_H = 14
        for text, color in rows:
            if y > H - LINE_H:
                break
            if not text:
                y += 5
                continue
            self._put(p, text, (4, y), color=color or (200, 200, 200), scale=0.36)
            y += LINE_H

        return p
