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
from .path_follow_provider import PathFollowProvider
from .navigation_provider import NavigationProvider
from .kf_position_provider import KfPositionProvider
from .realsense_camera_provider import RealSenseCameraProvider
from .segmentation_provider import SegmentationProvider
from .pointcloud_provider import PointCloudProvider


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
        # cv2.putText 는 ASCII 만 지원 — non-ASCII 를 '?' 로 대체해 깨짐 방지
        text = text.encode("ascii", "replace").decode("ascii")
        # 검정 outline 먼저 → 밝은/어두운 배경 모두에서 가독성 보장
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,     thick,     cv2.LINE_AA)

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
        path_rec = PathFollowProvider().data

        if grid.resolution > 0:
            scale_x = W / grid.width
            scale_y = H / grid.height
            px_per_m_x = scale_x / grid.resolution
            px_per_m_y = scale_y / grid.resolution
            rx, ry = W // 2, H - 1

            if path_rec is not None:
                gx = max(0, min(W - 1, int(rx - path_rec.dy * px_per_m_x)))
                gy = max(0, min(H - 1, int(ry - path_rec.dx * px_per_m_y)))
                cv2.line(p, (rx, ry), (gx, gy), (0, 160, 255), 2, cv2.LINE_AA)

            if dwa_rec is not None and dwa_rec.mode == "DWA":
                dwa_x = max(0, min(W - 1, int(rx - dwa_rec.dy_dwa * px_per_m_x)))
                dwa_y = max(0, min(H - 1, int(ry - dwa_rec.dx_dwa * px_per_m_y)))
                cv2.arrowedLine(p, (rx, ry), (dwa_x, dwa_y), (0, 255, 0), 2, tipLength=0.15, line_type=cv2.LINE_AA)

        self._put(p, f"BEV  {bev_frame.bev_fps:.0f}fps",
                  (4, 14), color=(220, 220, 50))
        pc_frame = PointCloudProvider().data
        if pc_frame is not None:
            self._put(p, f"PC   {pc_frame.pointcloud_fps:.0f}fps",
                      (4, 28), color=(220, 220, 50))
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

    # ── Panel 4: DWA Local View + Navigation Nodes ───────────────────────────

    def _panel_dwa_local(self, W: int, H: int) -> np.ndarray:
        dwa_rec = DwaRouteProvider().data
        path_rec = PathFollowProvider().data
        kf_rec   = KfPositionProvider().data

        p = self._blank(W, H, (15, 15, 25))
        cx, cy = W // 2, H // 2

        AXIS = (55, 55, 55)
        for dist_m in (1, 2, 3):
            r = dist_m * _DWA_M2PX
            cv2.circle(p, (cx, cy), r, AXIS, 1)
            self._put(p, f"+{dist_m}m", (cx - r - 25, cy), color=AXIS, scale=0.30)
            self._put(p, f"-{dist_m}m", (cx + r + 2,  cy), color=AXIS, scale=0.30)
        cv2.line(p, (cx, 0), (cx, H), AXIS, 1)
        cv2.line(p, (0, cy), (W, cy), AXIS, 1)

        mode = dwa_rec.mode if dwa_rec is not None else "IDLE"
        body_color = _MODE_COLOR.get(mode, (120, 120, 120))
        cv2.circle(p, (cx, cy), 13, body_color, 2)

        # ── 경로 노드 오버레이 ─────────────────────────────────
        # KF에서 로봇 world 위치와 heading을 구해 각 노드를 body frame으로 변환해 그린다.
        robot_wx = robot_wy = robot_th = None
        cos_t = sin_t = 0.0
        rtk_origin = None

        if kf_rec is not None:
            pf_prov = PathFollowProvider()
            pf_rec_local = pf_prov.data

            # 활성 frame에 따라 로봇 world 좌표 결정
            frame = pf_rec_local.frame if pf_rec_local else ""
            use_uwb = (frame != "wgs84" and kf_rec.uwb_ready
                       and kf_rec.uwb_x_m is not None and kf_rec.uwb_theta_rad is not None)
            use_rtk = (not use_uwb and kf_rec.rtk_ready
                       and kf_rec.rtk_lat is not None and kf_rec.rtk_theta_rad is not None)

            robot_wx = robot_wy = robot_th = None
            rtk_origin = KfPositionProvider().rtk_ekf.origin if use_rtk else None

            if use_uwb:
                robot_wx, robot_wy, robot_th = (
                    kf_rec.uwb_x_m, kf_rec.uwb_y_m, kf_rec.uwb_theta_rad
                )
            elif use_rtk and rtk_origin:
                from .utils.geo_utils import latlon_to_enu
                robot_wx, robot_wy = latlon_to_enu(
                    kf_rec.rtk_lat, kf_rec.rtk_lon, rtk_origin[0], rtk_origin[1]
                )
                robot_th = kf_rec.rtk_theta_rad

            if robot_wx is not None:
                import math
                cos_t = math.cos(robot_th)
                sin_t = math.sin(robot_th)

                with pf_prov._lock:
                    trackers    = pf_prov._trackers
                    active_idx  = pf_prov._active_idx

                for seg_i, tracker in enumerate(trackers):
                    passed_cnt = tracker._idx
                    for node_i, ref in enumerate(tracker.path):
                        graph = tracker._loader.get_graph(ref[0])
                        if graph is None:
                            continue
                        nd = graph.nodes.get(ref[1])
                        if nd is None:
                            continue

                        # 노드 world 좌표
                        if graph.coordinate_frame == "wgs84":
                            if nd.lat is None or rtk_origin is None:
                                continue
                            from .utils.geo_utils import latlon_to_enu as _l2e
                            nx, ny = _l2e(nd.lat, nd.lon, rtk_origin[0], rtk_origin[1])
                        else:
                            if nd.x is None:
                                continue
                            nx, ny = float(nd.x), float(nd.y)

                        # world → body frame
                        dx_w = nx - robot_wx
                        dy_w = ny - robot_wy
                        bfwd  =  cos_t * dx_w + sin_t * dy_w
                        bleft = -sin_t * dx_w + cos_t * dy_w

                        # body frame → pixel
                        px_ = int(cx - bleft * _DWA_M2PX)
                        py_ = int(cy - bfwd  * _DWA_M2PX)

                        if not (0 <= px_ < W and 0 <= py_ < H):
                            continue

                        # 색상: 통과=회색, 현재 목표=빨강, 미통과=파랑
                        is_passed  = (seg_i < active_idx or
                                      (seg_i == active_idx and node_i < passed_cnt))
                        is_current = (seg_i == active_idx and node_i == passed_cnt)

                        if is_current:
                            color_n, radius = (0, 30, 255), 8
                        elif is_passed:
                            color_n, radius = (110, 110, 110), 4
                        else:
                            color_n, radius = (0, 165, 255), 6

                        cv2.circle(p, (px_, py_), radius, color_n, -1)
                        if is_current:
                            cv2.circle(p, (px_, py_), radius + 3, (0, 80, 255), 2)
                            # 목표 노드 ID 표시
                            lbl = f"#{ref[1]}"
                            self._put(p, lbl, (px_ + 8, py_), color=(80, 160, 255), scale=0.30)

        # ── raw 센서 위치를 X 마커로 표시 ─────────────────────
        if robot_wx is not None:
            import math as _math

            def _draw_raw_x(world_x: float, world_y: float,
                            color: tuple, size: int = 5) -> None:
                dx_w = world_x - robot_wx
                dy_w = world_y - robot_wy
                bfwd_  =  cos_t * dx_w + sin_t * dy_w
                bleft_ = -sin_t * dx_w + cos_t * dy_w
                qx = int(cx - bleft_ * _DWA_M2PX)
                qy = int(cy - bfwd_  * _DWA_M2PX)
                if 0 <= qx < W and 0 <= qy < H:
                    cv2.drawMarker(p, (qx, qy), color,
                                   markerType=cv2.MARKER_TILTED_CROSS,
                                   markerSize=size * 2, thickness=2,
                                   line_type=cv2.LINE_AA)

            # UWB raw
            uwb_raw = UwbProvider().data
            if uwb_raw is not None and uwb_raw.x_m is not None:
                _draw_raw_x(uwb_raw.x_m, uwb_raw.y_m, (30, 30, 255), size=7)  # 진한 파랑 X

            # RTK raw (ENU 변환)
            rtk_raw = RtkProvider().data
            if rtk_raw is not None and rtk_raw.lat is not None and rtk_origin is not None:
                from .utils.geo_utils import latlon_to_enu as _l2e2
                rx_enu, ry_enu = _l2e2(rtk_raw.lat, rtk_raw.lon,
                                        rtk_origin[0], rtk_origin[1])
                _draw_raw_x(rx_enu, ry_enu, (0, 230, 230), size=7)  # 진한 청록 X

        # ── KF yaw 기반 heading 화살표 ───────────────────────
        if kf_rec is not None:
            import math as _math
            theta = None
            yaw_src = ""
            if kf_rec.uwb_ready and kf_rec.uwb_theta_rad is not None:
                theta = kf_rec.uwb_theta_rad
                yaw_src = "UWB"
            elif kf_rec.rtk_ready and kf_rec.rtk_theta_rad is not None:
                theta = kf_rec.rtk_theta_rad
                yaw_src = "RTK"

            if theta is not None:
                # body frame 기준: forward = 화면 위, left = 화면 왼쪽
                # heading은 world frame 각도이므로 body frame에서는 항상 정면(위쪽)으로 그린다
                arrow_len = int(1.0 * _DWA_M2PX)
                cv2.arrowedLine(
                    p, (cx, cy), (cx, cy - arrow_len),
                    (0, 255, 160), 2, tipLength=0.20, line_type=cv2.LINE_AA,
                )
                self._put(p, f"{_math.degrees(theta):+.1f}deg ({yaw_src})",
                          (cx + 4, cy - arrow_len + 4), color=(0, 220, 130), scale=0.30)

        lines: list[tuple[str, tuple]] = [
            ("Nav Map  (body-frame)", (180, 180, 180)),
        ]
        if kf_rec is not None:
            cal_str = ""
            if kf_rec.uwb_ready:
                cal_str += f"UWB {'yaw OK' if kf_rec.uwb_yaw_calibrated else 'yaw...'}"
            if kf_rec.rtk_ready:
                cal_str += f"  RTK {'yaw OK' if kf_rec.rtk_yaw_calibrated else 'yaw...'}"
            if cal_str:
                lines.append((cal_str, (80, 220, 80)))
        if dwa_rec is not None:
            lines += [
                (f"mode: {dwa_rec.mode}  vx={dwa_rec.vx_cmd:.2f}  vyaw={dwa_rec.vyaw_cmd:.2f}",
                 _MODE_COLOR.get(dwa_rec.mode, (120, 120, 120))),
            ]
            if dwa_rec.stop_reason != "none":
                lines.append((f"stop: {dwa_rec.stop_reason}", (80, 80, 220)))
        if path_rec is not None:
            lines.append(
                (f"path: ({path_rec.dx:+.2f}, {path_rec.dy:+.2f}) m  "
                 f"[{path_rec.node_idx}/{path_rec.node_total}]", (80, 160, 255))
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
        path_prov = PathFollowProvider()

        SECTION = (100, 200, 255)
        DIM = (80, 80, 80)
        OK = (80, 220, 80)
        WARN = (80, 80, 220)

        rows: list[tuple[str, tuple | None]] = []

        rows.append(("[ NAV ]", SECTION))
        if nav_prov.running:
            try:
                state = nav_prov.get_state()
                goal = nav_prov.get_active_goal() or "(none)"
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
        path_status = path_prov.data
        if path_status is not None:
            rows.append((f" node {path_status.node_idx} / {path_status.node_total}", None))
            rows.append((f" dx={path_status.dx:+.2f} dy={path_status.dy:+.2f} m", None))
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
