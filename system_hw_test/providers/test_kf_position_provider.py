# test_kf_position_provider.py
"""
KfPositionProvider — Hardware test script with matplotlib visualization.

두 EKF 파이프라인(UWB + RTK)의 상태를 실시간으로 시각화한다.
  - UWB subplot : raw UWB 측정값, EKF 추정 궤적, heading 화살표, b_θ 수렴 상태
  - RTK subplot : raw GNSS 측정값 (ENU), EKF 추정 궤적, heading 화살표, hAcc / b_θ 상태

Usage:
    python system_hw_test/providers/test_kf_position_provider.py
    python system_hw_test/providers/test_kf_position_provider.py --uwb-port /dev/uwb
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import logging
import threading
from collections import deque
from typing import Optional

sys.path.insert(0, __file__.rsplit("/system_hw_test", 1)[0])

from src.providers.kf_position_provider import KfPositionProvider, KfPositionRecord
from src.providers.unitree_go2_provider import UnitreeGo2Provider
from src.providers.uwb_provider import UwbProvider
from src.providers.rtk_provider import RtkProvider
from src.providers.utils.geo_utils import latlon_to_enu

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_log = logging.getLogger(__name__)

# ── 시각화 파라미터 ────────────────────────────────────────────
_TRAIL_LEN: int = 500
_MARGIN_M: float = 3.0
_ARROW_LEN_M: float = 0.5
_REFRESH_MS: int = 200


# ── 데이터 버퍼 ───────────────────────────────────────────────
class _Buffer:
    def __init__(self) -> None:
        self.ekf_x: deque[float] = deque(maxlen=_TRAIL_LEN)
        self.ekf_y: deque[float] = deque(maxlen=_TRAIL_LEN)
        self.raw_x: deque[float] = deque(maxlen=_TRAIL_LEN)
        self.raw_y: deque[float] = deque(maxlen=_TRAIL_LEN)
        self._lock = threading.Lock()

    def push_ekf(self, x: float, y: float) -> None:
        with self._lock:
            self.ekf_x.append(x)
            self.ekf_y.append(y)

    def push_raw(self, x: float, y: float) -> None:
        with self._lock:
            self.raw_x.append(x)
            self.raw_y.append(y)

    def snapshot(self):
        with self._lock:
            return (
                list(self.ekf_x), list(self.ekf_y),
                list(self.raw_x), list(self.raw_y),
            )


def _auto_lim(ax, xs: list[float], ys: list[float]) -> None:
    if not xs:
        return
    xmin, xmax = min(xs) - _MARGIN_M, max(xs) + _MARGIN_M
    ymin, ymax = min(ys) - _MARGIN_M, max(ys) + _MARGIN_M
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max((xmax - xmin) / 2, (ymax - ymin) / 2, 1.0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def _make_fig(
    kf: KfPositionProvider,
    uwb_buf: _Buffer,
    rtk_buf: _Buffer,
    stop_evt: threading.Event,
) -> None:
    fig, (ax_uwb, ax_rtk) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("KfPositionProvider — Dual EKF Live View", fontsize=13)

    # ── UWB subplot ──────────────────────────────────────────────
    ax_uwb.set_title("UWB + Odom EKF", fontsize=11)
    ax_uwb.set_xlabel("x (m)")
    ax_uwb.set_ylabel("y (m)")
    ax_uwb.set_aspect("equal")
    ax_uwb.grid(True, linestyle="--", alpha=0.4)

    uwb_raw_sc, = ax_uwb.plot([], [], "rx", ms=4, alpha=0.5, label="raw UWB")
    uwb_ekf_ln, = ax_uwb.plot([], [], "b-", lw=1.2, label="EKF path")
    uwb_cur_sc, = ax_uwb.plot([], [], "bo", ms=8, zorder=6, label="EKF pos")
    uwb_arrow = ax_uwb.quiver(
        [0], [0], [0], [0],
        angles="xy", scale_units="xy", scale=1,
        color="royalblue", width=0.006, headwidth=4, headlength=5,
        zorder=6, visible=False,
    )
    ax_uwb.legend(loc="upper left", fontsize=8)
    uwb_txt = ax_uwb.text(
        0.02, 0.98, "", transform=ax_uwb.transAxes,
        va="top", ha="left", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    # ── RTK subplot ──────────────────────────────────────────────
    ax_rtk.set_title("RTK/GNSS + Odom EKF  (ENU from origin)", fontsize=11)
    ax_rtk.set_xlabel("East (m)")
    ax_rtk.set_ylabel("North (m)")
    ax_rtk.set_aspect("equal")
    ax_rtk.grid(True, linestyle="--", alpha=0.4)

    rtk_raw_sc, = ax_rtk.plot([], [], "rx", ms=4, alpha=0.5, label="raw GNSS")
    rtk_ekf_ln, = ax_rtk.plot([], [], "b-", lw=1.2, label="EKF path")
    rtk_cur_sc, = ax_rtk.plot([], [], "bo", ms=8, zorder=6, label="EKF pos")
    rtk_arrow_cal = ax_rtk.quiver(
        [0], [0], [0], [0],
        angles="xy", scale_units="xy", scale=1,
        color="royalblue", width=0.006, headwidth=4, headlength=5,
        zorder=6, visible=False,
    )
    rtk_arrow_uncal = ax_rtk.quiver(
        [0], [0], [0], [0],
        angles="xy", scale_units="xy", scale=1,
        color="darkorange", width=0.006, headwidth=4, headlength=5,
        zorder=6, visible=False,
    )
    ax_rtk.legend(loc="upper left", fontsize=8)
    rtk_txt = ax_rtk.text(
        0.02, 0.98, "", transform=ax_rtk.transAxes,
        va="top", ha="left", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    def _update(_frame):
        if stop_evt.is_set():
            return

        rec: Optional[KfPositionRecord] = kf.data
        uwb_ekf = kf.uwb_ekf
        rtk_ekf = kf.rtk_ekf
        uwb_res = kf.uwb_residual

        # ── UWB ──────────────────────────────────────────────────
        ex, ey, rx, ry = uwb_buf.snapshot()
        uwb_raw_sc.set_data(rx, ry)
        uwb_ekf_ln.set_data(ex, ey)
        uwb_cur_sc.set_data([ex[-1]] if ex else [], [ey[-1]] if ey else [])

        if ex and rec and rec.uwb_ready and rec.uwb_theta_rad is not None:
            dx = _ARROW_LEN_M * math.cos(rec.uwb_theta_rad)
            dy = _ARROW_LEN_M * math.sin(rec.uwb_theta_rad)
            uwb_arrow.set_offsets([[ex[-1], ey[-1]]])
            uwb_arrow.set_UVC([dx], [dy])
            uwb_arrow.set_visible(True)
        else:
            uwb_arrow.set_visible(False)

        _auto_lim(ax_uwb, ex + rx, ey + ry)

        cal_mark = "✓" if (rec and rec.uwb_yaw_calibrated) else "…"
        ready_str = "READY" if (rec and rec.uwb_ready) else "WAIT"
        stable_str = "stable" if (rec and rec.uwb_signal_stable) else "unstable"
        b_theta_deg = math.degrees(uwb_ekf.b_theta_rad) if uwb_ekf.initialized else float("nan")
        std_bias = uwb_ekf.std_bias_deg if uwb_ekf.initialized else float("nan")
        std_xy = rec.uwb_std_xy_m if (rec and rec.uwb_std_xy_m is not None) else float("nan")
        residual_std = uwb_res.std_m

        uwb_txt.set_text(
            f"[UWB EKF]  {ready_str}\n"
            f"signal : {stable_str}  (res_std={residual_std:.3f}m)\n"
            f"pos std: {std_xy:.3f} m\n"
            f"b_θ    : {b_theta_deg:+.1f}° ±{std_bias:.1f}°  {cal_mark}"
        )

        # ── RTK ──────────────────────────────────────────────────
        ex, ey, rx, ry = rtk_buf.snapshot()
        rtk_raw_sc.set_data(rx, ry)
        rtk_ekf_ln.set_data(ex, ey)
        rtk_cur_sc.set_data([ex[-1]] if ex else [], [ey[-1]] if ey else [])

        if ex and rec and rec.rtk_ready and rec.rtk_theta_rad is not None:
            dx = _ARROW_LEN_M * math.cos(rec.rtk_theta_rad)
            dy = _ARROW_LEN_M * math.sin(rec.rtk_theta_rad)
            if rec.rtk_yaw_calibrated:
                rtk_arrow_cal.set_offsets([[ex[-1], ey[-1]]])
                rtk_arrow_cal.set_UVC([dx], [dy])
                rtk_arrow_cal.set_visible(True)
                rtk_arrow_uncal.set_visible(False)
            else:
                rtk_arrow_uncal.set_offsets([[ex[-1], ey[-1]]])
                rtk_arrow_uncal.set_UVC([dx], [dy])
                rtk_arrow_uncal.set_visible(True)
                rtk_arrow_cal.set_visible(False)
        else:
            rtk_arrow_cal.set_visible(False)
            rtk_arrow_uncal.set_visible(False)

        _auto_lim(ax_rtk, ex + rx, ey + ry)

        cal_mark = "✓" if (rec and rec.rtk_yaw_calibrated) else "…"
        ready_str = "READY" if (rec and rec.rtk_ready) else "WAIT"
        b_theta_deg = math.degrees(rtk_ekf.b_theta_rad) if rtk_ekf.initialized else float("nan")
        std_bias = rtk_ekf.std_bias_deg if rtk_ekf.initialized else float("nan")
        std_xy = rec.rtk_std_xy_m if (rec and rec.rtk_std_xy_m is not None) else float("nan")

        rtk_raw = kf._rtk.data
        hacc = rtk_raw.hAcc_m if rtk_raw is not None else None
        hacc_str = f"{hacc:.3f} m" if hacc is not None else "N/A"

        rtk_txt.set_text(
            f"[RTK EKF]  {ready_str}\n"
            f"hAcc   : {hacc_str}\n"
            f"pos std: {std_xy:.3f} m\n"
            f"b_θ    : {b_theta_deg:+.1f}° ±{std_bias:.1f}°  {cal_mark}"
        )

        return (
            uwb_raw_sc, uwb_ekf_ln, uwb_cur_sc, uwb_arrow, uwb_txt,
            rtk_raw_sc, rtk_ekf_ln, rtk_cur_sc,
            rtk_arrow_cal, rtk_arrow_uncal, rtk_txt,
        )

    ani = animation.FuncAnimation(
        fig, _update, interval=_REFRESH_MS, blit=False, cache_frame_data=False,
    )

    try:
        plt.tight_layout()
        plt.show()
    finally:
        stop_evt.set()
        ani.event_source.stop()


def main() -> int:
    p = argparse.ArgumentParser(description="KfPositionProvider dual EKF visualization")
    p.add_argument("--uwb-port",    default="/dev/uwb",  help="UWB 시리얼 포트")
    p.add_argument("--uwb-baud",    default=115200, type=int)
    p.add_argument("--ntrip-user",  default="dori0126",  help="NTRIP 사용자명")
    p.add_argument("--ntrip-pass",  default="ngii",      help="NTRIP 비밀번호")
    p.add_argument("--robot-iface", default="eno1",      help="Unitree 이더넷 인터페이스")
    args = p.parse_args()

    # ── Providers 시작 ────────────────────────────────────────────
    robot = UnitreeGo2Provider(channel=args.robot_iface)
    robot.start()

    UwbProvider.reset()   # type: ignore[attr-defined]
    uwb = UwbProvider(port=args.uwb_port, baud=args.uwb_baud)
    uwb.start()

    RtkProvider.reset()   # type: ignore[attr-defined]
    rtk = RtkProvider(user=args.ntrip_user, password=args.ntrip_pass)
    rtk.start()

    KfPositionProvider.reset()  # type: ignore[attr-defined]
    kf = KfPositionProvider()
    try:
        kf.start()
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    stop_evt = threading.Event()
    uwb_buf = _Buffer()
    rtk_buf = _Buffer()

    def _collect_loop() -> None:
        while not stop_evt.is_set():
            rec = kf.data
            if rec is None:
                time.sleep(0.02)
                continue

            rtk_ekf = kf.rtk_ekf
            origin = rtk_ekf.origin  # (lat, lon) or None

            # UWB raw + EKF
            uwb_data = kf._uwb.data
            if uwb_data is not None and uwb_data.x_m is not None:
                uwb_buf.push_raw(uwb_data.x_m, uwb_data.y_m)
            if rec.uwb_ready and rec.uwb_x_m is not None:
                uwb_buf.push_ekf(rec.uwb_x_m, rec.uwb_y_m)

            # RTK raw + EKF (EKF origin 기준 ENU)
            rtk_data = kf._rtk.data
            if origin and rtk_data is not None and rtk_data.lat is not None:
                rx, ry = latlon_to_enu(rtk_data.lat, rtk_data.lon, origin[0], origin[1])
                rtk_buf.push_raw(rx, ry)
            if rec.rtk_ready and origin and rec.rtk_lat is not None:
                ex, ey = latlon_to_enu(rec.rtk_lat, rec.rtk_lon, origin[0], origin[1])
                rtk_buf.push_ekf(ex, ey)

            time.sleep(0.02)

    collect_thread = threading.Thread(target=_collect_loop, daemon=True, name="VisCollect")
    collect_thread.start()

    _log.info("EKF 초기화 대기 중... (RTK hAcc ≤ 1m 혹은 UWB 신호 안정화 필요)")

    try:
        _make_fig(kf, uwb_buf, rtk_buf, stop_evt)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        kf.stop()
        rtk.stop()
        uwb.stop()
        robot.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
