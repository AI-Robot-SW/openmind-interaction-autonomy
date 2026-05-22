# kf_position_provider.py
"""
병렬 EKF 파이프라인 provider.

두 EKF 파이프라인이 모드와 무관하게 매 tick 동시에 동작한다:

  [UWB  + Odom] → UwbOdomResidual (신호 품질 판단) → UwbOdomAEKF → (x, y, θ)
  [GNSS + Odom] → hAcc 연속 조건 판단              → RtkOdomAEKF → (lat, lon, θ_enu)

RTK EKF on/off 조건 (호출자 판단):
  - 켜기  : hAcc ≤ _HACC_INIT_M 인 fix 가 _HACC_CONSEC_INIT 회 연속 → initialize()
  - 끄기  : hAcc ≥ _HACC_RESET_M 인 fix 가 _HACC_CONSEC_RESET 회 연속 → reset()
  - 켜진 상태에서 hAcc 변동은 Sage-Husa 가 R 을 통해 자동 흡수한다.
    단, Sage-Husa 는 yaw_calibrated=True 이후에만 활성화된다.
    그 전에는 R 고정으로 GNSS 를 충분히 신뢰해 b_θ 를 빠르게 수렴시킨다.

UWB EKF on/off 조건 (호출자 판단):
  - 켜기  : UwbOdomResidual.is_stable=True 인 시점에서 initialize()
  - 신호 품질(is_stable) 판단은 kf_position_provider 가 직접 담당한다.
    UwbOdomAEKF 는 순수 EKF 수식만 포함하며, RTK 와 대칭 구조다.

초기화 이후 신호 손실 구간에는 odom dead-reckoning 으로 계속 동작한다.
경로 추종 로직은 KfPositionRecord 를 소비하는 외부 컴포넌트에서 구현한다.
"""

import time
import logging
import threading

from dataclasses import dataclass
from typing import Optional

from .singleton import singleton

from .unitree_go2_provider import UnitreeGo2Provider
from .uwb_provider import UwbProvider
from .rtk_provider import RtkProvider

from .utils.kf_utils import UwbOdomAEKF, UwbOdomResidual
from .utils.kf_utils.rtk_odom_aekf import RtkOdomAEKF


_TICK_SEC = 0.05  # 20 Hz

# RTK EKF on/off 조건
_HACC_INIT_M: float = 0.2       # 켜기: hAcc 이 값 이하
_HACC_RESET_M: float = 5.0      # 끄기: hAcc 이 값 이상
_HACC_CONSEC_INIT: int = 5      # 켜기에 필요한 연속 횟수
_HACC_CONSEC_RESET: int = 10    # 끄기에 필요한 연속 횟수

# UWB EKF on/off 조건 (UwbOdomResidual.is_stable 기반)
_UWB_CONSEC_INIT: int = 5       # 켜기에 필요한 연속 횟수
_UWB_CONSEC_RESET: int = 10      # 끄기에 필요한 연속 횟수


# ===================================================================================
# KfPositionRecord
# ===================================================================================

@dataclass(frozen=True)
class KfPositionRecord:
    """
    두 EKF 파이프라인의 현재 상태 스냅샷.

    Fields
    ------
    t_monotonic       : 레코드 생성 시각 (monotonic, s)

    uwb_ready             : UWB EKF 초기화 완료 여부
    uwb_yaw_calibrated    : UWB yaw bias (b_θ) 수렴 여부
    uwb_x_m               : UWB EKF 추정 x (m)           — uwb_ready=False 이면 None
    uwb_y_m               : UWB EKF 추정 y (m)           — uwb_ready=False 이면 None
    uwb_theta_rad         : UWB EKF 추정 heading (rad)   — uwb_ready=False 이면 None
    uwb_std_xy_m          : UWB EKF 위치 1-std 불확실성 (m) — uwb_ready=False 이면 None
    uwb_signal_stable     : UWB 잔차 안정화 여부

    rtk_ready             : RTK EKF 초기화 완료 여부
    rtk_yaw_calibrated    : RTK yaw bias (b_θ) 수렴 여부 — False 이면 heading 신뢰 불가
    rtk_lat               : RTK EKF 추정 위도             — rtk_ready=False 이면 None
    rtk_lon               : RTK EKF 추정 경도             — rtk_ready=False 이면 None
    rtk_theta_rad         : RTK EKF 추정 heading (rad)   — rtk_ready=False 이면 None
    rtk_std_xy_m          : RTK EKF 위치 1-std 불확실성 (m) — rtk_ready=False 이면 None
    """
    t_monotonic: float

    # UWB pipeline
    uwb_ready: bool = False
    uwb_yaw_calibrated: bool = False
    uwb_x_m: Optional[float] = None
    uwb_y_m: Optional[float] = None
    uwb_theta_rad: Optional[float] = None
    uwb_std_xy_m: Optional[float] = None
    uwb_signal_stable: bool = False

    # RTK pipeline
    rtk_ready: bool = False
    rtk_yaw_calibrated: bool = False
    rtk_lat: Optional[float] = None
    rtk_lon: Optional[float] = None
    rtk_theta_rad: Optional[float] = None
    rtk_std_xy_m: Optional[float] = None


# ===================================================================================
# KfPositionProvider
# ===================================================================================

@singleton
class KfPositionProvider:
    """
    UWB + RTK 병렬 EKF 파이프라인 provider.

    start() 이후 매 tick KfPositionRecord 를 생성하여 data 프로퍼티로 노출한다.
    두 EKF 는 항상 동시에 동작하며, 초기화 조건 충족 시 외부에서 initialize() 를 호출한다.
    """

    def __init__(self) -> None:
        self._unitree = UnitreeGo2Provider()
        self._uwb = UwbProvider()
        self._rtk = RtkProvider()

        self._uwb_ekf = UwbOdomAEKF()
        self._uwb_residual = UwbOdomResidual()
        self._rtk_ekf = RtkOdomAEKF()

        # UWB on/off 연속 카운터
        self._uwb_good_cnt: int = 0   # is_stable=True  연속 횟수
        self._uwb_bad_cnt: int = 0    # is_stable=False 연속 횟수

        # RTK on/off 연속 카운터
        self._rtk_good_cnt: int = 0   # hAcc ≤ _HACC_INIT_M 연속 횟수
        self._rtk_bad_cnt: int = 0    # hAcc ≥ _HACC_RESET_M 연속 횟수

        self._data: Optional[KfPositionRecord] = None
        self._lock = threading.Lock()
        self.running: bool = False
        self._thread: Optional[threading.Thread] = None


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("KfPositionProvider already running")
            return
        
        self.running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="KfPositionProviderWorker"
        )
        self._thread.start()
        logging.info("KfPositionProvider started")

    def stop(self) -> None:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        logging.info("KfPositionProvider stopped")

    # ------------------------------------------------------------------
    # Data API
    # ------------------------------------------------------------------

    @property
    def data(self) -> Optional[KfPositionRecord]:
        with self._lock:
            return self._data

    @property
    def uwb_ekf(self) -> UwbOdomAEKF:
        """UWB EKF 직접 접근 (진단·시각화 용도)."""
        return self._uwb_ekf

    @property
    def uwb_residual(self) -> UwbOdomResidual:
        """UWB 신호 품질 모니터 직접 접근 (진단·시각화 용도)."""
        return self._uwb_residual

    @property
    def rtk_ekf(self) -> RtkOdomAEKF:
        """RTK EKF 직접 접근 (진단·시각화 용도)."""
        return self._rtk_ekf

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while self.running:
                self._step_both_ekfs()
                rec = self._build_record()
                with self._lock:
                    self._data = rec
                time.sleep(_TICK_SEC)
        finally:
            self.running = False

    def _step_both_ekfs(self) -> None:
        """두 EKF 에 odom + 센서 데이터를 공급한다."""
        # ---- odom predict (두 EKF 공통) ----
        odom = self._unitree.get_odometry()
        if odom is not None and None not in (odom.x, odom.y, odom.yaw):
            self._uwb_ekf.predict(odom.x, odom.y, odom.yaw)
            self._rtk_ekf.predict(odom.x, odom.y, odom.yaw)

        # ---- UWB update ----
        uwb = self._uwb.data
        if uwb is not None and uwb.x_m is not None and uwb.y_m is not None:
            self._update_uwb_gate(uwb.x_m, uwb.y_m, odom)
            if self._uwb_ekf.initialized:
                self._uwb_ekf.update(uwb.x_m, uwb.y_m)

        # ---- RTK update ----
        rtk = self._rtk.data
        if rtk is not None and rtk.lat is not None and rtk.lon is not None:
            self._update_rtk_gate(rtk.lat, rtk.lon, rtk.hAcc_m)
            if self._rtk_ekf.initialized:
                self._rtk_ekf.update(rtk.lat, rtk.lon)

    def _update_uwb_gate(
        self,
        x_m: float,
        y_m: float,
        odom: Optional[object],
    ) -> None:
        """residual 안정성 조건으로 UWB EKF 를 켜고 끈다."""
        if odom is not None:
            self._uwb_residual.push_uwb(x_m, y_m, odom.x, odom.y)

        if not self._uwb_ekf.initialized:
            if self._uwb_residual.is_stable:
                self._uwb_good_cnt += 1
                self._uwb_bad_cnt = 0
                if self._uwb_good_cnt >= _UWB_CONSEC_INIT:
                    odom_yaw = odom.yaw if odom is not None else 0.0
                    self._uwb_ekf.initialize(x_m, y_m, odom_yaw)
                    self._uwb_good_cnt = 0
                    logging.info("UWB EKF initialized at (%.3f, %.3f)", x_m, y_m)
            else:
                self._uwb_good_cnt = 0
                self._uwb_bad_cnt = 0
        else:
            if not self._uwb_residual.is_stable:
                self._uwb_bad_cnt += 1
                self._uwb_good_cnt = 0
                if self._uwb_bad_cnt >= _UWB_CONSEC_RESET:
                    self._uwb_ekf.reset()
                    self._uwb_bad_cnt = 0
                    logging.info(
                        "UWB EKF reset — residual std=%.3fm exceeded threshold for %d consecutive fixes",
                        self._uwb_residual.std_m, _UWB_CONSEC_RESET,
                    )
            else:
                self._uwb_bad_cnt = 0

    def _update_rtk_gate(
        self,
        lat: float,
        lon: float,
        hAcc_m: Optional[float],
    ) -> None:
        """hAcc 연속 조건으로 RTK EKF 를 켜고 끈다."""
        if hAcc_m is None:
            self._rtk_good_cnt = 0
            self._rtk_bad_cnt = 0
            return

        if not self._rtk_ekf.initialized:
            # 꺼진 상태 → 켜기 조건 확인
            if hAcc_m <= _HACC_INIT_M:
                self._rtk_good_cnt += 1
                self._rtk_bad_cnt = 0
                if self._rtk_good_cnt >= _HACC_CONSEC_INIT:
                    self._rtk_ekf.initialize(lat, lon)
                    self._rtk_good_cnt = 0
                    logging.info(
                        "RTK EKF initialized at (%.7f, %.7f) — hAcc=%.2fm",
                        lat, lon, hAcc_m,
                    )
            else:
                self._rtk_good_cnt = 0
                self._rtk_bad_cnt = 0
        else:
            # 켜진 상태 → 끄기 조건 확인
            if hAcc_m >= _HACC_RESET_M:
                self._rtk_bad_cnt += 1
                self._rtk_good_cnt = 0
                if self._rtk_bad_cnt >= _HACC_CONSEC_RESET:
                    self._rtk_ekf.reset()
                    self._rtk_bad_cnt = 0
                    logging.info(
                        "RTK EKF reset — hAcc=%.2fm exceeded threshold %.1fm for %d consecutive fixes",
                        hAcc_m, _HACC_RESET_M, _HACC_CONSEC_RESET,
                    )
            else:
                self._rtk_bad_cnt = 0

    def _build_record(self) -> KfPositionRecord:
        uwb_ready = self._uwb_ekf.initialized
        rtk_ready = self._rtk_ekf.initialized
        return KfPositionRecord(
            t_monotonic=time.monotonic(),
            uwb_ready=uwb_ready,
            uwb_yaw_calibrated=self._uwb_ekf.yaw_calibrated if uwb_ready else False,
            uwb_x_m=self._uwb_ekf.x_m if uwb_ready else None,
            uwb_y_m=self._uwb_ekf.y_m if uwb_ready else None,
            uwb_theta_rad=self._uwb_ekf.theta_rad if uwb_ready else None,
            uwb_std_xy_m=self._uwb_ekf.std_xy_m if uwb_ready else None,
            uwb_signal_stable=self._uwb_residual.is_stable,
            rtk_ready=rtk_ready,
            rtk_yaw_calibrated=self._rtk_ekf.yaw_calibrated if rtk_ready else False,
            rtk_lat=self._rtk_ekf.lat if rtk_ready else None,
            rtk_lon=self._rtk_ekf.lon if rtk_ready else None,
            rtk_theta_rad=self._rtk_ekf.theta_rad if rtk_ready else None,
            rtk_std_xy_m=self._rtk_ekf.std_xy_m if rtk_ready else None,
        )


