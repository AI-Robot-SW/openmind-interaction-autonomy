# rtk_odom_aekf.py
"""
RTK/GNSS + odometry 융합 Adaptive EKF with Yaw Bias (실외, 홀로노믹).

State  : [x, y, b_θ]  — 로컬 ENU 프레임
           x   = East offset from origin (m)
           y   = North offset from origin (m)
           b_θ = odom position frame 의 x/y 축을 ENU x/y 축에 맞추는 회전 오프셋 (rad)

Process: 오도메트리 증분을 b_θ 로 ENU 로 회전 (홀로노믹)
           [dx_enu, dy_enu]^T = R(b_θ) · [dx_odom, dy_odom]^T
           b_θ 는 거의 상수 (slow random walk)
Measure: RTK/GNSS lat/lon → flat-earth ENU 변환 후 측정 갱신

홀로노믹 모델
-------------
전진/후진/측면 이동 모두 처리한다. odom 이 보고하는 변위 벡터
(dx_odom, dy_odom) 를 b_θ 만큼 회전해 ENU displacement 로 변환한다.
odom_yaw 는 state propagation 에 사용하지 않으며, global_yaw 출력 계산에만 쓰인다.

    global_yaw = odom_yaw + b_θ  (출력용 ENU heading)

Yaw Bias Estimation
-------------------
GNSS 측정 모델 H = [[1,0,0],[0,1,0]] 에서 b_θ 열이 0이므로,
GNSS가 b_θ 를 직접 관측하지는 않는다.

그러나 예측 모델에서:
    F[0,2] = ∂x/∂b_θ = -dx_odom·sin(b) - dy_odom·cos(b)
    F[1,2] = ∂y/∂b_θ =  dx_odom·cos(b) - dy_odom·sin(b)

로봇이 이동하면 P_xb, P_yb 가 0이 아니게 되어
GNSS 위치 innovation 이 K_b 를 통해 b_θ 를 간접 업데이트한다.

    b_θ ← b_θ + K_b · ν

즉 odom frame 이 ENU 와 어긋나면 예측 경로가 GNSS 경로와
체계적으로 어긋나고, EKF가 이 위치 오차 패턴으로 b_θ 를 보정한다.

주의
----
* 정지 상태에서는 dx_odom = dy_odom = 0 이므로 F의 coupling term 이 0 → b_θ 추정 불가.
  이동 중에만 b_θ 가 수렴한다.
* yaw_calibrated 프로퍼티로 b_θ 불확실성이 충분히 줄었는지 확인한다.
* R은 Sage-Husa algorithm 으로 online 추정한다.
"""

import math
from typing import Optional

import numpy as np

from ..geo_utils import enu_to_latlon, latlon_to_enu, wrap_rad


class RtkOdomAEKF:
    """Adaptive EKF with online yaw bias estimation for RTK/GNSS + odometry fusion.

    State: [x (m), y (m), b_θ (rad)]
      - x, y  : ENU position relative to origin
      - b_θ   : odom position frame → ENU frame 회전 오프셋
                (odom 의 x/y 축을 ENU 의 x/y 축에 맞추는 각도)

    Motion model (holonomic):
        [dx_enu, dy_enu]^T = R(b_θ) · [dx_odom, dy_odom]^T
        전진/후진/측면 이동 모두 처리.

    global_yaw = odom_yaw + b_θ  (출력용 ENU heading — propagation 에는 미사용)

    b_θ is not directly observed by GNSS, but is estimated indirectly
    through position-bias covariance coupling built up during motion.
    """

    # --- Process noise (Q) tuning ---
    _Q_XY_M2_PER_M: float = 0.005        # x/y 프로세스 노이즈 (m²/m) — std≈7cm/m
    _Q_BIAS_RAD2_PER_STEP: float = 1e-6   # b_θ random walk — std≈0.057°/step (거의 고정)

    # --- Initial covariance (P) ---
    _P_INIT_XY_M2: float = 1.0
    _P_INIT_BIAS_RAD2: float = math.pi ** 2   # b_θ 초기값 완전 불명

    # --- Adaptive R (Sage-Husa) ---
    _R_INIT_M2: float = 0.25              # R 초기값 (m²)
    _R_MIN_M2: float = 0.01               # R 하한 (std=10cm) — RTK Fix 실측 정확도 기반
    _R_MAX_M2: float = 25.0               # R 상한 — 이상치 과대적합 방지
    _FORGET_B: float = 0.98               # 망각 인수 b (클수록 과거 가중)
    _WARMUP_STEPS: int = 20               # 초기 R 고정 스텝 수

    # --- Yaw calibration threshold ---
    _BIAS_CALIBRATED_STD_DEG: float = 5.0  # P[2,2] < (5°)² → yaw_calibrated = True

    def __init__(self) -> None:
        self._x = np.zeros(3, dtype=float)   # [x, y, b_θ]
        self._P = np.diag([
            self._P_INIT_XY_M2,
            self._P_INIT_XY_M2,
            self._P_INIT_BIAS_RAD2,
        ])
        self._R = np.diag([self._R_INIT_M2, self._R_INIT_M2])

        self._initialized: bool = False
        self._n_updates: int = 0
        self._sh_step: int = 0               # Sage-Husa 전용 스텝 카운터 (yaw_calibrated 후 리셋)
        self._prev_yaw_cal: bool = False     # yaw_calibrated 전환 감지용
        self._prev_odom: Optional[tuple[float, float, float]] = None  # (x, y, yaw)
        self._last_odom_yaw: float = 0.0   # global_yaw 계산용

        self._origin_lat: Optional[float] = None
        self._origin_lon: Optional[float] = None
        self._prev_gnss: Optional[tuple[float, float]] = None  # 직전 GNSS fix (lat, lon)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_origin(self, lat: float, lon: float) -> None:
        """ENU 원점을 명시적으로 설정한다 (선택적; 미설정 시 initialize() 에서 자동 설정)."""
        self._origin_lat = lat
        self._origin_lon = lon

    def initialize(
        self,
        lat: float,
        lon: float,
        odom_yaw_rad: float = 0.0,
    ) -> None:
        """
        lat/lon 으로 EKF 를 초기화한다 (켜기).

        Parameters
        ----------
        lat, lon : float
            WGS84 위경도.
        odom_yaw_rad : float
            초기화 시점의 odom yaw (rad). global_yaw 프로퍼티 계산에 사용.
            기본값 0.0.

        Notes
        -----
        b_θ 는 항상 0.0 으로 초기화하고 P[2,2] = π² 으로 둔다.
        이동 후 GNSS position innovation 을 통해 자동 수렴한다.
        """
        if self._origin_lat is None:
            self._origin_lat = lat
            self._origin_lon = lon

        x_m, y_m = latlon_to_enu(lat, lon, self._origin_lat, self._origin_lon)
        self._x[:] = [x_m, y_m, 0.0]   # b_θ = 0 (unknown)
        self._P = np.diag([
            self._P_INIT_XY_M2,
            self._P_INIT_XY_M2,
            self._P_INIT_BIAS_RAD2,
        ])
        self._R = np.diag([self._R_INIT_M2, self._R_INIT_M2])
        self._n_updates = 0
        self._sh_step = 0
        self._prev_yaw_cal = False
        self._prev_odom = None
        self._prev_gnss = None
        self._last_odom_yaw = odom_yaw_rad
        self._initialized = True

    def reset(self) -> None:
        """EKF 를 초기 상태로 되돌린다 (끄기). 다음 initialize() 호출 전까지 predict/update 가 무시된다."""
        self._x = np.zeros(3, dtype=float)
        self._P = np.diag([
            self._P_INIT_XY_M2,
            self._P_INIT_XY_M2,
            self._P_INIT_BIAS_RAD2,
        ])
        self._R = np.diag([self._R_INIT_M2, self._R_INIT_M2])
        self._initialized = False
        self._n_updates = 0
        self._sh_step = 0
        self._prev_yaw_cal = False
        self._prev_odom = None
        self._prev_gnss = None
        self._last_odom_yaw = 0.0
        self._origin_lat = None
        self._origin_lon = None

    def predict(self, odom_x_m: float, odom_y_m: float, odom_yaw_rad: float) -> None:
        """
        오도메트리 raw 좌표로 시간 갱신 (홀로노믹 모델).

        odom frame 변위 벡터를 b_θ 만큼 회전해 ENU displacement 로 변환한다.
        전진/후진/측면 이동 모두 올바르게 처리된다.

            x_new = x + dx_odom·cos(b) - dy_odom·sin(b)
            y_new = y + dx_odom·sin(b) + dy_odom·cos(b)
            b_new = b   (상수 모델)

        Jacobian coupling (b_θ 가 움직일 때만 0이 아님):
            F[0,2] = -dx_odom·sin(b) - dy_odom·cos(b)
            F[1,2] =  dx_odom·cos(b) - dy_odom·sin(b)

        Parameters
        ----------
        odom_x_m, odom_y_m : float
            오도메트리 프레임의 현재 위치 (m).
        odom_yaw_rad : float
            odom heading (rad, CCW positive). 상태 전파에는 사용하지 않고
            global_yaw 프로퍼티 계산(_last_odom_yaw)에만 쓰인다.
        """
        self._last_odom_yaw = odom_yaw_rad

        if self._prev_odom is None:
            self._prev_odom = (odom_x_m, odom_y_m, odom_yaw_rad)
            return

        dx_odom = odom_x_m - self._prev_odom[0]
        dy_odom = odom_y_m - self._prev_odom[1]
        delta_s_m = math.sqrt(dx_odom * dx_odom + dy_odom * dy_odom)

        self._prev_odom = (odom_x_m, odom_y_m, odom_yaw_rad)

        if not self._initialized:
            return

        x, y, b = self._x
        cos_b = math.cos(b)
        sin_b = math.sin(b)

        # 홀로노믹 상태 전파: odom 변위를 b_θ 로 ENU 로 회전
        self._x[0] = x + dx_odom * cos_b - dy_odom * sin_b
        self._x[1] = y + dx_odom * sin_b + dy_odom * cos_b
        # self._x[2] = b  (변경 없음)

        # Jacobian F (3×3)
        F = np.array([
            [1.0, 0.0, -dx_odom * sin_b - dy_odom * cos_b],
            [0.0, 1.0,  dx_odom * cos_b - dy_odom * sin_b],
            [0.0, 0.0,  1.0],
        ])

        abs_s = max(delta_s_m, 1e-6)
        Q = np.diag([
            self._Q_XY_M2_PER_M * abs_s,
            self._Q_XY_M2_PER_M * abs_s,
            self._Q_BIAS_RAD2_PER_STEP,   # b_θ: 이동량 무관, 고정 drift
        ])

        self._P = F @ self._P @ F.T + Q

    def update(self, lat: float, lon: float) -> bool:
        """
        RTK/GNSS lat/lon 으로 측정 갱신.

        Parameters
        ----------
        lat, lon : float
            WGS84 위경도.

        Returns
        -------
        bool
            측정값이 EKF 에 반영되면 True.
        """
        if not self._initialized:
            return False

        if self._origin_lat is None:
            return False

        # 직전과 완전히 동일한 값 → 수신기 freeze, 무시
        if self._prev_gnss is not None and lat == self._prev_gnss[0] and lon == self._prev_gnss[1]:
            return False

        self._prev_gnss = (lat, lon)

        x_m, y_m = latlon_to_enu(lat, lon, self._origin_lat, self._origin_lon)
        return self._do_update(x_m, y_m)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_update(self, meas_x_m: float, meas_y_m: float) -> bool:
        """EKF 측정 갱신 + Sage-Husa adaptive R 갱신 (Joseph form 공분산).

        H = [[1, 0, 0],   ← x 관측
             [0, 1, 0]]   ← y 관측, b_θ 열 = 0 (직접 관측 안 함)

        b_θ 는 P_xb, P_yb coupling 을 통해 K_b 로 간접 업데이트된다.
        """
        H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        z = np.array([meas_x_m, meas_y_m])
        innov = z - self._x[:2]
        P_pred = self._P.copy()

        # Sage-Husa R 갱신: b_θ 수렴(yaw_calibrated) 후에만 수행.
        # yaw 가 크게 틀린 상태에서 Sage-Husa 를 켜면 큰 innovation 을 GNSS 노이즈로
        # 오해해 R 을 올려버리고, 그러면 K 가 줄어 bias 보정도 느려지는
        # 양성 피드백 루프에 빠진다.  P[2,2] 가 충분히 수렴한 뒤에야 R 을 적응시킨다.
        #
        # _sh_step 은 yaw_calibrated 가 처음 True 가 될 때 0 으로 리셋된다.
        # 덕분에 d_k 가 처음엔 크게 시작해 R 이 빠르게 수렴하고, 이후 점진적으로 느려진다.
        cal_now = self.yaw_calibrated
        if cal_now and not self._prev_yaw_cal:
            # yaw_calibrated 전환 순간: Sage-Husa 카운터 리셋
            self._sh_step = 0
        self._prev_yaw_cal = cal_now

        if self._n_updates >= self._WARMUP_STEPS and cal_now:
            d_k = (1.0 - self._FORGET_B) / (1.0 - self._FORGET_B ** (self._sh_step + 1))
            R_innov = np.outer(innov, innov) - H @ P_pred @ H.T
            self._R = (1.0 - d_k) * self._R + d_k * R_innov
            diag_clamped = np.clip(np.diag(self._R), self._R_MIN_M2, self._R_MAX_M2)
            self._R = np.diag(diag_clamped)
            self._sh_step += 1

        S = H @ P_pred @ H.T + self._R
        K = np.linalg.solve(S, H @ P_pred).T

        self._x = self._x + K @ innov
        self._x[2] = wrap_rad(self._x[2])   # b_θ wrap

        # Joseph form: 수치 안정성 보장
        I_KH = np.eye(3) - K @ H
        self._P = I_KH @ P_pred @ I_KH.T + K @ self._R @ K.T

        self._n_updates += 1
        return True

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def lat(self) -> Optional[float]:
        """현재 추정 위도 (WGS84)."""
        if self._origin_lat is None or not self._initialized:
            return None
        lat, _ = enu_to_latlon(self._x[0], self._x[1], self._origin_lat, self._origin_lon)
        return lat

    @property
    def lon(self) -> Optional[float]:
        """현재 추정 경도 (WGS84)."""
        if self._origin_lon is None or not self._initialized:
            return None
        _, lon = enu_to_latlon(self._x[0], self._x[1], self._origin_lat, self._origin_lon)
        return lon

    @property
    def origin(self) -> Optional[tuple[float, float]]:
        """ENU 원점 (lat, lon). 미설정 시 None."""
        if self._origin_lat is None:
            return None
        return (self._origin_lat, self._origin_lon)

    @property
    def x_m(self) -> float:
        return float(self._x[0])

    @property
    def y_m(self) -> float:
        return float(self._x[1])

    @property
    def b_theta_rad(self) -> float:
        """추정된 yaw bias (rad). odom position frame → ENU frame 회전 오프셋."""
        return float(self._x[2])

    @property
    def global_yaw_rad(self) -> float:
        """ENU 기준 절대 heading (rad). odom_yaw + b_θ. 출력용 — propagation 에는 미사용."""
        return wrap_rad(self._last_odom_yaw + self._x[2])

    @property
    def theta_rad(self) -> float:
        """global_yaw_rad 의 alias (kf_position_provider 호환용)."""
        return self.global_yaw_rad

    @property
    def theta_deg(self) -> float:
        return math.degrees(self.global_yaw_rad)

    @property
    def std_xy_m(self) -> float:
        """평균 1-std 위치 불확실성 (m)."""
        return float(math.sqrt((self._P[0, 0] + self._P[1, 1]) / 2.0))

    @property
    def std_bias_deg(self) -> float:
        """yaw bias 1-std 불확실성 (deg)."""
        return float(math.degrees(math.sqrt(max(0.0, self._P[2, 2]))))

    @property
    def yaw_calibrated(self) -> bool:
        """b_θ 불확실성이 임계값 이하로 수렴했으면 True.

        이동 중 GNSS position innovation 이 충분히 누적되면 True 로 전환된다.
        True 가 될 때까지 global_yaw_rad / theta_rad 는 신뢰하지 말 것.
        """
        threshold_rad2 = math.radians(self._BIAS_CALIBRATED_STD_DEG) ** 2
        return bool(self._P[2, 2] < threshold_rad2)

    @property
    def r_xy_m2(self) -> float:
        """현재 적응된 R 대각 평균 (m²). 추정 측정 노이즈 크기 지표."""
        return float((self._R[0, 0] + self._R[1, 1]) / 2.0)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def n_updates(self) -> int:
        return self._n_updates
