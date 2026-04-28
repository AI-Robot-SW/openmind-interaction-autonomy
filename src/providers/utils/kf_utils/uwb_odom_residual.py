# uwb_odom_residual.py
"""
UWB + odometry 잔차 기반 신호 품질 모니터.

Classes
-------
OdomResidualBase
    센서 + odom 잔차 기반 신호 품질 모니터 베이스.

UwbOdomResidual
    UWB 전용 신호 품질 모니터. OdomResidualBase 를 상속.

사용 방법
---------
    monitor = UwbOdomResidual()

    # UWB fix 마다
    monitor.push_uwb(raw.x_m, raw.y_m, odom_x_m, odom_y_m)

    if monitor.is_stable:
        ...  # UWB 신뢰 가능
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from ..geo_utils import euclidean_dist_m


# ---------------------------------------------------------------------------
# OdomResidualBase
# ---------------------------------------------------------------------------

class OdomResidualBase:
    """센서 + odom 잔차 기반 신호 품질 모니터 베이스 클래스.

    잔차 정의
    ---------
        residual_i = sensor_displacement(fix_{i-1}, fix_i)
                   - odom_displacement(odom_at_fix_{i-1}, odom_at_fix_i)

    윈도우 구조
    -----------
        유효 fix 마다 잔차를 윈도우에 추가. 윈도우 가득 참 + std < _STD_ENTER → stable.

        히스테리시스:
            False → True: len(window) == _TOTAL_WINDOW AND std < _STD_ENTER_THRESHOLD_M
            True  → False: std >= _STD_EXIT_THRESHOLD_M

    서브클래스는 센서별 push_* 메서드를 구현하고
    유효 fix 확인 시 _record_sensor() 를 호출한다.
    """

    _TOTAL_WINDOW: int = 15
    _STD_ENTER_THRESHOLD_M: float = 0.001
    _STD_EXIT_THRESHOLD_M: float = 0.001

    def __init__(self) -> None:
        self._previous_pos: Optional[tuple[float, float]] = None
        self._previous_odom: Optional[tuple[float, float]] = None
        self._residual_window: deque[float] = deque(maxlen=self._TOTAL_WINDOW)
        self._stable: bool = False

    def reset(self) -> None:
        self._previous_pos = None
        self._previous_odom = None
        self._residual_window.clear()
        self._stable = False

    def _record_sensor(self, pos_x_m: float, pos_y_m: float, odom_x_m: float, odom_y_m: float) -> None:
        if self._previous_pos is not None and self._previous_odom is not None:
            self._residual_window.append(self._compute_residual(pos_x_m, pos_y_m, odom_x_m, odom_y_m))
        self._previous_pos = (pos_x_m, pos_y_m)
        self._previous_odom = (odom_x_m, odom_y_m)

    def _compute_residual(self, pos_x_m: float, pos_y_m: float, odom_x_m: float, odom_y_m: float) -> float:
        pos_dist = euclidean_dist_m(
            self._previous_pos[0], self._previous_pos[1], pos_x_m, pos_y_m  # type: ignore[index]
        )
        odom_dist = euclidean_dist_m(
            self._previous_odom[0], self._previous_odom[1], odom_x_m, odom_y_m  # type: ignore[index]
        )
        return pos_dist - odom_dist

    @property
    def std_m(self) -> Optional[float]:
        if len(self._residual_window) < self._TOTAL_WINDOW:
            return None
        return float(np.std(self._residual_window))

    @property
    def is_stable(self) -> bool:
        std_m = self.std_m
        if self._stable:
            if std_m is None or std_m >= self._STD_EXIT_THRESHOLD_M:
                self._stable = False
        else:
            if std_m is not None and std_m < self._STD_ENTER_THRESHOLD_M:
                self._stable = True
        return self._stable


# ---------------------------------------------------------------------------
# UwbOdomResidual
# ---------------------------------------------------------------------------

class UwbOdomResidual(OdomResidualBase):
    """UWB 신호 품질 모니터."""

    _STD_ENTER_THRESHOLD_M = 0.07
    _STD_EXIT_THRESHOLD_M = 1.3

    def push_uwb(
        self,
        x_m: Optional[float],
        y_m: Optional[float],
        odom_x_m: Optional[float],
        odom_y_m: Optional[float],
    ) -> None:
        """UWB fix 가 왔을 때 호출한다.

        x_m/y_m 또는 odom_x_m/odom_y_m 이 None 이면 윈도우에 기록하지 않는다.
        """
        if x_m is None or y_m is None or odom_x_m is None or odom_y_m is None:
            return
        if self._previous_pos is None or (x_m != self._previous_pos[0] or y_m != self._previous_pos[1]):
            self._record_sensor(x_m, y_m, odom_x_m, odom_y_m)


