"""
Location Sensor - Input Orchestrator Location 모듈

위치(RTK GPS, UWB) 및 주행 상태(NavigationProvider) 정보를
수집하여 Fuser에 전달합니다.

Architecture:
    RtkProvider  ──┐
    UwbProvider  ──┼── LocationSensor._poll() → LocationSnapshot
    NavigationProvider ─┘        ↓
                         raw_to_text() → formatted_latest_buffer() → Fuser

Dependencies:
    - RtkProvider: RTK GPS 위치 데이터 (singleton, rtk_bg에서 초기화)
    - UwbProvider: UWB 실내 위치 데이터 (singleton, uwb_bg에서 초기화)
    - NavigationProvider: 주행 상태/속도/목표 (singleton, gnss_route_bg에서 초기화)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput

from providers.io_provider import IOProvider
from providers.rtk_provider import RtkProvider
from providers.uwb_provider import UwbProvider
from providers.navigation_provider import NavigationProvider

logger = logging.getLogger(__name__)

# RTK carrSoln 값 → 사람이 읽을 수 있는 라벨
_CARR_SOLN_LABELS = {0: "No-RTK", 1: "RTK-Float", 2: "RTK-Fix"}


@dataclass
class LocationSnapshot:
    """
    위치 + 주행 상태 통합 스냅샷.

    세 Provider(RTK, UWB, Navigation)의 데이터를 한 시점에 수집한 결과.
    """

    timestamp: float

    # --- GNSS (RtkProvider) ---
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None
    gnss_fix_type: Optional[int] = None  # 0=no fix, 2=2D, 3=3D
    gnss_carr_soln: Optional[int] = None  # 0=none, 1=float, 2=fix
    gnss_h_acc_m: Optional[float] = None  # 수평 정확도 (m)
    gnss_num_sv: Optional[int] = None  # 위성 수

    # --- UWB (UwbProvider) ---
    uwb_x_m: Optional[float] = None
    uwb_y_m: Optional[float] = None
    uwb_z_m: Optional[float] = None
    uwb_quality: Optional[int] = None

    # --- Navigation (NavigationProvider) ---
    nav_mode: str = "IDLE"  # IDLE, CALIBRATING, DWA, STOP
    nav_heading_calibrated: bool = False
    nav_reached_goal: bool = False
    nav_vx: float = 0.0  # 전진 속도 (m/s)
    nav_vyaw: float = 0.0  # 회전 속도 (rad/s)
    nav_active_goal: Optional[str] = None  # 목표 place_id
    nav_remaining_distance_m: float = 0.0  # 목표까지 남은 거리 (m)
    nav_target_speed: float = 0.0  # 목표 속도 (m/s)


class LocationSensorConfig(SensorConfig):
    """Location Sensor 설정."""

    poll_interval_s: float = Field(
        default=1.0, description="폴링 간격 (초)"
    )
    enable_gps: bool = Field(default=False, description="GPS 섹션 활성화")
    enable_uwb: bool = Field(default=False, description="UWB 섹션 활성화")
    enable_navigation: bool = Field(
        default=True, description="Navigation 섹션 활성화"
    )


class LocationSensor(FuserInput[LocationSensorConfig, Optional[LocationSnapshot]]):
    """
    Location Sensor - 위치 및 주행 상태 입력 센서.

    RTK GPS, UWB, NavigationProvider로부터 데이터를 수집하여
    InputOrchestrator → Fuser로 전달합니다.
    """

    def __init__(self, config: LocationSensorConfig):
        super().__init__(config)

        self.descriptor_for_LLM = "Location"
        self.messages: list[Message] = []
        self.io_provider = IOProvider()

        # lazy singleton access — Background에서 먼저 초기화됨
        self._rtk: Optional[RtkProvider] = None
        self._uwb: Optional[UwbProvider] = None
        self._nav: Optional[NavigationProvider] = None

        logger.info(
            "LocationSensor initialized: poll_interval=%.1fs",
            config.poll_interval_s,
        )

    # ======================== Provider access ========================

    def _get_rtk(self) -> Optional[RtkProvider]:
        if self._rtk is None:
            try:
                self._rtk = RtkProvider()
            except Exception:
                return None
        return self._rtk

    def _get_uwb(self) -> Optional[UwbProvider]:
        if self._uwb is None:
            try:
                self._uwb = UwbProvider()
            except Exception:
                return None
        return self._uwb

    def _get_nav(self) -> Optional[NavigationProvider]:
        if self._nav is None:
            try:
                self._nav = NavigationProvider()
            except Exception:
                return None
        return self._nav

    # ======================== Polling ========================

    async def _poll(self) -> Optional[LocationSnapshot]:
        await asyncio.sleep(self.config.poll_interval_s)

        now = time.time()
        snap = LocationSnapshot(timestamp=now)
        has_data = False

        # --- RTK ---
        if self.config.enable_gps:
            rtk = self._get_rtk()
            if rtk is not None:
                rec = rtk.data
                if rec is not None and rec.lat is not None:
                    snap.gnss_lat = rec.lat
                    snap.gnss_lon = rec.lon
                    snap.gnss_fix_type = rec.fixType
                    snap.gnss_carr_soln = rec.carrSoln
                    snap.gnss_h_acc_m = rec.hAcc_m
                    snap.gnss_num_sv = rec.numSV
                    has_data = True

        # --- UWB ---
        if self.config.enable_uwb:
            uwb = self._get_uwb()
            if uwb is not None:
                rec = uwb.data
                if rec is not None and rec.x_m is not None:
                    snap.uwb_x_m = rec.x_m
                    snap.uwb_y_m = rec.y_m
                    snap.uwb_z_m = rec.z_m
                    snap.uwb_quality = rec.quality_factor
                    has_data = True

        # --- Navigation ---
        if self.config.enable_navigation and (nav := self._get_nav()) is not None and nav.running:
            state = nav.get_state()
            snap.nav_mode = state.mode
            snap.nav_heading_calibrated = state.heading_calibrated
            snap.nav_reached_goal = state.reached_goal
            snap.nav_vx = state.vx
            snap.nav_vyaw = state.vyaw
            snap.nav_active_goal = nav.get_active_goal()
            snap.nav_target_speed = nav.get_target_speed()
            try:
                snap.nav_remaining_distance_m = nav.get_remaining_distance()
            except Exception:
                snap.nav_remaining_distance_m = 0.0
            has_data = True

        return snap if has_data else None

    # ======================== Text conversion ========================

    async def _raw_to_text(
        self, raw_input: Optional[LocationSnapshot]
    ) -> Optional[Message]:
        if raw_input is None:
            return None

        parts: list[str] = []

        # --- GPS 섹션 ---
        if raw_input.gnss_lat is not None:
            lat, lon = raw_input.gnss_lat, raw_input.gnss_lon
            lat_dir = "N" if lat >= 0 else "S"
            lon_dir = "E" if lon >= 0 else "W"
            carr_label = _CARR_SOLN_LABELS.get(
                raw_input.gnss_carr_soln or 0, "Unknown"
            )

            gps_parts = [f"{abs(lat):.7f}°{lat_dir}, {abs(lon):.7f}°{lon_dir}"]
            gps_parts.append(carr_label)
            if raw_input.gnss_h_acc_m is not None:
                gps_parts.append(f"accuracy: {raw_input.gnss_h_acc_m:.2f}m")
            if raw_input.gnss_num_sv is not None:
                gps_parts.append(f"satellites: {raw_input.gnss_num_sv}")

            parts.append("[GPS] " + " | ".join(gps_parts))

        # --- UWB 섹션 ---
        if raw_input.uwb_x_m is not None:
            uwb_parts = [
                f"x={raw_input.uwb_x_m:.2f}m, y={raw_input.uwb_y_m:.2f}m"
            ]
            if raw_input.uwb_z_m is not None:
                uwb_parts[0] += f", z={raw_input.uwb_z_m:.2f}m"
            if raw_input.uwb_quality is not None:
                uwb_parts.append(f"quality: {raw_input.uwb_quality}")
            parts.append("[UWB] " + " | ".join(uwb_parts))

        # --- Navigation 섹션 ---
        nav_parts = [f"Mode: {raw_input.nav_mode}"]

        if raw_input.nav_active_goal:
            nav_parts.append(f"Goal: {raw_input.nav_active_goal}")

        if raw_input.nav_reached_goal:
            nav_parts.append("ARRIVED")
        elif raw_input.nav_remaining_distance_m > 0:
            nav_parts.append(
                f"Remaining: {raw_input.nav_remaining_distance_m:.1f}m"
            )

        if raw_input.nav_vx > 0:
            nav_parts.append(f"Speed: {raw_input.nav_vx:.2f}m/s")

        if not raw_input.nav_heading_calibrated and raw_input.nav_mode == "CALIBRATING":
            nav_parts.append("heading calibrating")

        parts.append("[Navigation] " + " | ".join(nav_parts))

        message = "\n".join(parts)
        return Message(timestamp=raw_input.timestamp, message=message)

    async def raw_to_text(self, raw_input: Optional[LocationSnapshot]) -> None:
        pending = await self._raw_to_text(raw_input)
        if pending is not None:
            # 위치 데이터는 항상 최신 1건만 유지
            self.messages = [pending]

    def formatted_latest_buffer(self) -> Optional[str]:
        if len(self.messages) == 0:
            return None

        latest = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__,
            latest.message,
            latest.timestamp,
        )

        self.messages = []
        return result
