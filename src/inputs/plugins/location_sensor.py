"""
Location Sensor - Input Orchestrator Location 모듈

이 모듈은 로봇의 위치 정보를 수집하고,
위치 기반 컨텍스트를 Fuser에 전달합니다.

Architecture:
    LocationBg -> LocationProvider -> LocationSensor -> InputOrchestrator -> Fuser

Dependencies:
    - LocationProvider: 위치 데이터 관리 (AMCL, Lidar Localization, GPS 등)

Note:
    LocationProvider는 다른 모듈(Sensor, MoveConnector, UnitreeGo2Provider)에서
    호출 가능한 Interface 역할을 수행합니다.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput

from providers.io_provider import IOProvider
from providers.location_provider import LocationProvider


@dataclass
class Pose2D:
    """
    2D 위치 및 방향.

    Attributes
    ----------
    x : float
        X 좌표 (meters)
    y : float
        Y 좌표 (meters)
    theta : float
        방향 (radians)
    """

    x: float
    y: float
    theta: float


@dataclass
class Pose3D:
    """
    3D 위치 및 방향 (Quaternion).

    Attributes
    ----------
    x : float
        X 좌표 (meters)
    y : float
        Y 좌표 (meters)
    z : float
        Z 좌표 (meters)
    qx : float
        Quaternion x
    qy : float
        Quaternion y
    qz : float
        Quaternion z
    qw : float
        Quaternion w
    """

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class NavigationGoal:
    """
    네비게이션 목표.

    Attributes
    ----------
    goal_id : str
        목표 ID
    name : Optional[str]
        목표 이름 (예: "회의실", "충전소")
    pose : Pose2D
        목표 위치
    is_active : bool
        현재 활성 목표 여부
    """

    goal_id: str
    name: Optional[str]
    pose: Pose2D
    is_active: bool = False


@dataclass
class LocationData:
    """
    위치 데이터 컨테이너.

    Attributes
    ----------
    timestamp : float
        데이터 타임스탬프
    current_pose : Optional[Pose2D]
        현재 위치
    current_pose_3d : Optional[Pose3D]
        현재 3D 위치
    current_goal : Optional[NavigationGoal]
        현재 네비게이션 목표
    distance_to_goal : Optional[float]
        목표까지 남은 거리 (meters)
    estimated_time_to_goal : Optional[float]
        목표까지 예상 시간 (seconds)
    current_location_name : Optional[str]
        현재 위치 이름 (예: "로비", "복도")
    nearby_locations : List[str]
        근처 위치 목록
    localization_confidence : float
        위치 추정 신뢰도 (0.0 ~ 1.0)
    is_navigating : bool
        네비게이션 진행 중 여부
    navigation_status : str
        네비게이션 상태 ("idle", "navigating", "arrived", "failed")
    """

    timestamp: float
    current_pose: Optional[Pose2D] = None
    current_pose_3d: Optional[Pose3D] = None
    current_goal: Optional[NavigationGoal] = None
    distance_to_goal: Optional[float] = None
    estimated_time_to_goal: Optional[float] = None
    current_location_name: Optional[str] = None
    nearby_locations: Optional[List[str]] = None
    localization_confidence: float = 0.0
    is_navigating: bool = False
    navigation_status: str = "idle"


class LocationSensorConfig(SensorConfig):
    """
    Location Sensor 설정.

    Parameters
    ----------
    localization_method : str
        위치 추정 방법 ("amcl", "lidar", "gps", "fusion")
    update_rate_hz : float
        위치 업데이트 주기 (Hz)
    min_confidence_threshold : float
        최소 신뢰도 임계값
    enable_location_names : bool
        위치 이름 매핑 활성화
    location_map_file : Optional[str]
        위치 이름 맵 파일 경로
    coordinate_frame : str
        좌표계 프레임 ("map", "odom", "base_link")
    report_interval_ms : int
        Fuser 보고 간격 (ms)
    """

    localization_method: str = Field(
        default="lidar", description="위치 추정 방법"
    )
    update_rate_hz: float = Field(default=10.0, description="위치 업데이트 주기 (Hz)")
    min_confidence_threshold: float = Field(
        default=0.5, description="최소 신뢰도 임계값"
    )
    enable_location_names: bool = Field(
        default=True, description="위치 이름 매핑 활성화"
    )
    location_map_file: Optional[str] = Field(
        default=None, description="위치 이름 맵 파일"
    )
    coordinate_frame: str = Field(default="map", description="좌표계 프레임")
    report_interval_ms: int = Field(default=1000, description="Fuser 보고 간격 (ms)")


class LocationSensor(FuserInput[LocationSensorConfig, Optional[LocationData]]):
    """
    Location Sensor - 위치 입력 처리 센서.

    로봇의 현재 위치, 목표 위치, 네비게이션 상태 등을 수집하여
    InputOrchestrator에 전달합니다.

    Attributes
    ----------
    descriptor_for_LLM : str
        LLM에 전달될 입력 소스 설명자
    message_buffer : Queue[LocationData]
        위치 데이터 버퍼
    latest_location : Optional[LocationData]
        최신 위치 데이터

    Methods
    -------
    _poll() -> Optional[LocationData]
        버퍼에서 새 위치 데이터 폴링
    _raw_to_text(raw_input) -> Optional[Message]
        위치 데이터를 텍스트 메시지로 변환
    formatted_latest_buffer() -> Optional[str]
        최신 버퍼 내용을 포맷팅하여 반환
    get_current_pose() -> Optional[Pose2D]
        현재 위치 반환
    get_distance_to_goal() -> Optional[float]
        목표까지 거리 반환
    has_active_goal() -> bool
        활성 목표 존재 여부 반환
    """

    def __init__(self, config: LocationSensorConfig):
        """
        Location Sensor 초기화.

        Parameters
        ----------
        config : LocationSensorConfig
            센서 설정
        """
        super().__init__(config)

        # LLM 입력 설명자
        self.descriptor_for_LLM = "Location"

        # 메시지 버퍼 (스킬 준수: list[Message])
        self.messages: list[Message] = []

        # IOProvider 싱글턴 — Fuser 입력 기록용
        self.io_provider = IOProvider()

        # 데이터 버퍼
        self.message_buffer: Queue[LocationData] = Queue()
        self.latest_location: Optional[LocationData] = None

        # 위치 이름 맵
        self._location_names: Dict[Tuple[float, float], str] = {}

        # LocationBg에서 singleton 초기화 후 _poll에서 lazy 접근
        self._location_provider = None

        logging.info(
            f"LocationSensor initialized with method={config.localization_method}, "
            f"update_rate={config.update_rate_hz}Hz"
        )

    def _handle_location_update(self, location_data: LocationData) -> None:
        """
        위치 업데이트 콜백 핸들러.

        Parameters
        ----------
        location_data : LocationData
            새 위치 데이터
        """
        self.latest_location = location_data
        self.message_buffer.put(location_data)
        logging.debug(
            f"Location update: pose=({location_data.current_pose}), "
            f"status={location_data.navigation_status}"
        )

    async def _poll(self) -> Optional[LocationData]:
        """
        LocationProvider에서 위치 데이터를 폴링.

        Returns
        -------
        Optional[LocationData]
            위치 데이터가 있으면 반환, 없으면 None
        """
        await asyncio.sleep(1.0 / self.config.update_rate_hz)

        # Lazy singleton access (LocationBg에서 먼저 초기화됨)
        if self._location_provider is None:
            try:
                self._location_provider = LocationProvider()
            except Exception:
                return None

        rec = self._location_provider.get_record()
        if rec.gnss is None and rec.uwb is None:
            return None

        pose = None
        confidence = 0.0
        if rec.gnss and rec.gnss.fixType and rec.gnss.fixType > 0:
            pose = Pose2D(x=rec.gnss.lat, y=rec.gnss.lon, theta=0.0)
            confidence = min(1.0, max(0.0, 1.0 - (rec.gnss.hAcc_m or 99) / 100))

        return LocationData(
            timestamp=time.time(),
            current_pose=pose,
            localization_confidence=confidence,
        )

    async def _raw_to_text(self, raw_input: Optional[LocationData]) -> Optional[Message]:
        """
        위치 데이터를 Message 객체로 변환.

        Parameters
        ----------
        raw_input : Optional[LocationData]
            위치 데이터

        Returns
        -------
        Optional[Message]
            타임스탬프가 포함된 Message 객체
        """
        if raw_input is None:
            return None

        # 위치 정보를 텍스트로 변환
        message_parts = []

        # 현재 위치
        if raw_input.current_location_name:
            message_parts.append(f"Current location: {raw_input.current_location_name}")
        elif raw_input.current_pose:
            pose = raw_input.current_pose
            message_parts.append(f"Position: ({pose.x:.2f}, {pose.y:.2f})")

        # 네비게이션 상태
        if raw_input.is_navigating and raw_input.current_goal:
            goal_name = raw_input.current_goal.name or "target"
            if raw_input.distance_to_goal:
                message_parts.append(
                    f"Navigating to {goal_name}, {raw_input.distance_to_goal:.1f}m remaining"
                )
            else:
                message_parts.append(f"Navigating to {goal_name}")

        # 네비게이션 완료/실패
        if raw_input.navigation_status == "arrived":
            message_parts.append("Arrived at destination")
        elif raw_input.navigation_status == "failed":
            message_parts.append("Navigation failed")

        message = " | ".join(message_parts) if message_parts else "Location unknown"

        return Message(timestamp=raw_input.timestamp, message=message)

    async def raw_to_text(self, raw_input: Optional[LocationData]) -> None:
        """
        위치 데이터를 처리하고 메시지 목록에 추가.

        Parameters
        ----------
        raw_input : Optional[LocationData]
            처리할 위치 데이터
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)
            if raw_input is not None:
                self.latest_location = raw_input

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        최신 버퍼 내용을 LLM 입력 형식으로 포맷팅.

        Returns
        -------
        Optional[str]
            포맷팅된 입력 문자열, 버퍼가 비어있으면 None
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        # 위치 정보를 추가로 풍부하게 포맷팅 (latest_location이 있으면 사용)
        if self.latest_location is not None:
            location = self.latest_location
            info_parts = []

            if location.current_location_name:
                info_parts.append(f"Location: {location.current_location_name}")

            if location.current_pose:
                pose = location.current_pose
                info_parts.append(f"Coordinates: ({pose.x:.2f}, {pose.y:.2f}, θ={pose.theta:.2f})")

            if location.is_navigating:
                info_parts.append("Status: Navigating")
                if location.distance_to_goal:
                    info_parts.append(f"Distance to goal: {location.distance_to_goal:.1f}m")
            else:
                info_parts.append(f"Status: {location.navigation_status}")

            display_text = "\n".join(info_parts) if info_parts else latest_message.message
        else:
            display_text = latest_message.message

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{display_text}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__,
            display_text,
            latest_message.timestamp,
        )

        # 버퍼 초기화
        self.messages = []
        self.latest_location = None
        return result

    # ========== Interface Methods (for other modules) ==========

    def get_current_pose(self) -> Optional[Pose2D]:
        """
        현재 2D 위치 반환.

        Returns
        -------
        Optional[Pose2D]
            현재 위치, 없으면 None
        """
        if self.latest_location:
            return self.latest_location.current_pose
        return None

    def get_current_pose_3d(self) -> Optional[Pose3D]:
        """
        현재 3D 위치 반환.

        Returns
        -------
        Optional[Pose3D]
            현재 3D 위치, 없으면 None
        """
        if self.latest_location:
            return self.latest_location.current_pose_3d
        return None

    def get_distance_to_goal(self) -> Optional[float]:
        """
        목표까지 남은 거리 반환.

        Returns
        -------
        Optional[float]
            목표까지 거리 (meters), 목표가 없으면 None
        """
        if self.latest_location:
            return self.latest_location.distance_to_goal
        return None

    def has_active_goal(self) -> bool:
        """
        활성 네비게이션 목표 존재 여부 반환.

        Returns
        -------
        bool
            활성 목표가 있으면 True
        """
        if self.latest_location and self.latest_location.current_goal:
            return self.latest_location.current_goal.is_active
        return False

    def is_navigating(self) -> bool:
        """
        네비게이션 진행 중 여부 반환.

        Returns
        -------
        bool
            네비게이션 중이면 True
        """
        if self.latest_location:
            return self.latest_location.is_navigating
        return False

    def get_navigation_status(self) -> str:
        """
        네비게이션 상태 반환.

        Returns
        -------
        str
            네비게이션 상태 ("idle", "navigating", "arrived", "failed")
        """
        if self.latest_location:
            return self.latest_location.navigation_status
        return "unknown"

    def get_localization_confidence(self) -> float:
        """
        위치 추정 신뢰도 반환.

        Returns
        -------
        float
            신뢰도 (0.0 ~ 1.0)
        """
        if self.latest_location:
            return self.latest_location.localization_confidence
        return 0.0

    def start(self) -> None:
        """
        Location Sensor 시작.

        LocationProvider를 시작합니다.
        """
        # TODO: Provider 시작 로직
        # self.location_provider.start()
        logging.info("LocationSensor started")

    def stop(self) -> None:
        """
        Location Sensor 정지.

        LocationProvider를 정지합니다.
        """
        # TODO: Provider 정지 로직
        # self.location_provider.stop()
        logging.info("LocationSensor stopped")
