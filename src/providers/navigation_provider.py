# navigation_provider.py
import csv
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from .singleton import singleton
from .location_provider import LocationProvider
from .gnss_route_provider import GnssRouteProvider
from .dwa_route_provider import DwaRouteProvider

logger = logging.getLogger(__name__)

EARTH_R = 6_371_000.0


def _haversine_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat_avg = math.radians((lat1 + lat2) * 0.5)
    north = EARTH_R * dlat
    west = -EARTH_R * math.cos(lat_avg) * dlon
    return math.hypot(west, north)

PathLike = Union[str, Path, List[Tuple[float, float]]]


def _load_waypoints(path: PathLike) -> List[Tuple[float, float]]:
    """
    지원 입력:
      1) [(lat, lon), ...]  직접 전달
      2) CSV: time,lat,lon,hdop,quality 형식 (quality 무시, 모든 행 로드)
      3) TXT: 줄 단위 "lat,lon" 또는 "lat lon"
    """
    if isinstance(path, list):
        return [(float(a), float(b)) for a, b in path]

    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    if not p.exists():
        raise FileNotFoundError(f"waypoint file not found: {p}")

    # CSV (헤더에 lat/lon 컬럼이 있으면)
    try:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "lat" in reader.fieldnames and "lon" in reader.fieldnames:
                coords: List[Tuple[float, float]] = []
                for row in reader:
                    try:
                        coords.append((float(row["lat"]), float(row["lon"])))
                    except (KeyError, ValueError):
                        continue
                logger.info("_load_waypoints: %d waypoints loaded from %s", len(coords), p)
                return coords
    except Exception:
        pass

    # TXT fallback: "lat,lon" or "lat lon"
    coords = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",") if "," in s else s.split()
            if len(parts) < 2:
                continue
            try:
                coords.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    logger.info("_load_waypoints: %d waypoints loaded from %s", len(coords), p)
    return coords


# ==============================================================================
# output state
# ==============================================================================
@dataclass(frozen=True)
class NavigationState:
    t_monotonic: float
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    mode: str = "IDLE"
    heading_calibrated: bool = False
    reached_goal: bool = False


# ==============================================================================
# NavigationProvider
# ==============================================================================
@singleton
class NavigationProvider:
    """
    GnssRouteProvider + DwaRouteProvider를 wrapping하는 Navigation Provider.

    LocationProvider와 동일한 패턴:
      - 두 하위 provider 인스턴스를 주입받아 start/stop 관리
      - worker thread가 DwaRouteRecord를 폴링하여 NavigationState로 통합
      - get_state() / data 로 최신 상태 제공
    """

    def __init__(
        self,
        gnss: GnssRouteProvider,
        dwa: DwaRouteProvider,
        tick_dt: float = 0.05,
        speed_step: float = 0.1,
        speed_min: float = 0.2,
        speed_max: Optional[float] = None,  # None이면 dwa.v_max 사용
    ) -> None:
        self._gnss = gnss
        self._dwa = dwa
        self._tick_dt = float(tick_dt)
        self._speed_step = float(speed_step)
        self._speed_min = float(speed_min)
        self._speed_max = float(speed_max) if speed_max is not None else float(dwa.v_max)
        self._active_path: Optional[PathLike] = None

        self._state_lock = threading.Lock()
        self._latest_state = NavigationState(t_monotonic=time.monotonic())

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running: bool = False

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._gnss.start()
        self._dwa.start()

        self._stop_evt.clear()
        self.running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="NavigationProviderWorker",
        )
        self._thread.start()
        logger.info("NavigationProvider started")

    def stop(self) -> None:
        if self._thread is None:
            return

        self.running = False
        self._stop_evt.set()

        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("NavigationProvider worker thread did not stop within timeout")
        self._thread = None

        self._dwa.stop()
        self._gnss.stop()

        with self._state_lock:
            self._latest_state = NavigationState(
                t_monotonic=time.monotonic(),
                mode="STOP",
            )

        logger.info("NavigationProvider stopped")

    # ---------------- control API ----------------
    def set_path(self, path: PathLike) -> None:
        """경로를 교체하고 재시작합니다."""
        waypoints = _load_waypoints(path)
        if not waypoints:
            raise ValueError(f"NavigationProvider.set_path: no waypoints loaded from {path!r}")

        self.stop()
        self._gnss.waypoints = waypoints
        self._active_path = path
        logger.info("NavigationProvider.set_path: %d waypoints, restarting", len(waypoints))
        self.start()

    def step_faster(self) -> None:
        """속도를 한 단계 높입니다 (최대 speed_max)."""
        current = self._dwa.vx_fixed
        new_speed = min(self._speed_max, round(current + self._speed_step, 2))
        self._dwa.vx_fixed = new_speed
        logger.info("NavigationProvider.step_faster: %.2f -> %.2f m/s", current, new_speed)

    def step_slower(self) -> None:
        """속도를 한 단계 낮춥니다 (최소 speed_min)."""
        current = self._dwa.vx_fixed
        new_speed = max(self._speed_min, round(current - self._speed_step, 2))
        self._dwa.vx_fixed = new_speed
        logger.info("NavigationProvider.step_slower: %.2f -> %.2f m/s", current, new_speed)

    def get_next_move(self) -> Tuple[float, float, float]:
        """현재 이동 명령 (vx, vy, vyaw) 을 반환합니다."""
        st = self.get_state()
        return (st.vx, st.vy, st.vyaw)

    def clear_path(self) -> None:
        """경로를 초기화하고 내비게이션을 중단합니다. 재시작하려면 set_path()를 호출하세요."""
        self.stop()
        self._active_path = None
        logger.info("NavigationProvider.clear_path: path cleared")

    def get_active_path(self) -> Optional[PathLike]:
        """현재 설정된 경로를 반환합니다. 경로가 없으면 None을 반환합니다."""
        return self._active_path

    def get_target_speed(self) -> float:
        """현재 목표 속도 (vx_fixed) 를 반환합니다."""
        return float(self._dwa.vx_fixed)

    def get_remaining_distance(self) -> float:
        """
        현재 위치에서 최종 목표까지의 남은 거리 (m) 를 반환합니다.

        - 현재 위치 → 다음 waypoint: 실시간 계산
        - 이후 waypoint 간격: 사전 합산
        """
        waypoints = self._gnss.waypoints
        if not waypoints:
            return 0.0

        try:
            loc = LocationProvider()
            gnss = loc.get_record().gnss
            if gnss is None:
                return 0.0
            lat, lon = float(gnss.lat), float(gnss.lon)
        except Exception:
            return 0.0

        reach_tol = self._gnss.reach_tol_m

        # 현재 위치 기준으로 이미 통과한 waypoint를 건너뜀
        idx = 0
        while idx < len(waypoints) - 1:
            if _haversine_dist_m(lat, lon, waypoints[idx][0], waypoints[idx][1]) < reach_tol:
                idx += 1
            else:
                break

        # 현재 위치 → 다음 waypoint (실시간)
        rem = _haversine_dist_m(lat, lon, waypoints[idx][0], waypoints[idx][1])

        # 나머지 waypoint 간 구간 합산
        for k in range(idx, len(waypoints) - 1):
            rem += _haversine_dist_m(
                waypoints[k][0], waypoints[k][1],
                waypoints[k + 1][0], waypoints[k + 1][1],
            )

        return float(rem)

    # ---------------- state API ----------------

    def get_state(self) -> NavigationState:
        with self._state_lock:
            return self._latest_state

    @property
    def data(self) -> Optional[Dict[str, Any]]:
        st = self.get_state()
        if not self.running:
            return None
        return {
            "t_monotonic": st.t_monotonic,
            "vx": st.vx,
            "vy": st.vy,
            "vyaw": st.vyaw,
            "mode": st.mode,
            "heading_calibrated": st.heading_calibrated,
            "reached_goal": st.reached_goal,
        }

    # ---------------- worker ----------------

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                rec = self._dwa.data
                gnss_rec = self._gnss.data
                heading_calibrated = bool(
                    gnss_rec.heading_calibrated if gnss_rec is not None else False
                )
                reached_goal = bool(
                    gnss_rec.reached_goal if gnss_rec is not None else False
                )
                if rec is None:
                    st = NavigationState(
                        t_monotonic=time.monotonic(),
                        mode="IDLE",
                        heading_calibrated=heading_calibrated,
                        reached_goal=reached_goal,
                    )
                else:
                    mode = str(rec.mode)
                    if mode == "DWA":
                        vx = float(rec.vx_cmd)
                        vyaw = float(rec.vyaw_cmd)
                    else:
                        # DWA가 정지 — heading 캘리브레이션 중이면 gnss 명령을 직접 전달
                        if gnss_rec is not None and not gnss_rec.heading_calibrated:
                            vx = float(gnss_rec.vx)
                            vyaw = float(gnss_rec.vyaw)
                            mode = "CALIBRATING"
                        else:
                            vx = 0.0
                            vyaw = 0.0
                    st = NavigationState(
                        t_monotonic=time.monotonic(),
                        vx=vx,
                        vy=0.0,
                        vyaw=vyaw,
                        mode=mode,
                        heading_calibrated=heading_calibrated,
                        reached_goal=reached_goal,
                    )
                with self._state_lock:
                    self._latest_state = st
            except Exception:
                logger.exception("Error in NavigationProvider worker loop")

            self._stop_evt.wait(self._tick_dt)
