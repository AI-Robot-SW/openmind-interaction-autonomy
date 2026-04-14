# geometry.py

import math
from typing import Tuple

EARTH_R = 6_371_000.0  # meter


def wrap_deg(a: float) -> float:
    """Wrap angle (degrees) to [-180, 180]"""
    return (a + 180.0) % 360.0 - 180.0


def latlon_to_west_north_offset_m(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> Tuple[float, float]:
    """
    (lat1_deg, lon1_deg)에서 (lat2_deg, lon2_deg)까지의 변위를
    (west_m, north_m) 로컬 평면 근사 벡터로 반환한다.
    짧은 거리(예: 100 m 내외)에서 사용한다.
    """
    dlat_rad = math.radians(lat2_deg - lat1_deg)
    dlon_rad = math.radians(lon2_deg - lon1_deg)
    north_m = EARTH_R * dlat_rad
    west_m = -EARTH_R * math.cos(math.radians(lat1_deg)) * dlon_rad
    return west_m, north_m


def haversine_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 WGS84 좌표 간 거리(미터)를 Haversine 공식으로 계산한다."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return EARTH_R * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """두 2D 좌표 간 유클리드 거리(미터)를 계산한다."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
