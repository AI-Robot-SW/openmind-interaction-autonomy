# geometry.py

import math
from typing import Tuple

EARTH_R = 6_371_000.0  # meter


def wrap_deg(a: float) -> float:
    """Wrap angle (degrees) to [-180, 180]"""
    return (a + 180.0) % 360.0 - 180.0


def wrap_rad(a: float) -> float:
    """Wrap angle (radians) to [-pi, pi]"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


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
    """두 WGS84 좌표 간 거리(미터)를 계산한다."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return EARTH_R * 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def latlon_to_enu(
    lat: float, lon: float, origin_lat: float, origin_lon: float
) -> Tuple[float, float]:
    """
    WGS84 lat/lon → 로컬 ENU (x=East, y=North) (m).

    Flat-earth 근사. 원점으로부터 ~10 km 이내에서 유효.
    """
    dlat_rad = math.radians(lat - origin_lat)
    dlon_rad = math.radians(lon - origin_lon)
    x_m = EARTH_R * math.cos(math.radians(origin_lat)) * dlon_rad
    y_m = EARTH_R * dlat_rad
    return x_m, y_m


def enu_to_latlon(
    x_m: float, y_m: float, origin_lat: float, origin_lon: float
) -> Tuple[float, float]:
    """
    로컬 ENU (x=East, y=North) (m) → WGS84 lat/lon.

    latlon_to_enu 의 역변환.
    """
    lat = origin_lat + math.degrees(y_m / EARTH_R)
    lon = origin_lon + math.degrees(
        x_m / (EARTH_R * math.cos(math.radians(origin_lat)))
    )
    return lat, lon


def euclidean_dist_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """두 평면 좌표 간 거리(미터)를 계산한다."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
