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


def euclidean_dist_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """두 평면 좌표 간 거리(미터)를 계산한다."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


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


def latlon_to_body_frame(
    cur_lat: float, cur_lon: float, theta_rad: float,
    tgt_lat: float, tgt_lon: float,
) -> Tuple[float, float]:
    """
    목표 lat/lon을 body frame (dx=전방, dy=좌방) 으로 변환.

    Parameters
    ----------
    cur_lat, cur_lon : 현재 위치 (WGS84)
    theta_rad        : 현재 heading (rad, ENU 기준 — East=0, CCW positive)
    tgt_lat, tgt_lon : 목표 위치 (WGS84)

    Returns
    -------
    dx : 전방 거리 (m)
    dy : 좌방 거리 (m)
    """
    east_m, north_m = latlon_to_enu(tgt_lat, tgt_lon, cur_lat, cur_lon)
    dx = math.cos(theta_rad) * east_m + math.sin(theta_rad) * north_m
    dy = -math.sin(theta_rad) * east_m + math.cos(theta_rad) * north_m
    return dx, dy


def xy_to_body_frame(
    cur_x: float, cur_y: float, theta_rad: float,
    tgt_x: float, tgt_y: float,
) -> Tuple[float, float]:
    """
    월드 좌표 목표를 body frame (dx=전방, dy=좌방) 으로 변환.

    Parameters
    ----------
    cur_x, cur_y : 현재 위치 (m)
    theta_rad    : 현재 heading (rad)
    tgt_x, tgt_y : 목표 위치 (m)

    Returns
    -------
    dx : 전방 거리 (m)
    dy : 좌방 거리 (m)
    """
    dx_w = tgt_x - cur_x
    dy_w = tgt_y - cur_y
    dx = math.cos(theta_rad) * dx_w + math.sin(theta_rad) * dy_w
    dy = -math.sin(theta_rad) * dx_w + math.cos(theta_rad) * dy_w
    return dx, dy
