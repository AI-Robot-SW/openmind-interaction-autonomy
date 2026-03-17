#!/usr/bin/env python3
"""
실제 주행 테스트 — waypoint 파일을 넣으면 그대로 주행.

전체 provider 스택을 실제 HW로 초기화:
  RealSenseCameraProvider → SegmentationProvider → PointCloudProvider
  → BEVOccupancyGridProvider → DwaRouteProvider
  RtkProvider + NTRIP → LocationProvider → GnssRouteProvider
  → NavigationProvider → UnitreeGo2Provider.move(vx, vy, vyaw)

사용법:
  python run_navigation_test.py \
      --waypoints route.csv \
      --speed 0.4

  Ctrl+C 로 긴급 정지
"""

import argparse
import logging
import math
import signal
import time
import threading

import serial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nav_test")

_running = True


def _sig_handler(sig, frame):
    global _running
    logger.warning("Ctrl+C — 긴급 정지")
    _running = False


def run(args):
    global _running
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    speed_calib = 0.5
    speed_run = 0.8

    # ================================================================
    # 1) UnitreeGo2Provider
    # ================================================================
    from providers.unitree_go2_provider import UnitreeGo2Provider

    go2 = UnitreeGo2Provider(channel=args.channel)
    go2.start()
    time.sleep(1.0)
    logger.info("Go2 stand up...")
    go2.stand_up()
    time.sleep(2.0)

    # ================================================================
    # 2) RtkProvider (GnssProvider + NTRIP) → LocationProvider
    # ================================================================
    from providers.rtk_provider import RtkProvider
    from providers.uwb_provider import UwbProvider
    from providers.location_provider import LocationProvider

    gps_lock = threading.RLock()
    gps_ser = serial.Serial(args.gps_port, args.gps_baud, timeout=1)

    rtk = RtkProvider(
        ser=gps_ser,
        measRate_ms=100,
        caster=args.ntrip_caster,
        port=args.ntrip_port,
        mountpoint=args.ntrip_mount,
        user=args.ntrip_user,
        password=args.ntrip_pw,
        write_lock=gps_lock,
    )

    if args.uwb_port:
        uwb_ser = serial.Serial(args.uwb_port, 115200, timeout=1)
        uwb = UwbProvider(uwb_ser, write_lock=gps_lock)
    else:
        uwb = type("Dummy", (), {
            "start": lambda s: None, "stop": lambda s: None,
            "get_record": lambda s: None,
        })()

    loc = LocationProvider(gnss=rtk, uwb=uwb)
    loc.start()
    logger.info("LocationProvider started")
    
    
    # GPS 수신 대기
    logger.info("RTK GPS 수신 대기...")
    for i in range(600):
        if not _running:
            go2.stand_down(); return
        rec = loc.get_record()
        if rec.gnss and rec.gnss.lat and rec.gnss.lat != 0:
            logger.info("GPS OK: lat=%.7f lon=%.7f carrSoln=%s numSV=%s",
                        rec.gnss.lat, rec.gnss.lon, rec.gnss.carrSoln, rec.gnss.numSV)
            break
        if i % 50 == 0 and i > 0:
            logger.info("  GPS 대기 중... %d초", i // 10)
        time.sleep(0.1)
    else:
        logger.error("GPS 수신 실패 (60초). 종료.")
        go2.stand_down(); return

    # ================================================================
    # 3) RealSense → PointCloud → BEV 파이프라인
    # ================================================================
    from providers.realsense_camera_provider import RealSenseCameraProvider
    from providers.segmentation_provider import SegmentationProvider
    from providers.pointcloud_provider import PointCloudProvider
    from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider
    from providers.distmap_provider import DistMapProvider

    rs = RealSenseCameraProvider()
    rs.start()

    smt = SegmentationProvider()
    smt.start()

    pc = PointCloudProvider()
    pc.start()

    bev = BEVOccupancyGridProvider()
    bev.start()
    
    dist = DistMapProvider()
    dist.start()

    logger.info("RealSense → PointCloud → BEV → DistMap 파이프라인 started")

    # BEV 데이터 올라올 때까지 잠시 대기
    for i in range(100):
        if not _running:
            break
        if bev.data is not None and bev.data.occupancy_grid is not None:
            logger.info("BEV occupancy grid 수신 확인")
            break
        time.sleep(0.1)
    else:
        logger.warning("BEV 데이터 아직 없음 — DWA가 IDLE로 시작될 수 있음")

    # ================================================================
    # 4) Navigation 스택
    # ================================================================
    from providers.gnss_route_provider import GnssRouteProvider
    from providers.dwa_route_provider import DwaRouteProvider
    from providers.navigation_provider import NavigationProvider

    gnss_route = GnssRouteProvider(
        waypoints=[],
        reach_tol_m=args.reach_tol,
        max_vx=speed_run,
    )

    dwa_route = DwaRouteProvider(
        vx_fixed=speed_run,
        v_max=speed_run,
    )

    nav = NavigationProvider(
        gnss=gnss_route,
        dwa=dwa_route,
        tick_dt=0.05,
        speed_step=0.1,
        speed_min=0.15,
        speed_max=speed_run,
    )

    # ================================================================
    # 5) 경로 설정 → 자동 start
    # ================================================================
    logger.info("=" * 60)
    logger.info("경로 로드: %s", args.waypoints)
    nav.set_path(args.waypoints)
    logger.info("주행 시작! Ctrl+C 긴급 정지")
    logger.info("=" * 60)

    # ================================================================
    # 6) 주행 루프 (10Hz)
    # ================================================================
    already_stopped = False
    tick = 0

    try:
        while _running:
            time.sleep(0.1)

            if not nav.running:
                logger.info("NavigationProvider 종료됨")
                break

            state = nav.get_state()
            vx, vy, vyaw = state.vx, state.vy, state.vyaw

            speed_limit = speed_calib if state.mode == "CALIBRATING" else speed_run
            lin_speed = math.hypot(vx, vy)
            if lin_speed > speed_limit and lin_speed > 1e-6:
                scale = speed_limit / lin_speed
                vx *= scale
                vy *= scale

            # 목표 도달
            if state.reached_goal:
                if not already_stopped:
                    go2.stop_move()
                    already_stopped = True
                    logger.info("🎉 목표 도달! 주행 완료.")
                break

            # 속도 명령 전달
            if abs(vx) > 1e-6 or abs(vy) > 1e-6 or abs(vyaw) > 1e-6:
                go2.move(vx, vy, vyaw)
                already_stopped = False
            else:
                if not already_stopped:
                    go2.stop_move()
                    already_stopped = True

            # 1초마다 상태 로그
            tick += 1
            if tick % 10 == 0:
                odom = go2.get_odometry()
                gnss = loc.get_record().gnss
                dwa_rec = dwa_route.data
                rem = nav.get_remaining_distance()
                logger.info(
                    "mode=%-12s vx=%+5.2f vyaw=%+5.2f | "
                    "odom=(%+6.2f,%+6.2f,%+6.1f°) | "
                    "gps=(%.7f,%.7f) carr=%s | "
                    "dwa=%s reason=%s | 남은=%.1fm",
                    state.mode, vx, vyaw,
                    odom.x if odom else 0,
                    odom.y if odom else 0,
                    math.degrees(odom.yaw) if odom else 0,
                    gnss.lat if gnss else 0,
                    gnss.lon if gnss else 0,
                    gnss.carrSoln if gnss else "?",
                    dwa_rec.mode if dwa_rec else "None",
                    dwa_rec.stop_reason if dwa_rec else "",
                    rem,
                )

    except Exception as e:
        logger.error("주행 중 에러: %s", e, exc_info=True)

    finally:
        logger.info("정리 중...")
        try: nav.stop()
        except: pass
        go2.stop_move()
        time.sleep(0.5)
        go2.stand_down()
        time.sleep(1.0)
        try: smt.stop()
        except: pass
        try: dist.stop()
        except: pass
        try: bev.stop()
        except: pass
        try: pc.stop()
        except: pass
        try: rs.stop()
        except: pass
        try: loc.stop()
        except: pass
        logger.info("=== 테스트 종료 ===")


def main():
    p = argparse.ArgumentParser(
        description="실제 주행 테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--channel",      default="eno1")
    p.add_argument("--gps-port",     default="/dev/gps")
    p.add_argument("--gps-baud",     type=int, default=115200)
    p.add_argument("--ntrip-caster", default="rts2.ngii.go.kr")
    p.add_argument("--ntrip-port",   type=int, default=2101)
    p.add_argument("--ntrip-mount",  default="VRS-RTCM32")
    p.add_argument("--ntrip-user",   default="dori0126")
    p.add_argument("--ntrip-pw",     default="ngii")
    p.add_argument("--uwb-port",     default=None)
    p.add_argument("--waypoints",    required=True, help="waypoint CSV/TXT")
    p.add_argument("--speed",        type=float, default=0.4)
    p.add_argument("--reach-tol",    type=float, default=5.0)

    import logging
    from pathlib import Path

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"nav_test_{int(time.time())}.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    run(p.parse_args())


if __name__ == "__main__":
    main()
