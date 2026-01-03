# Vendor Code Analysis

이 문서는 KIST의 Interaction_AutonomousNavigation 레포지토리 코드 분석 결과를 담고 있습니다.

## 목차

1. [ROS2 노드 구조](#ros2-노드-구조)
2. [토픽 및 메시지 타입](#토픽-및-메시지-타입)
3. [데이터 플로우](#데이터-플로우)
4. [OM1 통합 매핑](#om1-통합-매핑)

---

## ROS2 노드 구조

### 1. RealSense Camera Node

**패키지**: `realsense-ros`  
**실행 명령**: 
```bash
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30 \
    enable_depth:=true \
    enable_color:=true \
    pointcloud.enable:=false \
    align_depth.enable:=true
```

**기능**:
- RGB 및 Depth 스트림 출력
- Depth-Color 정렬 수행

**출력 토픽**:
- `/camera/color/image_raw` (sensor_msgs/Image)
- `/camera/depth/image_rect_raw` (sensor_msgs/Image)
- `/camera/aligned_depth_to_color/image_raw` (sensor_msgs/Image)

---

### 2. Segmentation Node

**패키지**: `jetson_quantization-master`  
**실행 명령**: 
```bash
cd <file_path>
python3 demo_trt.py
```

**기능**:
- TensorRT 최적화된 세그멘테이션 모델 실행
- 실시간 클래스 마스크 생성

**입력**: RealSense 카메라 이미지  
**출력**: 세그멘테이션 마스크 (토픽명 확인 필요)

---

### 3. GPU PointCloud Generation Node

**패키지**: `pointcloud_xyzrgb`  
**파일**: `pointcloud_xyzrgb/pointcloud_gpu_xyzrgb.py`  
**실행 명령**: 
```bash
ros2 run pointcloud_xyzrgb pointcloud_gpu_node
```

**기능**:
- CUDA 기반 XYZRGB PointCloud 생성
- Depth 이미지와 세그멘테이션 결과 사용

**입력 토픽**:
- RealSense depth/color 이미지 (토픽명 확인 필요)

**출력 토픽**:
- `/camera/depth_registered/points` (sensor_msgs/PointCloud2)
  - Frame ID: `camera_color_optical_frame`

**특징**:
- CUDA 커널을 사용한 GPU 가속 처리
- RGB 정보를 포함한 PointCloud 생성

---

### 4. BEV Occupancy Grid Node

**패키지**: `bev_cuda`  
**파일**: `bev_cuda/bev_cuda/bev_node_success.py`  
**실행 명령**: 
```bash
ros2 launch bev_cuda bev.launch.py
```

**기능**:
- CUDA 기반 Bird's-Eye-View (상단 뷰) Occupancy Grid 생성
- PointCloud 입력을 받아 2D 격자로 변환

**입력 토픽**:
- `/camera/depth_registered/points` (sensor_msgs/PointCloud2)

**출력 토픽**:
- `/bev/occupancy_grid` (nav_msgs/OccupancyGrid)
- `/bev/image` (sensor_msgs/Image)

**파라미터**:
- `res`: 격자 해상도 (기본값: 0.05m)
- `width`: 격자 너비 (기본값: 80)
- `height`: 격자 높이 (기본값: 80)
- `origin_x`, `origin_y`: 원점 위치
- `dx`, `dy`: 오프셋

**특징**:
- CUDA 커널을 사용한 GPU 가속 처리
- PointCloud를 BEV로 투영하여 장애물 감지

---

### 5. GPS-based Global Planner

**패키지**: `gps_nav`  
**파일**: `gps_nav/gps_nav/planner_server11_success.py`  
**실행 명령**: 
```bash
cd ~/ros2_ws/src/gps_nav/gps_nav
ros2 run gps_nav planner_server
```

**기능**:
- RTK-GPS 데이터와 로봇 오도메트리를 사용하여 전역 목표 위치 계산
- GPS 좌표를 로봇 기준 (x, y) 좌표로 변환
- 경로 추종 제어

**입력**:
- GPS 시리얼 포트 (`/dev/gps`, 115200 baud)
- NTRIP RTK 보정 데이터 (선택)
- `/sportmodestate` (unitree_go/SportModeState) - Go2 오도메트리

**출력 토픽**:
- `/cmd_vel` (geometry_msgs/Twist)
  - `linear.x`: 전진 속도 (m/s)
  - `linear.y`: 횡 방향 속도 (m/s)
  - `angular.z`: 회전 속도 (rad/s)
  - `linear.z`: 제어 신호
    - `-10.0`: 초기화 중 (yaw 보정 전)
    - `-100.0`: 목표 도달
    - `0.0`: 정상 동작
- `/dxdy` (geometry_msgs/Point)
  - `x`: 목표까지 x 방향 오프셋 (m)
  - `y`: 목표까지 y 방향 오프셋 (m)

**특징**:
- GPS 데이터를 내부 스레드로 직접 수신 (UBX/NMEA 파싱)
- NTRIP 클라이언트를 통한 RTK 보정 지원
- Redis를 통한 GPS 상태 브로드캐스팅
- 경로 파일 기반 waypoint 추종
- yaw_offset 자동 보정

**환경 변수**:
- `GPS_SERIAL`: GPS 시리얼 포트 (기본값: `/dev/gps`)
- `GPS_BAUD`: 시리얼 보드레이트 (기본값: `115200`)
- `GO2_TOPIC`: Go2 상태 토픽 (기본값: `/sportmodestate`)
- `CMD_TOPIC`: 명령 토픽 (기본값: `/cmd_vel`)
- `DXDY_TOPIC`: 목표 오프셋 토픽 (기본값: `/dxdy`)
- NTRIP 관련: `caster`, `port`, `mountpoint`, `user`, `password`

---

### 6. DWA Local Planner

**패키지**: `dwa_nav`  
**파일**: `dwa_nav/dwa_nav/dwa_node_success.py`  
**실행 명령**: 
```bash
ros2 run dwa_nav dwa_node
```

**기능**:
- BEV Occupancy Grid 기반 장애물 회피 경로 계산
- 전방 창(window)에서 최소 코스트 셀 선택
- 속도 명령 생성

**입력 토픽**:
- `/bev/occupancy_grid` (nav_msgs/OccupancyGrid) - BEV 장애물 맵
- `/dxdy` (geometry_msgs/Point) - GPS 플래너가 준 목표 오프셋
- `/cmd_vel` (geometry_msgs/Twist) - 외부 제어 명령 (패스스루 모드)

**출력 토픽**:
- `/cmd` (geometry_msgs/Twist)
  - `linear.x`: 전진 속도 (m/s)
  - `angular.z`: 회전 속도 (rad/s)
- `/dwa/local_goal_marker` (visualization_msgs/Marker) - 선택된 로컬 목표 시각화
- `/dwa/dist_grid` (nav_msgs/OccupancyGrid) - 거리 맵 시각화 (선택)

**파라미터**:
- `penalty`: 장애물 페널티 상수 (기본값: 13.0)
- `margin`: 안전 여유 거리 (기본값: 1.2m)
- `dx`, `dy`: GPS가 준 목표 위치 (로봇 기준)
- `ahead_m`: 전방 검사 길이 (기본값: 2.0m)
- `half_width_m`: 좌우 반폭 (기본값: 1.2m)
- `kv`: 거리→전진속도 게인 (기본값: 0.6)
- `kyaw`: 각도→회전속도 게인 (기본값: 1.0)
- `v_max`: 최대 전진 속도 (기본값: 0.9 m/s)
- `w_max`: 최대 회전 속도 (기본값: 0.75 rad/s)
- `person_stop_dist`: 사람 감지 시 정지 거리 (기본값: 1.2m)
- `occ_topic`: Occupancy Grid 토픽 (기본값: `/bev/occupancy_grid`)
- `cmd_topic`: 출력 명령 토픽 (기본값: `/cmd`)

**특징**:
- CUDA 기반 거리 맵 생성 (BFS 또는 Brute-Force)
- 전방 창 내 장애물 회피
- 사람 감지 (occ=88) 시 자동 정지
- `/cmd_vel`의 `linear.z` 값에 따른 패스스루 모드 지원
  - `linear.z = -10.0`: GPS 플래너 직접 제어 모드
  - `linear.z = -100.0`: 최종 목표 도달 정지 신호

---

### 7. Go2 Control Bridge Node

**패키지**: `unitree_ros2`  
**파일**: `unitree_ros2/example/src/src/go2/dwa2go2_node.cpp`  
**실행 명령**: 
```bash
ros2 run unitree_ros2_example dwa2go2_node
```

**기능**:
- DWA 또는 GPS 플래너의 `/cmd` 메시지를 Unitree Go2 제어 명령으로 변환
- Unitree Go2 SportClient API 호출
- Watchdog 기능 (일정 시간 `/cmd` 미수신 시 자동 정지)

**입력 토픽**:
- `/cmd` (geometry_msgs/Twist) - DWA/GPS 플래너 명령
- `lf/sportmodestate` (unitree_go/SportModeState) - Go2 상태 (선택)

**출력**:
- Unitree Go2 SportClient API 호출 (`Move`, `StopMove`)

**특징**:
- Watchdog 타이머 (100ms 주기)
- `/cmd` 미수신 0.3초 이상 시 자동 `StopMove` 호출
- 명령 주파수 모니터링

---

## 토픽 및 메시지 타입

### 주요 ROS2 토픽

| 토픽명 | 메시지 타입 | Publisher | Subscriber | 설명 |
|--------|------------|-----------|------------|------|
| `/camera/color/image_raw` | sensor_msgs/Image | RealSense | Segmentation, PointCloud | RGB 이미지 |
| `/camera/depth/image_rect_raw` | sensor_msgs/Image | RealSense | PointCloud | Depth 이미지 |
| `/camera/depth_registered/points` | sensor_msgs/PointCloud2 | PointCloud | BEV | 정렬된 PointCloud |
| `/bev/occupancy_grid` | nav_msgs/OccupancyGrid | BEV | DWA | BEV 장애물 맵 |
| `/bev/image` | sensor_msgs/Image | BEV | - | BEV 이미지 시각화 |
| `/dxdy` | geometry_msgs/Point | GPS Planner | DWA | 목표 오프셋 (x, y) |
| `/cmd_vel` | geometry_msgs/Twist | GPS Planner | DWA | GPS 플래너 명령 |
| `/cmd` | geometry_msgs/Twist | DWA | Go2 Bridge | 최종 제어 명령 |
| `/sportmodestate` | unitree_go/SportModeState | Go2 | GPS Planner | Go2 오도메트리 |
| `/dwa/local_goal_marker` | visualization_msgs/Marker | DWA | - | DWA 로컬 목표 시각화 |
| `/dwa/dist_grid` | nav_msgs/OccupancyGrid | DWA | - | 거리 맵 시각화 |

### 메시지 타입 상세

#### geometry_msgs/Twist
```python
linear:
  x: float  # 전진 속도 (m/s)
  y: float  # 횡 방향 속도 (m/s)
  z: float  # 제어 신호 (-10.0: 초기화, -100.0: 목표 도달)
angular:
  z: float  # 회전 속도 (rad/s)
```

#### geometry_msgs/Point
```python
x: float  # x 방향 오프셋 (m)
y: float  # y 방향 오프셋 (m)
z: float  # 미사용
```

#### nav_msgs/OccupancyGrid
```python
header:
  frame_id: string  # 좌표계 (예: "base_link")
info:
  resolution: float  # 격자 해상도 (m)
  width: int        # 격자 너비
  height: int       # 격자 높이
  origin: Pose      # 원점 위치
data: int8[]        # 격자 데이터 (0: free, 100: occupied, -1: unknown, 88: person)
```

#### unitree_go/SportModeState
```python
position: float[]  # [x, y, z] 위치 (m)
imu_state:
  rpy: float[]     # [roll, pitch, yaw] 오일러 각 (rad)
```

---

## 데이터 플로우

### 전체 시스템 플로우

```
┌─────────────────┐
│ RealSense Camera│
│  (RGB + Depth)  │
└────────┬────────┘
         │
         ├─→ Segmentation Node
         │   (TensorRT)
         │
         └─→ PointCloud Node (CUDA)
             │
             └─→ /camera/depth_registered/points
                 │
                 └─→ BEV Node (CUDA)
                     │
                     └─→ /bev/occupancy_grid
                         │
                         └─→ DWA Local Planner
                             │
                             ├─→ /dxdy (GPS Planner)
                             │
                             └─→ /cmd
                                 │
                                 └─→ Go2 Control Bridge
                                     │
                                     └─→ Unitree Go2 Robot
```

### GPS Navigation 플로우

```
GPS Serial Port
    │
    ├─→ UBX/NMEA Parser (내부 스레드)
    │   │
    │   └─→ GPS 좌표 (lat, lon)
    │
    └─→ NTRIP Client (RTK 보정, 선택)
        │
        └─→ RTCM 데이터

/sportmodestate (Go2 Odometry)
    │
    └─→ GPS Planner
        │
        ├─→ 경로 파일 로드
        ├─→ GPS → 로봇 좌표 변환
        ├─→ 경로 추종 제어
        │
        └─→ /cmd_vel, /dxdy
```

### DWA Obstacle Avoidance 플로우

```
/bev/occupancy_grid
    │
    └─→ DWA Node
        │
        ├─→ CUDA 거리 맵 생성
        ├─→ 전방 창 검사
        ├─→ 최소 코스트 셀 선택
        ├─→ 속도 명령 계산
        │
        └─→ /cmd

/dxdy (GPS 목표)
    │
    └─→ DWA Node (목표 위치 참조)
```

---

## OM1 통합 매핑

### Perception Inputs (PER-INPUT)

| Vendor Node | OM1 Module | 파일 위치 |
|------------|-----------|----------|
| RealSense Camera | `RealsenseSensor` | `src/inputs/plugins/realsense.py` |
| Segmentation | `SegmentationSensor` | `src/inputs/plugins/segmentation.py` |
| PointCloud | `PointCloudSensor` | `src/inputs/plugins/pointcloud.py` |
| BEV Occupancy Grid | `BEVSensor` | `src/inputs/plugins/bev.py` |

### Navigation Actions (NAV-ACTION)

| Vendor Node | OM1 Module | 파일 위치 |
|------------|-----------|----------|
| GPS Global Planner | `NavigateGPSConnector` | `src/actions/navigate_gps/connector/ros2.py` |
| DWA Local Planner | `NavigateDWAConnector` | `src/actions/navigate_dwa/connector/ros2.py` |
| Go2 Control Bridge | `NavigateGo2Connector` | `src/actions/navigate_go2/connector/ros2.py` |

### Providers (SYS-PROVIDER)

| Vendor Node | OM1 Provider | 파일 위치 |
|------------|-------------|----------|
| RealSense Camera | `RealsenseProvider` | `src/providers/realsense_provider.py` |
| Segmentation | `SegmentationProvider` | `src/providers/segmentation_provider.py` |
| PointCloud | `PointCloudProvider` | `src/providers/pointcloud_provider.py` |
| BEV Occupancy Grid | `BEVProvider` | `src/providers/bev_provider.py` |
| GPS Global Planner | `GPSNavProvider` | `src/providers/gps_nav_provider.py` |
| DWA Local Planner | `DWANavProvider` | `src/providers/dwa_nav_provider.py` |
| Go2 Control Bridge | `Go2ControlProvider` | `src/providers/go2_control_provider.py` |

### Backgrounds (SYS-BG)

| Vendor Node | OM1 Background | 파일 위치 |
|------------|---------------|----------|
| RealSense Camera | `RealsenseBackground` | `src/backgrounds/plugins/realsense.py` |
| Segmentation | `SegmentationBackground` | `src/backgrounds/plugins/segmentation.py` |
| PointCloud | `PointCloudBackground` | `src/backgrounds/plugins/pointcloud.py` |
| BEV Occupancy Grid | `BEVBackground` | `src/backgrounds/plugins/bev.py` |
| GPS Global Planner | `GPSNavigationBackground` | `src/backgrounds/plugins/gps_navigation.py` |
| DWA Local Planner | `DWANavigationBackground` | `src/backgrounds/plugins/dwa_navigation.py` |
| Go2 Control Bridge | `Go2ControlBackground` | `src/backgrounds/plugins/go2_control.py` |

---

## 통합 시 고려사항

### 1. ROS2 토픽 통신

- OM1 Provider는 ROS2 토픽을 구독/발행하여 vendor 노드와 통신
- Zenoh를 통한 DDS 통신도 지원 가능

### 2. GPS 시리얼 통신

- GPS Provider는 시리얼 포트를 직접 열어 GPS 데이터 수신
- NTRIP 클라이언트 기능 포함 필요

### 3. CUDA 의존성

- PointCloud 및 BEV 노드는 CUDA를 사용
- GPU 환경에서만 동작 가능

### 4. 경로 파일 관리

- GPS Planner는 경로 파일을 로드하여 waypoint 추종
- 경로 파일 형식 및 위치 확인 필요

### 5. 제어 신호 프로토콜

- `/cmd_vel`의 `linear.z` 값으로 제어 모드 전환
- GPS Planner와 DWA 간 협조 필요

---

## 다음 단계

1. **각 모듈별 상세 구현 계획 수립**
2. **ROS2 메시지 타입 정의 확인**
3. **경로 파일 형식 분석**
4. **통합 테스트 시나리오 작성**

