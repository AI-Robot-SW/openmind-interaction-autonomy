# OM1 HW/SW Test 가이드라인 (KIST 배포용)

목표: OM1 기반 플랫폼을 KIST 환경에 설치한 뒤, **(1) 소프트웨어 품질(단위/통합)** 과 **(2) 하드웨어 연결성(카메라/오디오/센서/로봇)** 을 빠르게 검증하고, 문제 발생 시 **재현 가능한 로그/결과물**을 수집하기 위한 절차를 제공한다.

---

## 1) 테스트 종류 개요

OM1에는 크게 2가지 테스트 트랙이 있습니다.

### A. SW 테스트(자동화)

- 목적: 코드 레벨에서 회귀(regression) 방지, 기능 단위 검증
- 위치: `OM1/tests/`
- 러너: `pytest` (기본은 integration 제외)
- 특징:
  - 대부분 Mock 기반으로 외부 의존성(로봇/센서/네트워크)을 줄임
  - 일부는 “integration” 마커로 분리되어 별도 실행

### B. HW 테스트(현장 점검/수동)

- 목적: 실제 장비/네트워크/드라이버/권한 문제를 빠르게 확인
- 위치: `OM1/system_hw_test/`
- 특징:
  - “장비별 스모크 테스트 스크립트” 모음
  - 결과는 주로 콘솔 출력/영상 창/간단 파일 저장으로 확인

---

## 2) 실행 전 공통 준비 사항

### 2.1 Python/패키지
- OM1은 `uv` 기반 실행을 권장합니다.
- 기본 실행 예: `uv run src/run.py spot`

### 2.2 환경 변수(.env)
현장/고객사 환경에서 자주 필요한 값:
- `OM_API_KEY`: OM1 클라우드 API/LLM 사용 시
- `ROBOT_IP`: 로봇 제어가 IP 기반인 경우
- `URID`: Zenoh/멀티로봇 네임스페이스에 사용(특히 TurtleBot4)

### 2.3 OS 권한 (가장 흔한 실패 원인)
- 카메라/마이크 권한(특히 macOS)
- 오디오 디바이스 선택/기본 장치 설정(특히 Linux)
- 시리얼/USB 접근 권한(`/dev/tty*`, `/dev/cu*`)

---

## 3) SW 테스트 가이드(자동화)

### 3.1 기본 테스트(권장: 먼저 실행)
OM1의 기본 pytest 옵션은 integration을 제외합니다(`pyproject.toml`의 `addopts`).

```bash
uv run pytest -q
```

### 3.2 특정 영역만 빠르게
예:
```bash
uv run pytest -q tests/llm
uv run pytest -q tests/runtime
uv run pytest -q tests/actions
```

### 3.3 Integration 테스트(선택)
설정 기반으로 “입력 → 프롬프트 → LLM → 액션”의 흐름을 통합적으로 검증합니다.

- 안내 문서: `OM1/tests/integration/README.md`
- 실행:
```bash
uv run pytest -m "integration" tests/integration/test_case_runner.py -v
```

주의:
- 일부 테스트는 환경변수가 필요할 수 있습니다. API가 필요한 모델을 사용하는 경우, 환경변수로서 `OM_API_KEY`가 설정되어 있어야 합니다.
- 테스트가 외부 네트워크/클라우드에 의존하는지 여부는 테스트 케이스에 따라 다릅니다.

---

## 4) HW 테스트 가이드(system_hw_test)

### 4.1 언제 무엇을 실행하나 (추천 순서)

1) **오디오 입력/출력** (대부분의 데모/대화형 에이전트의 최소 조건)
2) **카메라** (VLM/시각 기반 입력을 쓰는 경우)
3) **로봇 상태/데이터 스트림** (Go2/TB4 등)
4) **LIDAR/거리 센서** (자율주행/회피 기능이 있을 때)
5) **BLE/GPS/나침반** (RF mapping, 야외 내비 등 특수 기능)

---

## 5) 장비별 HW 테스트 체크리스트

### 5.1 오디오(PC/노트북)

- 디바이스 나열 (기본 확인)
  - `OM1/system_hw_test/test_audio.py`
  - macOS 전용 참고: `OM1/system_hw_test/test_audio_mac.py`

문서 참고:
- `OM1/docs/examples/conversation.mdx` (오디오 디바이스 확인/권한 관련)

결과물 요청 (오디오 테스트 시 다음과 같은 로그/파일을 요청할 예정입니다.):
- 실행 로그(디바이스 목록, 선택된 default input/output)
- OS 설정 스크린샷(선택): 기본 입력/출력 장치

---

### 5.2 Unitree Go2 (DDS/CycloneDDS + Ethernet)

Go2는 네트워크 인터페이스(예: `en0`) 지정이 중요합니다.

문서 참고:
- `OM1/docs/robotics/unitree_go2_quadruped.mdx`

빠른 점검 스크립트(문서에 명시됨):
```bash
uv run system_hw_test/go2_camera_opencv.py <ethernet_ifname>
uv run system_hw_test/go2_capture_image.py <ethernet_ifname>
uv run system_hw_test/go2_data_stream.py <ethernet_ifname>
uv run system_hw_test/go2_lidar.py <ethernet_ifname>
```

결과물 요청 (Go2 테스트 시 다음과 같은 로그/파일을 요청할 예정입니다.):
- 각 스크립트 실행 로그 전체
- `go2_capture_image.py` 결과 이미지 파일(있다면)
- 네트워크 설정(인터페이스명, IP 설정) 요약

---

### 5.3 TurtleBot4 (Zenoh/URID + RPi/Create3)

문서 참고:
- `OM1/docs/robotics/turtlebot4_zenoh.mdx`

원격(노트북)에서 디버깅용 스크립트(문서에 명시됨):
```bash
uv run system_hw_test/turtlebot4_keyboard_movement.py --URID <URID>
uv run system_hw_test/rptest.py --URID <URID>
uv run system_hw_test/turtlebot4_battery.py --URID <URID>
```

결과물 요청 (TurtleBot4 테스트 시 다음과 같은 로그/파일을 요청할 예정입니다.):
- `ros2 topic list` 출력(URID prefix 포함 여부 확인)
- 각 스크립트 실행 로그

---

### 5.4 Intel RealSense D435 (옵션)

관련 코드 위치:
- `OM1/system_hw_test/intel435/`

권장 확인 항목:
- 장치 인식(USB)
- 스트림 수신/프레임 레이트
- (Zenoh/ROS2 중 무엇을 쓰는지) 실행 스크립트 선택

결과물 요청 (Intel RealSense D435 테스트 시 다음과 같은 로그/파일을 요청할 예정입니다.):
- 실행 로그 + (가능하면) 캡처 이미지 1장

---

### 5.5 BLE / RF mapping (옵션)

문서 참고:
- `OM1/docs/robotics/rf_mapping.mdx`

빠른 점검:
```bash
uv run system_hw_test/ble.py
uv run src/run.py test_ble
```

결과물 요청:
- BLE 스캔 결과 로그(근접 장치 목록/신호세기)
- 생성된 로컬 파일(있다면) 또는 샘플 JSON 1줄

---

### 5.6 GPS/나침반(Arduino 기반, 옵션)

문서 참고:
- `OM1/docs/robotics/gps_compass.mdx`

핵심 포인트:
- 센서 캘리브레이션(자력계/IMU) 없으면 yaw/heading 값이 “그럴듯해도 틀릴 수 있음”
- 시리얼 포트 식별(`/dev/tty*`, `/dev/cu*`)

테스트(OM1 런타임 레벨):
```bash
uv run src/run.py test_gps
```

결과물 요청:
- 시리얼 로그(센서 출력)
- OM1 실행 로그(`test_gps` 동작 결과)

---

## 6) 장애 발생 시 “표준 제출물”(KIST 공통)

테스트 실패 시, 아래를 제공해주시면 재현/원인 분석 속도가 크게 빨라집니다.

### 6.1 필수
- 실행한 명령어(그대로 복사)
- 사용한 config 파일(`OM1/config/<name>.json5`)
- 콘솔 로그 전체(앞/뒤가 끊기지 않게)

### 6.2 가능하면 추가
- OS/하드웨어 정보: OS, CPU/GPU, Python 버전
- 연결 정보: 로봇 IP, 인터페이스명(en0 등), URID
- `ros2 topic list` 또는 Zenoh 라우터 상태(해당 시)
- 사진/영상(카메라/로봇 동작/센서 장착 상태)

---

## 7) 권장 운영 원칙 (안전/재현성)

- “움직임” 테스트는 반드시 안전 구역에서 수행(인명/장비 보호)
- 초기에는 **속도/출력 제한**(최대 속도, 짧은 거리)으로 시작
- 실패 시 즉시 중단할 수 있는 E-Stop/Stop 절차를 사전에 합의
- “성공 로그”를 먼저 확보한 뒤 기능을 단계적으로 추가(오디오→카메라→이동→자율주행)

