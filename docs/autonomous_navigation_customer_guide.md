# OM1 + Autonomous_Navigation 통합을 위한 고객사 가이드 (공부 범위/실험 요청)

목표: 고객사가 OM1을 처음 접하더라도, **Autonomous_Navigation을 OM1 플랫폼에 대입(통합)한 배포물**을 실행·검증하고, 필요한 결과(로그/토픽/재현 절차)를 제공할 수 있도록 OM1의 **필수 모듈**만 선별해서 설명합니다.

---

## 0. 전제(이번 통합에서 “Autonomous_Navigation”이 의미하는 것)

- Autonomous_Navigation은 OM1 외부(또는 OM1에 포함된 별도 폴더/서비스)에서 실행되는 **자율주행 스택(로컬 플래너/상태추정/센서 처리 등)** 으로 가정합니다.
- OM1은 “의사결정/명령 생성” 역할을 담당하고, 실제 이동 실행은 Autonomous_Navigation이 담당합니다.

통합에서 가장 중요한 것은 **OM1 ↔ Autonomous_Navigation 사이의 인터페이스(토픽/메시지/상태)** 입니다.

---

## 1. 고객사가 알아야 할 OM1 핵심 개념(최소)

OM1은 다음 루프를 반복합니다.

1) **Inputs**가 센서/상태를 텍스트로 요약해 버퍼에 저장  
2) **Fuser**가 “시스템 프롬프트 + 입력 요약 + 가능한 Actions 목록”을 한 프롬프트로 합침  
3) **LLM plugin**이 프롬프트를 받아 “수행할 Actions”를 생성  
4) **Action Orchestrator**가 Actions를 커넥터(ROS2/Zenoh/SDK)로 전달해 실제 실행  

관련 코드:
- 런타임 루프: `OM1/src/runtime/single_mode/cortex.py`
- 프롬프트 합성: `OM1/src/fuser/__init__.py`
- 액션 실행: `OM1/src/actions/orchestrator.py`

---

## 2. Autonomous_Navigation 통합 시 “OM1에서” 봐야 하는 레이어

Autonomous_Navigation은 보통 아래 두 방식 중 하나로 OM1과 연결됩니다.

### 2.1 속도 명령 기반(velocity control)

OM1이 “앞으로/회전/정지” 같은 기본 명령을 만들고, 커넥터가 `cmd_vel`(예: `geometry_msgs/Twist`)로 변환하여 Autonomous_Navigation에 전달하는 방식입니다.

- OM1 쪽 핵심: `agent_actions` 중 `move*` 계열 액션 + 해당 connector
- OM1 참고 구현(탐색/회피 기반):  
  - Go2: `OM1/src/actions/move_go2_autonomy/connector/unitree_sdk_advance.py`  
  - TurtleBot4: `OM1/src/actions/move_turtle/connector/zenoh.py`

### 2.2 목표점 기반(goal navigation / Nav2 스타일)

OM1이 “테이블로 가”, “A 지점으로 가”처럼 목적지 기반 명령을 만들고, 커넥터가 goal pose를 publish 하며 상태를 모니터링하는 방식입니다.

- OM1 쪽 핵심: `navigate_location` 액션 + navigation provider
- OM1 참고 구현:
  - 액션 커넥터: `OM1/src/actions/navigate_location/connector/unitree_go2_nav.py`
  - goal pose publish / status subscribe: `OM1/src/providers/unitree_go2_navigation_provider.py`
  - 목적지 목록(HTTP): `OM1/src/providers/unitree_go2_locations_provider.py`

---

## 3. 고객사가 공부해야 할 “필수 모듈” 목록(우선순위)

### 3.1 설정(JSON5)과 실행 진입점

- 설정 파일 위치: `OM1/config/*.json5`
  - 예시(자율 탐색 기반): `OM1/config/unitree_go2_autonomy.json5`, `OM1/config/turtlebot4_lidar.json5`
  - goal navigation 포함: `OM1/config/unitree_go2_autonomy_advance.json5` (`navigate_location` 포함)
- 실행 CLI: `OM1/src/run.py`
  - 일반 실행: `uv run src/run.py <config_name>`
- 설정 로더(중요): `OM1/src/runtime/single_mode/config.py`
  - `.env`의 `OM_API_KEY`, `ROBOT_IP`, `URID` 등을 설정에 주입할 수 있습니다.

### 3.2 자율주행 관련 Inputs(센서/상태 → LLM 입력)

자율주행 통합에서 고객사가 가장 자주 확인해야 하는 Input들입니다.

- LIDAR 요약(장애물/가용 경로)
  - 플러그인: `OM1/src/inputs/plugins/rplidar.py`
  - 프로바이더: `OM1/src/providers/rplidar_provider.py`
- Odom/자세(이동 중 여부, 자세)
  - 플러그인: `OM1/src/inputs/plugins/odom.py`
  - 프로바이더: `OM1/src/providers/odom_provider.py`
- 경로 가능성(간단 경로 선택, Zenoh 기반)
  - 플러그인: `OM1/src/inputs/plugins/simple_paths.py`
  - 프로바이더: `OM1/src/providers/simple_paths_provider.py`
- Localization 상태(“네비게이션 안전 실행 가능 여부”를 LLM에 명시)
  - AMCL 기반: `OM1/src/inputs/plugins/amcl_localization_input.py`
  - Lidar localization 기반: `OM1/src/inputs/plugins/lidar_localization_input.py`

### 3.3 자율주행 관련 Actions(OM1 → 실제 제어)

- 자율 탐색/회피(로봇별 구현)
  - Go2 이동 액션: `OM1/src/actions/move_go2_autonomy/`
  - TurtleBot4 이동 액션: `OM1/src/actions/move_turtle/`
- 목적지 네비게이션(저장 위치로 이동)
  - 인터페이스: `OM1/src/actions/navigate_location/interface.py`
  - Go2 커넥터: `OM1/src/actions/navigate_location/connector/unitree_go2_nav.py`

### 3.4 네비게이션 상태/목적지 제공(Providers)

goal 기반 통합을 쓰는 경우 고객사가 반드시 알아야 합니다.

- 네비게이션 상태 구독 + goal pose publish: `OM1/src/providers/unitree_go2_navigation_provider.py`
  - Nav2 action status를 모니터링하고, 성공 시에만 AI 모드를 재활성화하는 로직이 포함되어 있습니다.
- 목적지 목록(HTTP API 폴링): `OM1/src/providers/unitree_go2_locations_provider.py`
  - 기본 URL: `http://localhost:5000/maps/locations/list`

---

## 4. 고객사 실험 요청 사항(플랫폼 검증용)

고객사가 OM1을 잘 몰라도 수행 가능한 형태로 제시합니다.

### 4.1 공통 사전 준비(필수)
- 실행 환경 정보 공유: OS/CPU/GPU, Python, 네트워크(폐쇄망 여부)
- 사용 장비 유형 확정: Go2/TurtleBot4/기타
- 통합 인터페이스 확정:
  - goal 기반인지(`goal_pose`, `navigate_to_pose/_action/status` 등)
  - velocity 기반인지(`cmd_vel` 등)

### 4.2 OM1 단독(기본 자율주행 로직) 스모크 테스트(필수, 10~20분)
목적: 고객 환경에서 OM1 자체의 입력/액션 루프가 정상인지 확인

- Go2(예): `uv run src/run.py unitree_go2_autonomy`
- TurtleBot4(예): `uv run src/run.py turtlebot4_lidar`

제출물:
- 사용한 config 파일 사본
- 콘솔 로그(입력 텍스트/액션 출력이 보이도록)

### 4.3 goal 기반 네비게이션(`navigate_location`) 테스트(통합형, 30~60분)
목적: “목적지 → goal pose publish → 상태 모니터링”이 연결되는지 확인

관련 설정/코드:
- 설정 예시: `OM1/config/unitree_go2_autonomy_advance.json5`
- 액션 커넥터: `OM1/src/actions/navigate_location/connector/unitree_go2_nav.py`
- 상태/goal provider: `OM1/src/providers/unitree_go2_navigation_provider.py`

고객사에게 요청할 확인 항목:
- 목적지 목록 API가 살아있는지(기본: `http://localhost:5000/maps/locations/list`)
- “Navigate to the <location>” 형태 발화 시 `navigate_location` 액션이 호출되는지
- 네비게이션 상태 토픽에서 SUCCEEDED가 들어오는지(성공 시 AI mode 재활성화)

제출물:
- 사용한 목적지 목록 응답 샘플(JSON)
- nav 상태 로그(ACCEPTED/EXECUTING/SUCCEEDED 등)

### 4.4 velocity 기반 연동 테스트(선택)
목적: OM1이 만든 이동 명령이 Autonomous_Navigation 입력 채널로 전달되는지 확인

고객사에게 요청할 확인 항목:
- Autonomous_Navigation 측에서 `cmd_vel`(또는 해당 입력 토픽)을 수신하는지
- 정지/회피가 즉시 반영되는지(Stop 우선권)

제출물:
- `ros2 topic list` 결과(관련 토픽 포함)
- `ros2 topic echo <cmd_topic>` 캡처(짧게)

---

## 5. 고객사 FAQ(자주 막히는 지점)

### Q1. “OM1은 돌고 있는데 네비게이션이 시작되지 않아요”
- goal 기반이라면 `LocationsProvider`(HTTP)에서 목적지를 못 받아오면 `navigate_location`이 실행되지 않습니다.
  - 관련: `OM1/src/providers/unitree_go2_locations_provider.py`

### Q2. “목적지로 가는 중에 AI가 갑자기 조용해졌어요”
- `unitree_go2_navigation_provider.py`는 네비게이션 중 AI 모드를 비활성화하고, 성공(SUCCEEDED) 때만 다시 켭니다.
  - 관련: `OM1/src/providers/unitree_go2_navigation_provider.py`

### Q3. “Localization이 안 되어 네비게이션이 위험하다고 나와요”
- localization input 플러그인이 “NOT LOCALIZED”를 LLM에 전달합니다.
  - AMCL: `OM1/src/inputs/plugins/amcl_localization_input.py`
  - Lidar localization: `OM1/src/inputs/plugins/lidar_localization_input.py`

---

## 6. (부록) 설정에서 사용 가능한 모듈 목록 확인

고객사가 “설정에서 선택 가능한 input/llm/action이 무엇인지” 빠르게 확인하고 싶을 때:
- 스키마 생성: `OM1/scripts/generate_schema.py`
  - 실행 후 `OM1/OM1_config_schema.json5`가 생성됩니다.

