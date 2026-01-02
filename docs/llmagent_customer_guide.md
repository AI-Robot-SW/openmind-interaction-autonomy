# OM1 + LLMAgent 통합을 위한 고객사 가이드

목표: 고객사가 OM1을 처음 접하더라도, **LLMAgent를 OM1에 대입(통합)한 플랫폼**을 실행·검증하고, 필요한 결과(로그/환경/재현 절차)를 제공할 수 있도록 OM1의 **필수 모듈**만 선별해서 설명하는 것을 목표로 하는 가이드입니다.

---

## 0. 전제(이번 통합에서 “LLMAgent”가 의미하는 것)

본 문서에서 말하는 LLMAgent는 기존 레거시 구현(ROS2 기반, LLM + Tool 선택 + FSM 실행)을 의미합니다.

- 레거시 위치(참고): `Interaction_AutonomousNavigation/LLMagent/`
- 레거시 핵심 흐름: `/user_question` → LLM tool 선택(`/llm/selected_tool`) → FSM(`/fsm/nav_cmd`, `/fsm/tts`, `/fsm/state`)

OM1 통합 목표는, 이 “LLMAgent 스타일(도구 기반 계획/안전한 실행)”을 OM1의 플러그인 구조(Inputs/Fuser/LLM/Actions)에 맞게 **내장/대체**하는 것입니다.

---

## 1. 고객사가 알아야 할 OM1 핵심 개념 (최소)

OM1은 다음 루프를 반복합니다.

1) **Inputs**가 센서/음성/상태를 텍스트로 축약해 버퍼에 저장  
2) **Fuser**가 “시스템 프롬프트 + 입력 요약 + 가능한 Actions 목록”을 한 프롬프트로 합침  
3) **LLM plugin**이 프롬프트를 받아 “수행할 Actions”를 생성  
4) **Action Orchestrator**가 Actions를 커넥터(ROS2/Zenoh/SDK 등)로 전달해 실제 실행  

코드 기준 전체 흐름:
- 런타임 루프: `OM1/src/runtime/single_mode/cortex.py`
- 입력 통합: `OM1/src/fuser/__init__.py`
- 액션 실행: `OM1/src/actions/orchestrator.py`

---

## 2. 고객사가 공부해야 할 “필수 모듈” 목록 (우선순위)

아래 파일들만 이해하면, OM1을 몰라도 LLMAgent 통합 버전을 실행/검증하는 데 충분합니다.

### 2.1 설정(JSON5)과 실행 진입점

- 설정 파일 위치: `OM1/config/*.json5`
  - 예시(가장 쉬움): `OM1/config/spot.json5`
  - 실제 로봇/시뮬용: `OM1/config/unitree_go2_autonomy*.json5`, `OM1/config/turtlebot4*.json5`
- 실행 CLI: `OM1/src/run.py`
  - 문서들에서는 `uv run src/run.py <config_name>` 형태를 사용합니다.
  - 만약 실행이 안 되면 `uv run src/run.py start <config_name>`도 시도해 주세요(환경에 따라 CLI 파싱 차이가 있을 수 있음).
- 설정 로더(중요): `OM1/src/runtime/single_mode/config.py`
  - `.env`의 `OM_API_KEY`, `ROBOT_IP`, `URID` 등을 설정 파일에 자동 주입할 수 있습니다.

### 2.2 Inputs(음성/상태가 어떻게 LLM 입력이 되는가)

- 입력 오케스트레이터: `OM1/src/inputs/orchestrator.py`
  - 여러 입력 플러그인을 동시에 listen하여 최신 입력 버퍼를 갱신합니다.
- Input 기본 인터페이스: `OM1/src/inputs/base/__init__.py`
  - 각 Input 플러그인은 `formatted_latest_buffer()`로 “LLM에 넣을 텍스트 한 줄/문단”을 제공합니다.

고객사 실험에서 중요한 이유:
- “음성이 잘 들어왔는지 / 어떤 텍스트로 변환되었는지”가 액션의 품질을 좌우합니다.

### 2.3 Fuser(LLM에게 어떤 프롬프트가 실제로 전달되는가)

- 프롬프트 합성: `OM1/src/fuser/__init__.py`
  - 시스템 프롬프트(`system_prompt_base`, `system_governance`, `system_prompt_examples`)
  - 최신 입력(ASR/VLM/GPS 등)
  - 가능한 Action 목록(설명 포함)
  - 위 3가지를 합쳐 최종 prompt를 만듭니다.

### 2.4 LLM plugin(LLMAgent가 “대입”될 자리)

- LLM 베이스/로더: `OM1/src/llm/__init__.py`
  - `cortex_llm.type`(설정)로 플러그인을 선택합니다.
  - 선택된 LLM은 `ask(prompt)`를 통해 OM1 루프에 액션을 돌려줍니다.
- 대표 플러그인 예시(OpenAI + function calling): `OM1/src/llm/plugins/openai_llm.py`
- 액션 스키마 생성/변환:
  - 함수 스키마 생성: `OM1/src/llm/function_schemas.py`
  - OM1 출력 모델: `OM1/src/llm/output_model.py` (`CortexOutputModel(actions=[...])`)

고객사 실험에서 중요한 이유:
- “LLMAgent 대입”은 결국 **여기(LLM plugin layer)** 를 교체/추가하는 형태가 됩니다.

### 2.5 Actions(실제 로봇/시뮬레이터로 나가는 명령)

- 액션 로더/설명: `OM1/src/actions/__init__.py`
- 실행 오케스트레이터: `OM1/src/actions/orchestrator.py`
  - LLM이 반환한 `Action(type, value)`를, 설정된 `agent_actions` 목록과 매칭해 커넥터에 전달합니다.
- 예시(스피치):
  - 인터페이스: `OM1/src/actions/speak/interface.py`
  - ROS2 커넥터(현재는 데모 로그 출력): `OM1/src/actions/speak/connector/ros2.py`
- 예시(이동/자율주행 계열):
  - Turtlebot4 Zenoh 이동(실제에 가까운 예시): `OM1/src/actions/move_turtle/connector/zenoh.py`
  - Go2 자율주행(유닛리 SDK): `OM1/src/actions/move_go2_autonomy/connector/unitree_sdk_advance.py`

고객사 실험에서 중요한 이유:
- “LLM이 무엇을 말했는지”보다 “액션 커넥터가 어떤 채널로 어떤 명령을 내보냈는지”가 최종 결과입니다.

### 2.6 디버깅 관점(IOProvider)

- 입력/프롬프트/타이밍 저장소: `OM1/src/providers/io_provider.py`
  - 무엇이 입력됐고, 어떤 프롬프트가 생성됐고, LLM 호출이 얼마나 걸렸는지 확인할 때 핵심입니다.

---

## 3. LLMAgent를 OM1에 대입할 때(고객사 입장에서) 봐야 하는 설정 포인트

고객사는 보통 “코드를 수정”하기보다 “설정(config)”만 만지게 됩니다. 아래만 확인하면 됩니다.

### 3.1 `config/*.json5`에서 확인할 항목

- `agent_inputs`: 고객사가 사용할 입력(ASR/VLM/GPS/상태 등)
- `cortex_llm`: 어떤 LLM 플러그인을 쓰는지
- `agent_actions`: 어떤 액션이 노출되는지(= LLM이 호출 가능한 능력)

예시 파일:
- `OM1/config/spot.json5` (WebSim + move/speak/emotion)
- `OM1/config/unitree_go2_autonomy_sim.json5` (Go2 sim 환경 입력/액션 조합)
- `OM1/config/turtlebot4.json5` (Zenoh 기반 이동)

### 3.2 “Autonomous_Navigation(외부 내비 스택)”과의 상호작용은 어디서 일어나는가?

OM1 관점에서 외부 자율주행 스택과의 연동은 보통 **Action connector** 레이어에서 일어납니다.

- OM1 액션(`move`, `navigate_location` 등) → (커넥터가) ROS2/Zenoh/SDK로 publish/call
- 고객사는 최소한 아래를 확정해 주셔야 합니다.
  - 목표 시스템이 ROS2인지/Zenoh인지/SDK인지
  - 외부 시스템이 받아들이는 토픽/메시지 타입(예: `/cmd_vel` `geometry_msgs/Twist`)
  - 안전 정책(최대 속도, E-stop, 충돌 회피 우선권)

---

## 4. 고객사 실험 요청 사항(플랫폼 검증용)

고객사가 OM1을 잘 몰라도 수행 가능한 형태로 제시합니다.

### 4.1 공통 사전 준비(필수)
- 실행 환경 정보 공유
  - OS/CPU/GPU, Python 버전, ROS2 여부, 네트워크 제한(폐쇄망 여부)
- 키/환경변수 설정
  - `.env` 또는 설정 파일에 `OM_API_KEY`(클라우드 LLM 사용 시) / 로봇 IP / URID
- 실행 로그 제공 방식 합의
  - 콘솔 로그(텍스트) + (가능하면) 화면 녹화(WebSim 또는 로봇)

### 4.2 OM1 기본 동작 확인(필수, 10분)
목적: “입력→프롬프트→LLM→액션” 루프가 정상인지 확인

- 실행(예시)
  - `uv run src/run.py spot`
- 확인 항목
  - LLM 호출이 정상인지(오류 없이 반복되는지)
  - 액션이 생성되는지(예: speak/move)
  - WebSim 또는 로그에서 입력/액션이 갱신되는지

### 4.3 LLMAgent 통합 버전 검증(필수, 30~60분)
목적: “LLMAgent 대입” 후, 고객 환경에서 의도대로 액션이 나오는지 확인

고객사에게 요청할 테스트 발화(예시):
- 이동/정지: “앞으로 가”, “멈춰”
- 회전: “왼쪽으로 돌아”, “오른쪽으로 돌아”
- 대화: “지금 상태 알려줘”, “오늘 날씨 어때?”(일반 대화)
- (RAG가 포함된 경우) “회사/시설 안내 질문 3개”

제공해 주실 결과물:
- 사용한 설정 파일(`config/*.json5`)
- 콘솔 로그(LLM 출력 + 액션 실행 흔적)
- (가능하면) 실행 화면/로봇 동작 영상
- 문제가 있었다면 “재현 절차(어떤 발화를 어떤 순서로 했는지)”

### 4.4 자율주행 스택(Autonomous_Navigation) 연동 검증(선택)
목적: 이동 명령이 실제 네비 스택 입력 채널로 전달되는지 검증

고객사에게 요청할 확인 항목:
- 외부 내비 스택이 수신하는 토픽/채널에서 메시지 수신 여부
- 충돌 회피/정지 우선권 동작 여부(Stop이 즉시 반영되는지)
- 안전 제한(최대 속도/거리/금지구역) 적용 여부

---

## 5. 고객사 FAQ(자주 막히는 지점)

### Q1. 설정 파일에서 뭘 바꿔야 하나요?
- 대부분은 `OM1/config/<agent>.json5`에서
  - `agent_inputs`(센서/ASR),
  - `cortex_llm`(LLMAgent 대입 플러그인),
  - `agent_actions`(move/speak/nav)
  만 조정합니다.

### Q2. “LLM은 동작하는데 로봇이 안 움직여요”
- OM1은 “액션”까지만 만들고, 실제 이동은 **Action connector**가 담당합니다.
- 고객 환경에서 로봇/내비 스택이 받아들이는 채널(ROS2/Zenoh/SDK)에 맞는 커넥터가 선택/구현되어 있어야 합니다.
  - 참고: `OM1/src/actions/move_turtle/connector/zenoh.py` (실제 publish 예시)

### Q3. 디버깅할 때 가장 먼저 볼 파일은?
- 루프/실행: `OM1/src/runtime/single_mode/cortex.py`
- 프롬프트 합성: `OM1/src/fuser/__init__.py`
- LLM 호출: `OM1/src/llm/plugins/<사용중인플러그인>.py`
- 액션 실행: `OM1/src/actions/orchestrator.py`

---

## 6. (부록) 설정 스키마 확인(개발/검증용)

고객사가 “설정에서 선택 가능한 입력/LLM/액션이 무엇인지” 빠르게 확인하고 싶을 때:

- 스키마 생성 스크립트: `OM1/scripts/generate_schema.py`
  - 실행 후 `OM1/OM1_config_schema.json5`가 생성됩니다(설치된 플러그인 기준).

