# EXAONE LLM Output 테스트 - 아키텍처 다이어그램

## 1. 시퀀스 다이어그램 (전체 Tick 사이클)

사용자 음성 입력 → LLM 응답 → Action 실행까지의 한 사이클 흐름

```mermaid
sequenceDiagram
    autonumber
    participant User as 사용자 (음성)
    participant STT as STTProvider
    participant SS as SoundSensor<br/>(Voice Input)
    participant LS as LocationSensor<br/>(Location Input)
    participant IO as InputOrchestrator
    participant CR as CortexRuntime<br/>(_tick)
    participant F as Fuser
    participant LLM as ExaoneOllamaLLM<br/>(exaone3.5:7.8b)
    participant AO as ActionOrchestrator
    participant MC as MoveConnector
    participant SC as SpeakConnector

    Note over CR: _tick() 0.5Hz 주기 실행

    %% Phase 1: Input 수집
    rect rgb(230, 245, 255)
        Note over User, IO: Phase 1 — Input 수집
        User->>STT: 음성 발화 (예: "L1으로 가줘")
        STT->>SS: STT 결과 콜백 (text)
        SS->>SS: message_buffer에 저장
        Note over LS: LocationProvider로부터<br/>위치 데이터 수신 (mock)
        LS->>LS: latest_location 업데이트
        IO->>SS: _poll() → raw_to_text()
        IO->>LS: _poll() → raw_to_text()
    end

    %% Phase 2: Fuser 프롬프트 조합
    rect rgb(255, 245, 230)
        Note over CR, F: Phase 2 — Prompt 조합
        CR->>AO: flush_promises() — 이전 Action 결과 수집
        AO-->>CR: finished_promises
        CR->>F: fuse(agent_inputs, finished_promises)
        F->>SS: formatted_latest_buffer()
        SS-->>F: "INPUT: Voice\n// START\nL1으로 가줘\n// END"
        F->>LS: formatted_latest_buffer()
        LS-->>F: "INPUT: Location\n// START\nStatus: idle\n// END"
        F->>F: system_prompt_base + governance<br/>+ examples + inputs + actions 조합
        F-->>CR: fused_prompt (최종 프롬프트)
    end

    %% Phase 3: LLM 추론
    rect rgb(245, 255, 230)
        Note over CR, LLM: Phase 3 — LLM 추론
        CR->>LLM: ask(fused_prompt)
        LLM->>LLM: function_schemas 생성<br/>(move enum + speak text)
        LLM->>LLM: _build_json_response_instruction()<br/>→ JSON array 형식 지시 추가
        LLM->>LLM: Ollama API 호출<br/>(POST /api/chat)
        LLM->>LLM: _parse_json_response()<br/>→ Action 파싱
        LLM->>LLM: convert_function_calls_to_actions()
        LLM-->>CR: CortexOutputModel<br/>(actions: [Action(move, "go to L1"),<br/>Action(speak, "L1으로 이동하겠습니다")])
    end

    %% Phase 4: Action 실행
    rect rgb(255, 230, 240)
        Note over CR, SC: Phase 4 — Action 실행
        CR->>AO: promise(output.actions)
        AO->>MC: execute Action(type="move", value="go to L1")
        MC->>MC: MovementAction 매핑<br/>→ NavProvider (TODO)
        AO->>SC: execute Action(type="speak", value="L1으로 이동하겠습니다")
        SC->>SC: TTSProvider.add_pending_message()
    end
```

## 2. 시퀀스 다이어그램 (테스트 시나리오별 — 주행 중 정지 예시)

TC-06c: 주행 중 "나 화장실 좀" 발화 시 흐름

```mermaid
sequenceDiagram
    autonumber
    participant User as 사용자
    participant SS as SoundSensor
    participant LS as LocationSensor
    participant F as Fuser
    participant LLM as ExaoneLLM
    participant AO as ActionOrchestrator

    Note over LS: PreCondition:<br/>navigation_status="navigating"<br/>distance_to_goal=20.0m<br/>current_goal=L1

    User->>SS: "나 화장실 좀"

    F->>SS: formatted_latest_buffer()
    SS-->>F: INPUT: Voice — "나 화장실 좀"

    F->>LS: formatted_latest_buffer()
    LS-->>F: INPUT: Location — "Status: navigating,<br/>Distance to goal: 20.0m"

    F-->>LLM: 조합된 프롬프트

    LLM-->>AO: actions: [<br/>  Action(move, "stop move"),<br/>  Action(speak, "넵, 기다리겠습니다")<br/>]

    Note over AO: Expected:<br/>move = "stop move" ✓<br/>speak = 대기 안내 ✓

    AO->>AO: MoveConnector → stop move
    AO->>AO: SpeakConnector → TTS 출력
```

## 3. C4 Context 다이어그램 (시스템 전체)

```mermaid
C4Context
    title Pedestrian Companion Robot — System Context

    Person(user, "보행자", "음성으로 로봇에 명령/질문")

    System(iris, "Pedestrian Robot System", "EXAONE LLM 기반 보행자 동반 로봇")

    System_Ext(ollama, "Ollama Server", "EXAONE 3.5 7.8B 모델 서빙")
    System_Ext(stt, "STT Service", "음성→텍스트 변환")
    System_Ext(tts, "TTS Service", "텍스트→음성 변환")
    System_Ext(nav, "Navigation System", "자율주행/위치추정 (AMCL/Lidar/GPS)")

    Rel(user, iris, "음성 명령", "마이크")
    Rel(iris, user, "음성 응답", "스피커")
    Rel(iris, ollama, "LLM 추론 요청", "HTTP API")
    Rel(iris, stt, "음성 인식", "STTProvider")
    Rel(iris, tts, "음성 합성", "TTSProvider")
    Rel(iris, nav, "이동 명령 / 위치 수신", "NavProvider")
```

## 4. C4 Container 다이어그램 (내부 모듈)

```mermaid
C4Container
    title Pedestrian Companion Robot — Container Diagram

    Person(user, "보행자")

    System_Boundary(iris, "Pedestrian Robot System") {

        Container(cortex, "CortexRuntime", "Python asyncio", "0.5Hz tick 주기로 전체 파이프라인 실행")

        Container(input_orch, "InputOrchestrator", "Python", "센서 입력 수집 및 폴링")
        Container(sound, "SoundSensor", "Input Plugin", "descriptor: Voice<br/>STTProvider 콜백 수신")
        Container(location, "LocationSensor", "Input Plugin", "descriptor: Location<br/>위치/네비게이션 상태")

        Container(fuser, "Fuser", "Python", "system_prompt + inputs + actions → 프롬프트 조합")

        Container(llm, "ExaoneOllamaLLM", "LLM Plugin", "function schema 생성<br/>JSON 파싱 → Action 변환")

        Container(action_orch, "ActionOrchestrator", "Python", "Action → Connector 라우팅<br/>ThreadPool 기반 병렬 실행")
        Container(move_conn, "MoveConnector", "Action Connector", "MovementAction enum 매핑<br/>go to L1~L3, slow/speed/stop")
        Container(speak_conn, "SpeakConnector", "Action Connector", "TTSProvider로 텍스트 전달")
    }

    System_Ext(ollama, "Ollama Server", "exaone3.5:7.8b")
    System_Ext(nav, "Navigation", "NavProvider (TODO)")
    System_Ext(tts, "TTS Engine", "TTSProvider → SpeakerProvider")

    Rel(user, sound, "음성", "마이크 → STT")
    Rel(nav, location, "위치 데이터", "LocationProvider")

    Rel(input_orch, sound, "poll + raw_to_text")
    Rel(input_orch, location, "poll + raw_to_text")

    Rel(cortex, input_orch, "센서 데이터 수집")
    Rel(cortex, fuser, "fuse(inputs, promises)")
    Rel(cortex, llm, "ask(prompt)")
    Rel(cortex, action_orch, "promise(actions)")

    Rel(fuser, sound, "formatted_latest_buffer()")
    Rel(fuser, location, "formatted_latest_buffer()")

    Rel(llm, ollama, "POST /api/chat", "HTTP")

    Rel(action_orch, move_conn, "Action(move, value)")
    Rel(action_orch, speak_conn, "Action(speak, value)")

    Rel(move_conn, nav, "이동 명령")
    Rel(speak_conn, tts, "TTS 요청")
```

## 5. 데이터 흐름 다이어그램 (Fuser 프롬프트 구조)

```mermaid
flowchart TD
    subgraph PROMPT["최종 프롬프트 구조"]
        direction TB
        SP["(1) BASIC CONTEXT<br/>system_prompt_base<br/><i>'You are pedestrian_robot...'</i>"]
        GOV["(2) LAWS<br/>system_governance<br/><i>로봇 3법칙</i>"]
        EX["(3) EXAMPLES<br/>system_prompt_examples<br/><i>인사/길안내/이동 예시</i>"]
        INP["(4) AVAILABLE INPUTS<br/>센서 데이터"]
        ACT["(5) AVAILABLE ACTIONS<br/>function schema 설명"]
        Q["(6) What will you do? Actions:"]

        SP --> GOV --> EX --> INP --> ACT --> Q
    end

    subgraph INPUTS["센서 입력"]
        V["INPUT: Voice<br/>// START<br/>'L1으로 가줘'<br/>// END"]
        L["INPUT: Location<br/>// START<br/>Status: navigating<br/>Distance to goal: 20.0m<br/>// END"]
    end

    subgraph ACTIONS["Action 스키마"]
        M["move(action)<br/>enum: go to L1, go to L2,<br/>go to L3, slow down,<br/>speed up, stop move, ..."]
        S["speak(action)<br/>type: string<br/>(자유 텍스트)"]
    end

    V --> INP
    L --> INP
    M --> ACT
    S --> ACT

    subgraph OUTPUT["LLM Output"]
        O["CortexOutputModel<br/>actions: [<br/>  Action(move, 'go to L1'),<br/>  Action(speak, 'L1으로 이동합니다')<br/>]"]
    end

    Q --> LLM["ExaoneOllamaLLM"]
    LLM --> O
```

## 6. 모드 전환 상태 다이어그램

```mermaid
stateDiagram-v2
    [*] --> Standby: 시스템 시작 (default_mode)

    state "Standby (대기)" as Standby
    state "Conversation (대화)" as Conv
    state "Accompany (동행)" as Accom
    state "Caution (주의)" as Caut

    Standby --> Conv: 사용자 접근/인사<br/>"안녕", "여기요", "도와줘"
    Conv --> Standby: 작별/비활성 3분<br/>"잘가", "끝"
    Conv --> Accom: 목적지 요청<br/>"L1으로 가자", "이동"
    Accom --> Conv: 정지/도착<br/>"멈춰", "도착"
    Accom --> Caut: 위험 감지<br/>"위험", "조심", "장애물"
    Caut --> Accom: 위험 해제<br/>"안전", "계속"
    Caut --> Conv: 완전 정지<br/>"정지", "대화"

    note right of Standby: speak만 가능<br/>move 없음
    note right of Conv: speak만 가능<br/>move 없음
    note right of Accom: speak + move (전체 9개 enum)
    note right of Caut: speak + move_caution<br/>(slow down, stop move만)
```

## 7. 테스트 시나리오 상태 전이 다이어그램 (Accompany 모드 내)

```mermaid
stateDiagram-v2
    [*] --> Idle: Accompany 모드 진입

    state "Idle (정지)" as Idle
    state "Navigating (주행 중)" as Nav
    state "Arrived (도착)" as Arrived
    state "Stopped (일시정지)" as Stopped

    Idle --> Nav: TC-01 목적지 설정<br/>Voice: "L1으로 가줘"<br/>move: "go to L1"

    Nav --> Nav: TC-04 감속<br/>Voice: "천천히 가줘"<br/>move: "slow down"
    Nav --> Nav: TC-05 가속<br/>Voice: "빨리 가줘"<br/>move: "speed up"
    Nav --> Nav: TC-07 남은거리 응답<br/>Voice: "얼마나 남았어?"<br/>move: (없음, speak만)

    Nav --> Stopped: TC-06 정지<br/>Voice: "멈춰" / "화장실 좀"<br/>move: "stop move"
    Nav --> Arrived: TC-03 도착<br/>Location: arrived<br/>move: "stop move"

    Stopped --> Nav: 재출발 요청<br/>Voice: "다시 가자"<br/>move: "go to L1"
    Arrived --> Idle: 목적지 도착 완료

    Arrived --> [*]
```
