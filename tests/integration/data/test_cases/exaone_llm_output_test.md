# EXAONE LLM Output 동작성 테스트 시나리오

## 1. 테스트 개요

| 항목 | 내용 |
|------|------|
| **목적** | 현재 구현된 LLM(EXAONE 3.5)의 Output이 시나리오별로 올바른 Action(move, speak)을 생성하는지 검증 |
| **대상 Config** | `config/pedestrian_robot_modes.json5` (멀티모드), `config/exaone_ollama.json5` / `config/exaone_vllm.json5` (싱글모드) |
| **LLM** | `ExaoneOllamaLLM` (exaone3.5:7.8b) / `ExaoneVllmLLM` (EXAONE-3.5-7.8B-Instruct-AWQ) |
| **대상 Action** | `move` (MovementAction enum), `speak` (자유 텍스트) |

## 2. 사용 Config / Input / Action 정리

### 2.1 Config

```json5
{
  "cortex_llm": {
    "type": "ExaoneOllamaLLM",       // 또는 ExaoneVllmLLM
    "config": {
      "model": "exaone3.5:7.8b",
      "temperature": 0.7,
      "num_ctx": 4096
    }
  }
}
```

### 2.2 Input (LLM에 주입되는 센서 컨텍스트)

| Input Sensor | descriptor_for_LLM | 테스트 시 Mock 데이터 |
|--------------|--------------------|-----------------------|
| **SoundSensor** (Voice) | `"Voice"` | 사용자 음성 텍스트 (예: "L1으로 가줘") |
| **LocationSensor** (Location) | `"Location"` | 현재 위치, 네비게이션 상태, 목표 거리 등 |

> VisionSensor는 이번 테스트 범위에서 제외

### 2.3 Action (LLM Output으로 기대하는 동작)

#### 모드별 사용 가능 Action

| 모드 | Action | LLM Label | 가능한 값 |
|------|--------|-----------|-----------|
| **Standby** | speak | `"speak"` | 자유 텍스트 (한국어/영어) |
| **Conversation** | speak | `"speak"` | 자유 텍스트 (한국어/영어) |
| **Accompany** | move | `"move"` | `"go to L1"`, `"go to L2"`, `"go to L3"`, `"slow down"`, `"speed up"`, `"stop move"`, `"stand up"`, `"stand down"`, `"damp"` |
| **Accompany** | speak | `"speak"` | 자유 텍스트 (한국어/영어) |
| **Caution** | move_caution | `"move"` | `"slow down"`, `"stop move"` (제한됨) |
| **Caution** | speak | `"speak"` | 자유 텍스트 (한국어/영어) |

> Standby/Conversation 모드에서는 move 액션이 없으므로 LLM이 이동 명령을 생성하지 않음

## 3. 테스트 시나리오

### TC-01: 목적지 설정

| 항목 | 내용 |
|------|------|
| **Test Name** | 목적지 설정 - L1 |
| **Mode** | **Accompany** |
| **PreCondition** | 정지 상태 (navigation_status: "idle") |
| **Input (Voice)** | "L1으로 가줘" |
| **Input (Location)** | `Status: idle` |
| **Expected Output** | move: `"go to L1"`, speak: 목적지 이동 안내 문구 (예: "L1으로 이동하겠습니다") |
| **Real Output** | |
| **Result** | |

---

| 항목 | 내용 |
|------|------|
| **Test Name** | 목적지 설정 - L2 |
| **Mode** | **Accompany** |
| **PreCondition** | 정지 상태 (navigation_status: "idle") |
| **Input (Voice)** | "L2로 데려다줘" |
| **Input (Location)** | `Status: idle` |
| **Expected Output** | move: `"go to L2"`, speak: 목적지 이동 안내 문구 |
| **Real Output** | |
| **Result** | |

---

| 항목 | 내용 |
|------|------|
| **Test Name** | 목적지 설정 - L3 |
| **Mode** | **Accompany** |
| **PreCondition** | 정지 상태 (navigation_status: "idle") |
| **Input (Voice)** | "L3로 가자" |
| **Input (Location)** | `Status: idle` |
| **Expected Output** | move: `"go to L3"`, speak: 목적지 이동 안내 문구 |
| **Real Output** | |
| **Result** | |

### TC-02: 목적지 이동 (주행 시작 확인)

| 항목 | 내용 |
|------|------|
| **Test Name** | 목적지 이동 시작 |
| **Mode** | **Accompany** |
| **PreCondition** | 사용자가 "L1으로 가줘"라고 발화한 직후 |
| **Input (Voice)** | (없음 - 이전 턴에서 이미 발화) |
| **Input (Location)** | `Status: navigating, Distance to goal: 50.0m` |
| **Expected Output** | move: `"go to L1"` (이동 명령 유지), speak: 이동 중 안내 (예: "L1으로 이동 중입니다") |
| **Real Output** | |
| **Result** | |

### TC-03: 목적지 도착

| 항목 | 내용 |
|------|------|
| **Test Name** | 목적지 도착 안내 |
| **Mode** | **Accompany** |
| **PreCondition** | L1으로 주행 중 |
| **Input (Voice)** | (없음) |
| **Input (Location)** | `Status: arrived, Location: L1` |
| **Expected Output** | move: `"stop move"`, speak: 도착 안내 (예: "L1에 도착했습니다") |
| **Real Output** | |
| **Result** | |

### TC-04: 목적지 주행 중 감속

| 항목 | 내용 |
|------|------|
| **Test Name** | 주행 중 감속 요청 |
| **Mode** | **Accompany** |
| **PreCondition** | L1으로 주행 중 (navigation_status: "navigating") |
| **Input (Voice)** | "좀 천천히 가줘" |
| **Input (Location)** | `Status: navigating, Distance to goal: 30.0m` |
| **Expected Output** | move: `"slow down"`, speak: 감속 확인 (예: "속도를 줄이겠습니다") |
| **Real Output** | |
| **Result** | |

---

| 항목 | 내용 |
|------|------|
| **Test Name** | 주행 중 감속 요청 (다른 표현) |
| **Mode** | **Accompany** |
| **PreCondition** | L2로 주행 중 |
| **Input (Voice)** | "느리게 가" |
| **Input (Location)** | `Status: navigating, Distance to goal: 20.0m` |
| **Expected Output** | move: `"slow down"`, speak: 감속 확인 문구 |
| **Real Output** | |
| **Result** | |

### TC-05: 목적지 주행 중 가속

| 항목 | 내용 |
|------|------|
| **Test Name** | 주행 중 가속 요청 |
| **Mode** | **Accompany** |
| **PreCondition** | L1으로 주행 중 (navigation_status: "navigating") |
| **Input (Voice)** | "좀 빨리 가줘" |
| **Input (Location)** | `Status: navigating, Distance to goal: 30.0m` |
| **Expected Output** | move: `"speed up"`, speak: 가속 확인 (예: "속도를 높이겠습니다") |
| **Real Output** | |
| **Result** | |

---

| 항목 | 내용 |
|------|------|
| **Test Name** | 주행 중 가속 요청 (다른 표현) |
| **Mode** | **Accompany** |
| **PreCondition** | L3로 주행 중 |
| **Input (Voice)** | "빨리빨리!" |
| **Input (Location)** | `Status: navigating, Distance to goal: 40.0m` |
| **Expected Output** | move: `"speed up"`, speak: 가속 확인 문구 |
| **Real Output** | |
| **Result** | |

### TC-06: 목적지 주행 중 정지

| 항목 | 내용 |
|------|------|
| **Test Name** | 주행 중 정지 요청 |
| **Mode** | **Accompany** |
| **PreCondition** | L1으로 주행 중 (navigation_status: "navigating") |
| **Input (Voice)** | "멈춰" |
| **Input (Location)** | `Status: navigating, Distance to goal: 15.0m` |
| **Expected Output** | move: `"stop move"`, speak: 정지 확인 (예: "멈추겠습니다") |
| **Real Output** | |
| **Result** | |

---

| 항목 | 내용 |
|------|------|
| **Test Name** | 주행 중 정지 요청 (다른 표현) |
| **Mode** | **Accompany** |
| **PreCondition** | L2로 주행 중 |
| **Input (Voice)** | "잠깐 서봐" |
| **Input (Location)** | `Status: navigating, Distance to goal: 10.0m` |
| **Expected Output** | move: `"stop move"`, speak: 정지 확인 문구 |
| **Real Output** | |
| **Result** | |

---

| 항목 | 내용 |
|------|------|
| **Test Name** | 주행 중 정지 - 용무 요청 |
| **Mode** | **Accompany** |
| **PreCondition** | L1으로 주행 중 |
| **Input (Voice)** | "나 화장실 좀" |
| **Input (Location)** | `Status: navigating, Distance to goal: 20.0m` |
| **Expected Output** | move: `"stop move"`, speak: 대기 안내 (예: "넵, 기다리겠습니다") |
| **Real Output** | |
| **Result** | |

### TC-07: 목적지 주행 중 남은거리 응답 (Optional)

| 항목 | 내용 |
|------|------|
| **Test Name** | 남은 거리 질문 |
| **Mode** | **Accompany** |
| **PreCondition** | L1으로 주행 중 (navigation_status: "navigating") |
| **Input (Voice)** | "얼마나 남았어?" |
| **Input (Location)** | `Status: navigating, Distance to goal: 25.0m` |
| **Expected Output** | speak: 남은 거리 안내 (예: "약 25미터 남았습니다") — move 명령 없음 (주행 유지) |
| **Real Output** | |
| **Result** | |

---

| 항목 | 내용 |
|------|------|
| **Test Name** | 남은 거리 질문 (다른 표현) |
| **Mode** | **Accompany** |
| **PreCondition** | L3으로 주행 중 |
| **Input (Voice)** | "아직 멀었어?" |
| **Input (Location)** | `Status: navigating, Distance to goal: 5.0m` |
| **Expected Output** | speak: 남은 거리/거의 도착 안내 (예: "거의 다 왔습니다, 5미터 남았어요") — move 명령 없음 |
| **Real Output** | |
| **Result** | |

## 4. 체크리스트

### 4.1 테스트 환경 체크리스트

- [ ] Ollama 서버 실행 확인 (`ollama serve`)
- [ ] EXAONE 모델 pull 확인 (`ollama pull exaone3.5:7.8b`)
- [ ] config 내 `base_url`의 `{server_ip}` 실제 IP로 치환
- [ ] LLM 연결 테스트 통과 (`tests/integration/test_exaone_server.py`)

### 4.2 Input Mock 데이터 준비 체크리스트

- [ ] Voice Input mock 준비 (각 시나리오별 사용자 발화 텍스트)
- [ ] Location Input mock 준비 (각 시나리오별 LocationData)
  - [ ] idle 상태 (TC-01)
  - [ ] navigating 상태 + distance_to_goal (TC-02, 04, 05, 06, 07)
  - [ ] arrived 상태 (TC-03)

### 4.3 LLM Output 검증 체크리스트

| # | 모드 | 시나리오 | move 값 검증 | speak 생성 여부 | Result |
|---|------|---------|-------------|----------------|--------|
| TC-01a | Accompany | 목적지 설정 L1 | `"go to L1"` | O | |
| TC-01b | Accompany | 목적지 설정 L2 | `"go to L2"` | O | |
| TC-01c | Accompany | 목적지 설정 L3 | `"go to L3"` | O | |
| TC-02 | Accompany | 목적지 이동 시작 | `"go to L1"` | O | |
| TC-03 | Accompany | 목적지 도착 | `"stop move"` | O | |
| TC-04a | Accompany | 주행 중 감속 | `"slow down"` | O | |
| TC-04b | Accompany | 주행 중 감속 (변형) | `"slow down"` | O | |
| TC-05a | Accompany | 주행 중 가속 | `"speed up"` | O | |
| TC-05b | Accompany | 주행 중 가속 (변형) | `"speed up"` | O | |
| TC-06a | Accompany | 주행 중 정지 | `"stop move"` | O | |
| TC-06b | Accompany | 주행 중 정지 (변형) | `"stop move"` | O | |
| TC-06c | Accompany | 주행 중 정지 (용무) | `"stop move"` | O | |
| TC-07a | Accompany | 남은거리 응답 | 없음 (주행 유지) | O (거리 언급) | |
| TC-07b | Accompany | 남은거리 응답 (변형) | 없음 (주행 유지) | O (거리 언급) | |

### 4.4 판정 기준

| 항목 | Pass 조건 |
|------|----------|
| **move** | Expected Output의 move 값과 Real Output의 move 값이 **정확히 일치** (MovementAction enum 값 기준) |
| **speak** | (1) speak action이 생성됨, (2) 문맥에 맞는 응답 (키워드 포함 여부로 판단) |
| **전체 Pass** | move 값 일치 **AND** speak 생성 확인 시 Pass |

