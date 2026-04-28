<div align="center">

<p align="center">
  <img src="assets/banner.png" alt="Pedestrian Companion Robot Banner" style="max-width:80%; min-width:200px;" />
</p>

# Pedestrian Companion Robot

**보행자 동행 안내 로봇 — OpenMind(OM1) 런타임 기반**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble%2FIron-22314E?style=flat-square&logo=ros&logoColor=white)](https://ros.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

[Overview](#overview) • [Getting Started](#getting-started) • [Configuration](#configuration) • [Architecture](#architecture)

---

</div>

## Overview

보행자와 함께 이동하며 음성 대화로 목적지를 안내하는 자율주행 로봇 플랫폼. OpenMind OM1 런타임 위에서 동작하며 다음 스택을 사용한다.

- **로봇**: Unitree Go2 (이더넷 연결)
- **센싱**: Intel RealSense (RGB-D), RTK GNSS, Mic Array
- **인지/로컬 플래닝**: BEV occupancy grid + Distance map + DWA local planner, Segmentation 기반 사람/장애물 회피
- **경로**: GNSS 글로벌 경로 + DWA 로컬 회피
- **음성**: SileroVAD + Whisper-timestamped(STT) + Naver Clova(TTS)
- **LLM**: Exaone 3.5 (Ollama 또는 vLLM 백엔드)
- **사전 지정 목적지**: `L8`, 북문(`NG`)

## Getting Started

### Prerequisites

- Python >= 3.10
- [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Ubuntu 22.04 (권장), NVIDIA GPU + CUDA 11.0+
- 실기 운용 시: Unitree Go2, Intel RealSense, RTK GPS 모듈

### Installation

1. **Clone**
   ```bash
   git clone https://github.com/AI-Robot-SW/pedestrian-companion-robot.git
   cd pedestrian-companion-robot
   ```

2. **System dependencies (Linux)**
   ```bash
   sudo apt-get update
   sudo apt-get install -y portaudio19-dev python3-dev ffmpeg \
       libasound2-dev libv4l-dev build-essential cmake \
       python3-pip python3-colcon-common-extensions
   ```

3. **Python environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```
   CycloneDDS, Unitree Go2 Python SDK는 `pyproject.toml`을 통해 자동 설치된다.

4. **API key 설정**
   - [OpenMind Portal](https://portal.openmind.org/)에서 API key 발급
   - `.env` 파일에 추가: `OM_API_KEY=your_api_key`
   - 또는 `config/pedestrian_robot_accompany.json5`의 `api_key` 필드 직접 수정

5. **로봇 네트워크 인터페이스 설정**
   `config/pedestrian_robot_accompany.json5`의 `UnitreeGo2Bg` 항목에서 `unitree_ethernet`을 환경에 맞게 수정 (예: `eno1`, `enp2s0`).

### Run

기본 실행 (single-mode, 동행 안내):
```bash
uv run src/run.py pedestrian_robot_accompany
```

GUI variant:
```bash
uv run src/run.py pedestrian_robot_accompany_gui
```

Multi-mode (예약, standby ↔ accompany 모드 전환 — future use):
```bash
uv run src/run.py pedestrian_robot_modes
```

## Configuration

`config/`에 보존된 config 파일:

| 파일 | 모드 | 설명 |
|---|---|---|
| `pedestrian_robot_accompany.json5` | single-mode | **기본 운영 config**. 보행자와 동행하며 목적지 안내 |
| `pedestrian_robot_accompany_gui.json5` | single-mode | GUI 출력 포함 변형 |
| `pedestrian_robot_modes.json5` | multi-mode | standby/accompany 두 모드 + 키워드 기반 전환 (future use) |
| `schema/` | — | JSON5 config 검증용 JSON Schema |

### 주요 동작 규칙 (요약)

`pedestrian_robot_accompany.json5`의 system prompt 기반:

- **목적지 설정**: "북문으로 가줘" → `move('go to NG')` + speak. 사전 지정 외 목적지 거부.
- **속도/정지/재개**: "빨리 가자", "천천히", "멈춰", "잠깐 기다려", "다시 가자" 등 키워드를 LLM이 의도 분류 후 `move(...)` 호출
- **상태 인지**: Navigation provider의 `DWA / IDLE / CALIBRATING / STOP / ARRIVED` 상태를 시스템 프롬프트에 주입, LLM이 상태에 맞춰 응답
- **발화 게이트**: 첫 tick / 음성 입력 tick / ARRIVED 첫 tick / CALIBRATING 첫 tick에서만 발화, 그 외 tick은 빈 액션

## Architecture

```
src/
├── run.py                 # entry point (typer CLI)
├── runtime/
│   ├── single_mode/       # CortexRuntime (기본 운영)
│   └── multi_mode/        # ModeCortexRuntime (future use)
├── inputs/plugins/        # SoundSensor, LocationSensor, VisionSensor
├── actions/
│   ├── speak/             # speak_connector + Naver Clova/기타 TTS connector
│   └── move/              # move_connector
├── llm/plugins/           # ExaoneOllamaLLM, ExaoneVLLM, gemini, openai 등
├── providers/             # RealSense, BEV occupancy grid, DistMap, DWA route,
│                          #   GNSS/RTK, Navigation, STT/TTS, UnitreeGo2 등
├── backgrounds/           # 백그라운드 태스크 (오케스트레이터 관리)
├── fuser/                 # 입력 fusion
└── simulators/            # 시뮬레이션
```

자세한 시스템 흐름은 `tests/integration/data/test_cases/exaone_llm_output_test_diagrams.md` 참고.

## Requirements

| 항목 | 요구사항 |
|---|---|
| OS | Ubuntu 22.04 (권장) |
| Python | >= 3.10 |
| GPU | NVIDIA GPU + CUDA 11.0+ (BEV occupancy grid CUDA kernel용, 옵션) |
| 로봇 | Unitree Go2 |
| 센서 | Intel RealSense (RGB-D), RTK GNSS, USB Audio mic |

Python 의존성은 `pyproject.toml`로 관리 (`uv sync` 또는 `uv pip install -e .`).

## License

MIT License. 자세한 내용은 [LICENSE](LICENSE) 참고.

---

<div align="center">

Built on OpenMind OM1

</div>
