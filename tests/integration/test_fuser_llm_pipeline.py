#!/usr/bin/env python3
"""
Fuser + LLM 통합 파이프라인 테스트 스크립트.

RTX 4090 서버에서 EXAONE 추론 서버가 실행 중일 때,
Jetson AGX Orin (또는 로컬)에서 전체 파이프라인을 테스트합니다.

Usage:
    # Config 파일 기반 테스트 (권장)
    python tests/integration/test_fuser_llm_pipeline.py --config exaone_vllm
    python tests/integration/test_fuser_llm_pipeline.py --config exaone_ollama

    # 직접 지정 테스트
    python tests/integration/test_fuser_llm_pipeline.py --backend vllm --host 192.168.1.100
    python tests/integration/test_fuser_llm_pipeline.py --backend ollama --host 192.168.1.100
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import json5

# Mock problematic modules BEFORE importing anything else
sys.modules['zenoh'] = MagicMock()
sys.modules['providers'] = MagicMock()
sys.modules['providers.io_provider'] = MagicMock()
sys.modules['providers.io_provider'].IOProvider = MagicMock
sys.modules['providers.context_provider'] = MagicMock()
sys.modules['providers.examples'] = MagicMock()
sys.modules['providers.examples.io_provider'] = MagicMock()
sys.modules['providers.examples.io_provider'].IOProvider = MagicMock
sys.modules['providers.examples.avatar_provider'] = MagicMock()
sys.modules['providers.examples.avatar_llm_state_provider'] = MagicMock()
sys.modules['providers.examples.llm_history_manager'] = MagicMock()

# Add src to path
sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class MockSensorInput:
    """센서 입력을 시뮬레이션하는 Mock 클래스."""
    input_type: str
    content: str

    def formatted_latest_buffer(self) -> str:
        return f"INPUT: {self.input_type}. {self.content}"


@dataclass
class MockAgentAction:
    """에이전트 액션을 시뮬레이션하는 Mock 클래스."""
    name: str
    llm_label: str
    exclude_from_prompt: bool = False


@dataclass
class MockRuntimeConfig:
    """런타임 설정을 시뮬레이션하는 Mock 클래스."""
    system_prompt_base: str
    system_governance: str
    system_prompt_examples: str
    agent_actions: List[MockAgentAction]


def load_config_file(config_name: str) -> Dict[str, Any]:
    """Config 파일 로드."""
    config_dir = os.path.join(os.path.dirname(__file__), "..", "..", "config")
    config_path = os.path.join(config_dir, f"{config_name}.json5")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json5.load(f)


def create_mock_config(config_data: Optional[Dict[str, Any]] = None) -> MockRuntimeConfig:
    """테스트용 런타임 설정 생성. config_data가 제공되면 해당 값 사용."""
    if config_data:
        # Config 파일에서 값 추출
        agent_actions = []
        for action in config_data.get("agent_actions", []):
            agent_actions.append(MockAgentAction(
                name=action.get("name", ""),
                llm_label=action.get("llm_label", ""),
                exclude_from_prompt=action.get("exclude_from_prompt", False),
            ))

        return MockRuntimeConfig(
            system_prompt_base=config_data.get("system_prompt_base", ""),
            system_governance=config_data.get("system_governance", ""),
            system_prompt_examples=config_data.get("system_prompt_examples", ""),
            agent_actions=agent_actions,
        )

    # 기본 Mock 설정
    return MockRuntimeConfig(
        system_prompt_base="""You are a helpful companion robot named IRIS.
You assist pedestrians with navigation, information, and friendly conversation.
When you hear something, react naturally with helpful responses.
You respond with one sequence of commands at a time.""",
        system_governance="""Here are the laws that govern your actions:
First Law: A robot cannot harm a human or allow a human to come to harm.
Second Law: A robot must obey orders from humans, unless those orders conflict with the First Law.
Third Law: A robot must protect itself, as long as that protection doesn't conflict with the First or Second Law.""",
        system_prompt_examples="""Here are examples of interactions:

1. If a person asks for directions:
    Speak: {{'text': 'I can help you find your way. Where would you like to go?'}}

2. If a person greets you:
    Speak: {{'text': 'Hello! How can I assist you today?'}}
    Move: 'wave'""",
        agent_actions=[
            MockAgentAction(name="speak", llm_label="speak"),
            MockAgentAction(name="move", llm_label="move"),
            MockAgentAction(name="face", llm_label="emotion"),
        ],
    )


def create_fuser_prompt(config: MockRuntimeConfig, inputs: List[MockSensorInput]) -> str:
    """Fuser 로직을 시뮬레이션하여 프롬프트 생성."""
    # System prompt
    system_prompt = "\nBASIC CONTEXT:\n" + config.system_prompt_base + "\n"
    system_prompt += "\nLAWS:\n" + config.system_governance
    system_prompt += "\n\nEXAMPLES:\n" + config.system_prompt_examples

    # Inputs
    input_strings = [inp.formatted_latest_buffer() for inp in inputs]
    inputs_fused = " ".join(input_strings)

    # Actions
    actions_fused = ""
    for action in config.agent_actions:
        if not action.exclude_from_prompt:
            actions_fused += f"- {action.llm_label}: Execute {action.name} action\n"

    question_prompt = "What will you do? Actions:"

    fused_prompt = f"{system_prompt}\n\nAVAILABLE INPUTS:\n{inputs_fused}\nAVAILABLE ACTIONS:\n\n{actions_fused}\n\n{question_prompt}"

    return fused_prompt


def get_function_schemas() -> List[dict]:
    """테스트용 함수 스키마 생성."""
    return [
        {
            "type": "function",
            "function": {
                "name": "speak",
                "description": "Speak a sentence to the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to speak aloud",
                        }
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move the robot in a direction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["forward", "backward", "left", "right", "stop", "wave"],
                            "description": "The movement action to perform",
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "emotion",
                "description": "Display an emotion on the robot's face",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["happy", "sad", "surprised", "thinking", "neutral"],
                            "description": "The emotion to display",
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    ]


async def run_pipeline_test(
    backend: str,
    host: str,
    port: int,
    inputs: List[MockSensorInput],
    config_data: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
) -> dict:
    """전체 파이프라인 테스트 실행 (직접 HTTP 호출)."""
    import httpx
    import openai

    results = {
        "success": False,
        "fuser_time": 0,
        "llm_time": 0,
        "total_time": 0,
        "actions": [],
        "error": None,
    }

    try:
        # Step 1: Create config and fused prompt
        start_time = time.time()
        config = create_mock_config(config_data)
        fused_prompt = create_fuser_prompt(config, inputs)
        fuser_time = time.time() - start_time
        results["fuser_time"] = fuser_time

        logging.info(f"Fuser completed in {fuser_time:.4f}s")
        logging.debug(f"Fused prompt:\n{fused_prompt[:500]}...")

        # Step 2: Send to LLM
        logging.info("Sending fused prompt to LLM...")
        llm_start = time.time()

        function_schemas = get_function_schemas()

        # 모델 이름 결정
        if model:
            model_name = model
        elif backend == "ollama":
            model_name = "exaone3.5:7.8b"
        else:
            model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

        if backend == "ollama":
            base_url = f"http://{host}:{port}"

            # Build JSON response instruction (no native tool calling for Ollama)
            json_instruction = """

Respond ONLY with a JSON array of actions. No other text.
Available actions:
  - speak: Speak a sentence to the user (params: {"text": string})
  - move: Move the robot in a direction (params: {"action": one of ['forward', 'backward', 'left', 'right', 'stop', 'wave']})
  - emotion: Display an emotion on the robot's face (params: {"action": one of ['happy', 'sad', 'surprised', 'thinking', 'neutral']})

Response format (JSON array only):
[{"action": "action_name", "params": {"param_name": "value"}}]

Example:
[{"action": "speak", "params": {"text": "Hello!"}}, {"action": "move", "params": {"action": "forward"}}]
"""
            full_prompt = fused_prompt + json_instruction

            async with httpx.AsyncClient(timeout=120) as client:
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": full_prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_ctx": 4096},
                }

                response = await client.post(f"{base_url}/api/chat", json=payload)
                llm_time = time.time() - llm_start
                results["llm_time"] = llm_time

                if response.status_code == 200:
                    result = response.json()
                    message = result.get("message", {})
                    content = message.get("content", "")

                    logging.info(f"Ollama response: {content[:300]}...")

                    # Parse JSON from response
                    if content:
                        try:
                            # Find JSON array in response
                            start_idx = content.find("[")
                            end_idx = content.rfind("]")
                            if start_idx != -1 and end_idx != -1:
                                json_str = content[start_idx:end_idx + 1]
                                actions_data = json.loads(json_str)
                                if not isinstance(actions_data, list):
                                    actions_data = [actions_data]

                                for action in actions_data:
                                    action_name = action.get("action", "")
                                    params = action.get("params", {})
                                    value = params.get("text") or params.get("action") or str(params)
                                    results["actions"].append({
                                        "type": action_name,
                                        "value": value,
                                    })
                                results["success"] = len(results["actions"]) > 0
                            else:
                                logging.warning("No JSON array found in response")
                                results["success"] = False
                        except json.JSONDecodeError as e:
                            logging.warning(f"Failed to parse JSON: {e}")
                            results["success"] = False
                    else:
                        results["success"] = False
                else:
                    results["error"] = f"HTTP {response.status_code}"
        else:
            base_url = f"http://{host}:{port}/v1"
            client = openai.AsyncClient(base_url=base_url, api_key="placeholder")

            # tool_choice="required" 로 하면 반드시 tool을 사용하도록 강제
            # tool_choice="auto" 로 하면 LLM이 선택
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": fused_prompt}],
                tools=function_schemas,
                tool_choice="required",  # "auto" -> "required"로 변경
                temperature=0.7,
                max_tokens=2048,
            )
            llm_time = time.time() - llm_start
            results["llm_time"] = llm_time

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if tool_calls:
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                        value = args.get("text") or args.get("action") or str(args)
                    except (json.JSONDecodeError, ValueError):
                        value = tc.function.arguments
                    results["actions"].append({
                        "type": tc.function.name,
                        "value": value,
                    })
                results["success"] = True
            else:
                content = message.content or ""
                logging.warning(f"No tool calls. LLM Response:\n{content[:500]}")
                results["success"] = False
                results["raw_response"] = content

        logging.info(f"LLM responded in {llm_time:.2f}s")
        results["total_time"] = time.time() - start_time

    except Exception as e:
        results["error"] = str(e)
        logging.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    return results


async def main():
    parser = argparse.ArgumentParser(description="Test Fuser + LLM pipeline")
    parser.add_argument(
        "--config",
        default=None,
        help="Config file name (e.g., exaone_vllm, exaone_ollama)",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm"],
        default=None,
        help="LLM backend type (auto-detected from config if --config is used)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Server host address (auto-detected from config if --config is used)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (auto-detected from config if --config is used)",
    )

    args = parser.parse_args()

    # Config 파일 로드
    config_data = None
    model = None

    if args.config:
        try:
            config_data = load_config_file(args.config)
            logging.info(f"Loaded config: {args.config}")

            # cortex_llm에서 설정 추출
            cortex_llm = config_data.get("cortex_llm", {})
            llm_config = cortex_llm.get("config", {})
            llm_type = cortex_llm.get("type", "")

            # Backend 자동 감지
            if args.backend is None:
                if "Ollama" in llm_type:
                    args.backend = "ollama"
                elif "Vllm" in llm_type:
                    args.backend = "vllm"
                else:
                    args.backend = "ollama"  # fallback

            # Host/Port 추출
            base_url = llm_config.get("base_url", "")
            if base_url:
                # URL 파싱: http://host:port/... -> host, port
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                if args.host is None and parsed.hostname:
                    args.host = parsed.hostname
                if args.port is None and parsed.port:
                    args.port = parsed.port

            # Model 추출
            model = llm_config.get("model")

        except FileNotFoundError as e:
            logging.error(str(e))
            return 1

    # Set defaults
    if args.backend is None:
        args.backend = "ollama"
    if args.host is None:
        args.host = "localhost"
    if args.port is None:
        args.port = 11434 if args.backend == "ollama" else 8000

    logging.info("=" * 70)
    logging.info("Fuser + LLM Pipeline Integration Test")
    logging.info("=" * 70)
    if args.config:
        logging.info(f"Config: {args.config}")
    logging.info(f"Backend: {args.backend}")
    logging.info(f"Server: {args.host}:{args.port}")
    if model:
        logging.info(f"Model: {model}")
    logging.info("=" * 70)

    # Test scenarios
    test_cases = [
        {
            "name": "Greeting",
            "inputs": [
                MockSensorInput("Voice", "안녕하세요! 저는 관광객입니다."),
            ],
        },
        {
            "name": "Direction Request",
            "inputs": [
                MockSensorInput("Voice", "가까운 지하철역이 어디인가요?"),
                MockSensorInput("GPS", "현재 위치: 서울시 강남구"),
            ],
        },
        {
            "name": "Multiple Inputs",
            "inputs": [
                MockSensorInput("Voice", "저기요, 잠시만요!"),
                MockSensorInput("Vision", "Person detected at 3 meters, waving hand"),
                MockSensorInput("Proximity", "Obstacle detected on left side"),
            ],
        },
    ]

    all_results = []

    for i, test_case in enumerate(test_cases, 1):
        logging.info(f"\n[Test {i}/{len(test_cases)}] {test_case['name']}")
        logging.info("-" * 50)

        for inp in test_case["inputs"]:
            logging.info(f"  Input: {inp.input_type} - {inp.content}")

        results = await run_pipeline_test(
            args.backend,
            args.host,
            args.port,
            test_case["inputs"],
            config_data=config_data,
            model=model,
        )

        all_results.append({
            "name": test_case["name"],
            **results,
        })

        if results["success"]:
            logging.info(f"  Status: SUCCESS")
            logging.info(f"  Actions received:")
            for action in results["actions"]:
                logging.info(f"    - {action['type']}: {action['value']}")
        else:
            logging.warning(f"  Status: NO TOOL CALLS")
            if results["error"]:
                logging.error(f"  Error: {results['error']}")

        logging.info(f"  Timing: Fuser={results['fuser_time']*1000:.1f}ms, LLM={results['llm_time']*1000:.1f}ms, Total={results['total_time']*1000:.1f}ms")

    # Summary
    logging.info("\n" + "=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)

    success_count = sum(1 for r in all_results if r["success"])
    logging.info(f"Tests passed: {success_count}/{len(all_results)}")

    avg_llm_time = sum(r["llm_time"] for r in all_results) / len(all_results)
    logging.info(f"Average LLM response time: {avg_llm_time:.2f}s")

    if success_count == len(all_results):
        logging.info("\nAll tests passed! Pipeline is ready for deployment.")
    else:
        logging.warning(f"\n{len(all_results) - success_count} test(s) did not return tool calls.")
        logging.info("This may be expected if the model doesn't support function calling well.")

    logging.info("=" * 70)

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
