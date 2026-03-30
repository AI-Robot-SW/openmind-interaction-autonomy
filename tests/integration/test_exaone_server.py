#!/usr/bin/env python3
"""
EXAONE LLM 서버 연결 및 추론 테스트 스크립트.

RTX 4090 서버에서 EXAONE 추론 서버가 실행 중일 때,
Jetson AGX Orin (또는 로컬)에서 이 스크립트를 실행하여 연결을 테스트합니다.

Usage:
    # Ollama 테스트
    python tests/integration/test_exaone_server.py --backend ollama --host 192.168.1.100

    # vLLM 테스트
    python tests/integration/test_exaone_server.py --backend vllm --host 192.168.1.100 --port 8000
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from unittest.mock import MagicMock

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


async def test_ollama_connection(host: str, port: int, model: str) -> bool:
    """Ollama 서버 연결 테스트."""
    import httpx

    base_url = f"http://{host}:{port}"
    logging.info(f"Testing Ollama connection: {base_url}")

    async with httpx.AsyncClient(timeout=30) as client:
        # 1. 서버 상태 확인
        try:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                logging.info(f"Available models: {model_names}")

                if model not in model_names and not any(model in m for m in model_names):
                    logging.warning(f"Model '{model}' not found. Available: {model_names}")
            else:
                logging.error(f"Failed to get models: {response.status_code}")
                return False
        except httpx.ConnectError as e:
            logging.error(f"Cannot connect to Ollama: {e}")
            return False

        # 2. 간단한 추론 테스트
        logging.info("Testing inference...")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello, please respond with 'OK'."}],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 2048},
        }

        start_time = time.time()
        try:
            response = await client.post(f"{base_url}/api/chat", json=payload, timeout=120)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                logging.info(f"Response: {content[:100]}...")
                logging.info(f"Inference time: {elapsed:.2f}s")
                return True
            else:
                logging.error(f"Inference failed: {response.status_code} - {response.text}")
                return False
        except httpx.TimeoutException:
            logging.error("Inference timed out")
            return False

    return True


async def test_vllm_connection(host: str, port: int, model: str) -> bool:
    """vLLM 서버 연결 테스트."""
    import openai

    base_url = f"http://{host}:{port}/v1"
    logging.info(f"Testing vLLM connection: {base_url}")

    client = openai.AsyncClient(base_url=base_url, api_key="placeholder")

    # 1. 모델 목록 확인
    try:
        models = await client.models.list()
        model_ids = [m.id for m in models.data]
        logging.info(f"Available models: {model_ids}")
    except openai.APIConnectionError as e:
        logging.error(f"Cannot connect to vLLM: {e}")
        return False

    # 2. 간단한 추론 테스트
    logging.info("Testing inference...")
    start_time = time.time()

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello, please respond with 'OK'."}],
            temperature=0.1,
            max_tokens=50,
        )
        elapsed = time.time() - start_time

        content = response.choices[0].message.content
        logging.info(f"Response: {content[:100]}...")
        logging.info(f"Inference time: {elapsed:.2f}s")
        return True

    except openai.APITimeoutError:
        logging.error("Inference timed out")
        return False
    except Exception as e:
        logging.error(f"Inference error: {e}")
        return False


async def test_exaone_llm_plugin(backend: str, host: str, port: int) -> bool:
    """ExaoneLLM 플러그인 직접 테스트 (의존성 없이 직접 HTTP 호출)."""
    import httpx
    import openai

    logging.info(f"Testing ExaoneLLM-style request ({backend})...")

    test_prompt = "안녕하세요. 너는 누구니?"
    logging.info(f"Sending test prompt: {test_prompt}")

    start_time = time.time()

    if backend == "ollama":
        base_url = f"http://{host}:{port}"

        async with httpx.AsyncClient(timeout=120) as client:
            payload = {
                "model": "exaone3.5:7.8b",
                "messages": [{"role": "user", "content": test_prompt}],
                "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 4096},
            }

            response = await client.post(f"{base_url}/api/chat", json=payload)

            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                logging.info(f"Response: {content[:200]}...")
            else:
                logging.error(f"Error: {response.status_code}")
                return False
    else:
        base_url = f"http://{host}:{port}/v1"
        client = openai.AsyncClient(base_url=base_url, api_key="placeholder")

        response = await client.chat.completions.create(
            model="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.7,
            max_tokens=256,
        )

        content = response.choices[0].message.content
        logging.info(f"Response: {content[:200]}...")

    elapsed = time.time() - start_time
    logging.info(f"Total time: {elapsed:.2f}s")

    return True


async def test_with_tool_calling(backend: str, host: str, port: int) -> bool:
    """Tool calling 기능 테스트 (직접 HTTP 호출)."""
    import httpx
    import openai

    logging.info("Testing tool calling functionality...")

    # Define test function schemas
    function_schemas = [
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
                            "description": "The text to speak",
                        }
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": "Move the robot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["forward", "backward", "left", "right", "stop"],
                            "description": "The movement action",
                        }
                    },
                    "required": ["action"],
                },
            },
        },
    ]

    # Test prompt with tool calling context
    test_prompt = """You are a helpful robot assistant.

AVAILABLE ACTIONS:
- speak: Say something to the user
- move: Move in a direction (forward, backward, left, right, stop)

USER INPUT: "안녕하세요! 앞으로 가주세요."

What will you do? Use the available tools to respond."""

    logging.info("Sending prompt with tool calling context...")
    start_time = time.time()

    if backend == "ollama":
        base_url = f"http://{host}:{port}"

        # Convert to Ollama format
        ollama_tools = []
        for schema in function_schemas:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": schema["function"]["name"],
                    "description": schema["function"].get("description", ""),
                    "parameters": schema["function"].get("parameters", {}),
                },
            })

        async with httpx.AsyncClient(timeout=120) as client:
            payload = {
                "model": "exaone3.5:7.8b",
                "messages": [{"role": "user", "content": test_prompt}],
                "stream": False,
                "tools": ollama_tools,
                "options": {"temperature": 0.7, "num_ctx": 4096},
            }

            response = await client.post(f"{base_url}/api/chat", json=payload)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                tool_calls = message.get("tool_calls", [])

                if tool_calls:
                    logging.info(f"Success! Received {len(tool_calls)} tool call(s):")
                    for i, tc in enumerate(tool_calls):
                        func = tc.get("function", {})
                        logging.info(f"  [{i+1}] {func.get('name')}: {func.get('arguments')}")
                    logging.info(f"Total time: {elapsed:.2f}s")
                    return True
                else:
                    content = message.get("content", "")
                    logging.warning(f"No tool calls. Response: {content[:200]}...")
                    logging.info(f"Total time: {elapsed:.2f}s")
                    return False
            else:
                logging.error(f"Error: {response.status_code} - {response.text}")
                return False
    else:
        base_url = f"http://{host}:{port}/v1"
        client = openai.AsyncClient(base_url=base_url, api_key="placeholder")

        try:
            response = await client.chat.completions.create(
                model="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                messages=[{"role": "user", "content": test_prompt}],
                tools=function_schemas,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=512,
            )
            elapsed = time.time() - start_time

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if tool_calls:
                logging.info(f"Success! Received {len(tool_calls)} tool call(s):")
                for i, tc in enumerate(tool_calls):
                    logging.info(f"  [{i+1}] {tc.function.name}: {tc.function.arguments}")
                logging.info(f"Total time: {elapsed:.2f}s")
                return True
            else:
                logging.warning(f"No tool calls. Response: {message.content[:200] if message.content else 'empty'}...")
                logging.info(f"Total time: {elapsed:.2f}s")
                return False

        except Exception as e:
            logging.error(f"Error: {e}")
            return False


async def main():
    parser = argparse.ArgumentParser(description="Test EXAONE LLM server connection")
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm"],
        default="ollama",
        help="LLM backend type",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 11434 for ollama, 8000 for vllm)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: exaone3.5:7.8b for ollama, LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct for vllm)",
    )
    parser.add_argument(
        "--test-tools",
        action="store_true",
        help="Test tool calling functionality",
    )

    args = parser.parse_args()

    # Set defaults
    if args.port is None:
        args.port = 11434 if args.backend == "ollama" else 8000

    if args.model is None:
        args.model = (
            "exaone3.5:7.8b"
            if args.backend == "ollama"
            else "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
        )

    logging.info("=" * 60)
    logging.info("EXAONE LLM Server Connection Test")
    logging.info("=" * 60)
    logging.info(f"Backend: {args.backend}")
    logging.info(f"Host: {args.host}")
    logging.info(f"Port: {args.port}")
    logging.info(f"Model: {args.model}")
    logging.info("=" * 60)

    # Test 1: Basic connection
    logging.info("\n[Test 1] Basic Server Connection")
    logging.info("-" * 40)

    if args.backend == "ollama":
        success = await test_ollama_connection(args.host, args.port, args.model)
    else:
        success = await test_vllm_connection(args.host, args.port, args.model)

    if not success:
        logging.error("Basic connection test failed!")
        return 1

    logging.info("Basic connection test passed!")

    # Test 2: Plugin test
    logging.info("\n[Test 2] ExaoneLLM Plugin Test")
    logging.info("-" * 40)

    try:
        success = await test_exaone_llm_plugin(args.backend, args.host, args.port)
        if success:
            logging.info("Plugin test passed!")
    except Exception as e:
        logging.error(f"Plugin test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Tool calling (optional)
    if args.test_tools:
        logging.info("\n[Test 3] Tool Calling Test")
        logging.info("-" * 40)

        try:
            success = await test_with_tool_calling(args.backend, args.host, args.port)
            if success:
                logging.info("Tool calling test passed!")
            else:
                logging.warning("Tool calling test: No tool calls received (model may not support it)")
        except Exception as e:
            logging.error(f"Tool calling test failed: {e}")
            import traceback
            traceback.print_exc()

    logging.info("\n" + "=" * 60)
    logging.info("All tests completed!")
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
