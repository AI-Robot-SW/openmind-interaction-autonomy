# text_input_client.py

"""
Interactive text client for TextSensor.

Connects to the TextSensor WebSocket server and lets you type messages
to the LLM while the system is running.

Usage:
    uv run python system_hw_test/text_input_client.py [--host 127.0.0.1] [--port 8765]
"""

import sys
import asyncio
import argparse
import websockets


async def chat(host: str, port: int) -> None:
    uri = f"ws://{host}:{port}"
    try:
        async with websockets.connect(uri) as ws:
            print(f"Connected to {uri}. Type a message and press Enter. (Ctrl+C to quit)\n")
            loop = asyncio.get_running_loop()
            while True:
                try:
                    text = await loop.run_in_executor(None, sys.stdin.readline)
                except (EOFError, KeyboardInterrupt):
                    break

                text = text.strip()
                if not text:
                    continue

                await ws.send(text)
                reply = await ws.recv()
                print(reply)

    except (ConnectionRefusedError, OSError) as e:
        print(f"Cannot connect to {uri} — {e}")
        print("Make sure the system is running with TextSensor in its agent_inputs.")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="TextSensor interactive client")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    try:
        asyncio.run(chat(args.host, args.port))
    except KeyboardInterrupt:
        print("\nBye.")


if __name__ == "__main__":
    main()
