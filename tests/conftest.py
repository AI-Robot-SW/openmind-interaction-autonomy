"""
Global test configuration for pedestrian-companion-robot.

Adds --live flag to run tests that require a live LLM server (Ollama/vLLM).

Usage:
    pytest                           # Unit tests only (default)
    pytest -m integration            # Mock integration tests
    pytest -m live --live             # Live server tests (requires Ollama)
    pytest -m live --live --model exaone3.5:7.8b  # Specify model
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live tests that require an actual LLM server",
    )
    parser.addoption(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL for live tests",
    )
    parser.addoption(
        "--model",
        default="exaone3.5:7.8b",
        help="Model to use for live tests",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires a live LLM server")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="need --live option to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)


@pytest.fixture
def ollama_url(request):
    return request.config.getoption("--ollama-url")


@pytest.fixture
def live_model(request):
    return request.config.getoption("--model")
