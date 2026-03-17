"""
Tests for EXAONE LLM implementations (Ollama and vLLM backends).
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# Mock problematic modules BEFORE importing anything else
sys.modules['zenoh'] = MagicMock()
sys.modules['providers.io_provider'] = MagicMock()
sys.modules['providers.io_provider'].IOProvider = MagicMock
sys.modules['providers.context_provider'] = MagicMock()
sys.modules['providers.examples.avatar_provider'] = MagicMock()


def _passthrough_decorator(*args, **kwargs):
    """Passthrough decorator that returns the function unchanged."""
    def decorator(func):
        return func
    if args and callable(args[0]):
        return args[0]
    return decorator


# Create proper mocks for AvatarLLMState and LLMHistoryManager
# so their decorators don't wrap async methods into MagicMocks
mock_avatar_llm_state = MagicMock()
mock_avatar_llm_state.trigger_thinking = _passthrough_decorator

mock_llm_history_manager = MagicMock()
mock_llm_history_manager.update_history = _passthrough_decorator

mock_avatar_module = MagicMock()
mock_avatar_module.AvatarLLMState = mock_avatar_llm_state

mock_history_module = MagicMock()
mock_history_module.LLMHistoryManager = mock_llm_history_manager

sys.modules['providers.examples.avatar_llm_state_provider'] = mock_avatar_module
sys.modules['providers.examples.llm_history_manager'] = mock_history_module

import pytest
from pydantic import BaseModel

from llm.output_model import Action, CortexOutputModel
from llm.plugins.exaone_llm import (
    ExaoneOllamaLLM,
    ExaoneOllamaLLMConfig,
    ExaoneVllmLLM,
    ExaoneVllmLLMConfig,
)


class DummyOutputModel(BaseModel):
    test_field: str


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ollama_config():
    """ExaoneOllamaLLM configuration fixture."""
    return ExaoneOllamaLLMConfig(
        base_url="http://localhost:11434",
        model="exaone3.5:7.8b",
        temperature=0.7,
        num_ctx=4096,
        timeout=120,
    )


@pytest.fixture
def vllm_config():
    """ExaoneVllmLLM configuration fixture."""
    return ExaoneVllmLLMConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        temperature=0.7,
        max_tokens=2048,
        timeout=120,
    )


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response without tool calls."""
    return {
        "message": {
            "role": "assistant",
            "content": "Hello, how can I help you?",
        }
    }


@pytest.fixture
def mock_ollama_response_with_json_actions():
    """Mock Ollama API response with JSON-formatted actions in content."""
    return {
        "message": {
            "role": "assistant",
            "content": '[{"action": "speak", "params": {"text": "Hello world"}}]',
        }
    }


@pytest.fixture
def mock_vllm_response():
    """Mock vLLM API response without tool calls."""
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content="Hello, how can I help you?",
                tool_calls=None,
            )
        )
    ]
    return response


@pytest.fixture
def mock_vllm_response_with_tool_calls():
    """Mock vLLM API response with tool calls."""
    tool_call = MagicMock()
    tool_call.function.name = "speak"
    tool_call.function.arguments = '{"text": "Hello world"}'

    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                content="",
                tool_calls=[tool_call],
            )
        )
    ]
    return response


@pytest.fixture(autouse=True)
def mock_avatar_components():
    """Mock Avatar/IO components to prevent Zenoh session creation.

    Note: AvatarLLMState and LLMHistoryManager are already mocked via
    sys.modules at the top of this file with passthrough decorators.
    """
    yield


# =============================================================================
# ExaoneOllamaLLMConfig Tests
# =============================================================================


class TestExaoneOllamaLLMConfig:
    def test_default_values(self):
        """Test default configuration values."""
        config = ExaoneOllamaLLMConfig()
        assert config.base_url == "http://localhost:11434"
        assert config.model == "exaone3.5:7.8b"
        assert config.temperature == 0.7
        assert config.num_ctx == 4096
        assert config.timeout == 120

    def test_custom_values(self, ollama_config):
        """Test custom configuration values."""
        assert ollama_config.base_url == "http://localhost:11434"
        assert ollama_config.model == "exaone3.5:7.8b"
        assert ollama_config.temperature == 0.7
        assert ollama_config.num_ctx == 4096
        assert ollama_config.timeout == 120


# =============================================================================
# ExaoneVllmLLMConfig Tests
# =============================================================================


class TestExaoneVllmLLMConfig:
    def test_default_values(self):
        """Test default configuration values."""
        config = ExaoneVllmLLMConfig()
        assert config.base_url == "http://127.0.0.1:8000/v1"
        assert config.model == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 120

    def test_custom_values(self, vllm_config):
        """Test custom configuration values."""
        assert vllm_config.base_url == "http://127.0.0.1:8000/v1"
        assert vllm_config.model == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
        assert vllm_config.temperature == 0.7
        assert vllm_config.max_tokens == 2048
        assert vllm_config.timeout == 120


# =============================================================================
# ExaoneOllamaLLM Tests
# =============================================================================


class TestExaoneOllamaLLM:
    @patch("llm.plugins.exaone_llm.httpx.AsyncClient")
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    def test_init(self, mock_history_manager, mock_openai_client, mock_httpx_client, ollama_config):
        """Test ExaoneOllamaLLM initialization."""
        llm = ExaoneOllamaLLM(ollama_config)

        assert llm._base_url == "http://localhost:11434"
        assert llm._chat_url == "http://localhost:11434/api/chat"
        assert mock_httpx_client.called

    @patch("llm.plugins.exaone_llm.httpx.AsyncClient")
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    def test_build_json_response_instruction(
        self, mock_history_manager, mock_openai_client, mock_httpx_client, ollama_config
    ):
        """Test JSON response instruction generation for prompt-based tool calling."""
        llm = ExaoneOllamaLLM(ollama_config)

        # Set function schemas manually
        llm.function_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "speak",
                    "description": "Speak a sentence",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]

        instruction = llm._build_json_response_instruction()
        assert "speak" in instruction
        assert "JSON" in instruction
        assert "action" in instruction

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.httpx.AsyncClient")
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_with_json_actions(
        self,
        mock_history_manager,
        mock_openai_client,
        mock_httpx_client_class,
        ollama_config,
        mock_ollama_response_with_json_actions,
    ):
        """Test ask method with JSON-formatted actions in content."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response_with_json_actions

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_httpx_client_class.return_value = mock_client

        llm = ExaoneOllamaLLM(ollama_config)
        result = await llm.ask("test prompt")

        assert isinstance(result, CortexOutputModel)
        assert len(result.actions) == 1
        assert result.actions[0].type == "speak"
        assert result.actions[0].value == "Hello world"

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.httpx.AsyncClient")
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_no_tool_calls(
        self,
        mock_history_manager,
        mock_openai_client,
        mock_httpx_client_class,
        ollama_config,
        mock_ollama_response,
    ):
        """Test ask method without tool calls."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_httpx_client_class.return_value = mock_client

        llm = ExaoneOllamaLLM(ollama_config)
        result = await llm.ask("test prompt")

        assert result is None

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.httpx.AsyncClient")
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_api_error(
        self, mock_history_manager, mock_openai_client, mock_httpx_client_class, ollama_config
    ):
        """Test ask method with API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_httpx_client_class.return_value = mock_client

        llm = ExaoneOllamaLLM(ollama_config)
        result = await llm.ask("test prompt")

        assert result is None

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.httpx.AsyncClient")
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_connection_error(
        self, mock_history_manager, mock_openai_client, mock_httpx_client_class, ollama_config
    ):
        """Test ask method with connection error."""
        import httpx

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_httpx_client_class.return_value = mock_client

        llm = ExaoneOllamaLLM(ollama_config)
        result = await llm.ask("test prompt")

        assert result is None

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.httpx.AsyncClient")
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_timeout_error(
        self, mock_history_manager, mock_openai_client, mock_httpx_client_class, ollama_config
    ):
        """Test ask method with timeout error."""
        import httpx

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))
        mock_httpx_client_class.return_value = mock_client

        llm = ExaoneOllamaLLM(ollama_config)
        result = await llm.ask("test prompt")

        assert result is None


# =============================================================================
# ExaoneVllmLLM Tests
# =============================================================================


class TestExaoneVllmLLM:
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    def test_init(self, mock_history_manager, mock_client_class, vllm_config):
        """Test ExaoneVllmLLM initialization."""
        llm = ExaoneVllmLLM(vllm_config)

        assert llm._config.model == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
        assert mock_client_class.called

    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    def test_init_default_model(self, mock_history_manager, mock_client_class):
        """Test initialization with default model."""
        config = ExaoneVllmLLMConfig(base_url="http://server:8000/v1")
        config.model = None  # Explicitly set to None

        llm = ExaoneVllmLLM(config)
        assert llm._config.model == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_with_tool_calls(
        self,
        mock_history_manager,
        mock_client_class,
        vllm_config,
        mock_vllm_response_with_tool_calls,
    ):
        """Test ask method with tool calls response."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_vllm_response_with_tool_calls
        )
        mock_client_class.return_value = mock_client

        llm = ExaoneVllmLLM(vllm_config)
        result = await llm.ask("test prompt")

        assert isinstance(result, CortexOutputModel)
        assert len(result.actions) == 1
        assert result.actions[0].type == "speak"
        assert result.actions[0].value == "Hello world"

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_no_tool_calls(
        self,
        mock_history_manager,
        mock_client_class,
        vllm_config,
        mock_vllm_response,
    ):
        """Test ask method without tool calls."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_vllm_response)
        mock_client_class.return_value = mock_client

        llm = ExaoneVllmLLM(vllm_config)
        result = await llm.ask("test prompt")

        assert result is None

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_empty_choices(
        self, mock_history_manager, mock_client_class, vllm_config
    ):
        """Test ask method with empty choices."""
        mock_response = MagicMock()
        mock_response.choices = []

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        llm = ExaoneVllmLLM(vllm_config)
        result = await llm.ask("test prompt")

        assert result is None

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_connection_error(
        self, mock_history_manager, mock_client_class, vllm_config
    ):
        """Test ask method with connection error."""
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )
        mock_client_class.return_value = mock_client

        llm = ExaoneVllmLLM(vllm_config)
        result = await llm.ask("test prompt")

        assert result is None

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_timeout_error(
        self, mock_history_manager, mock_client_class, vllm_config
    ):
        """Test ask method with timeout error."""
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )
        mock_client_class.return_value = mock_client

        llm = ExaoneVllmLLM(vllm_config)
        result = await llm.ask("test prompt")

        assert result is None

    @pytest.mark.asyncio
    @patch("llm.plugins.exaone_llm.openai.AsyncClient")
    @patch("llm.plugins.exaone_llm.LLMHistoryManager")
    async def test_ask_generic_error(
        self, mock_history_manager, mock_client_class, vllm_config
    ):
        """Test ask method with generic error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Unknown error")
        )
        mock_client_class.return_value = mock_client

        llm = ExaoneVllmLLM(vllm_config)
        result = await llm.ask("test prompt")

        assert result is None
