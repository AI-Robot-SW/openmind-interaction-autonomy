"""
EXAONE LLM implementations for Ollama and vLLM backends.

This module provides two LLM implementations for the EXAONE 3.5 model:
- ExaoneOllamaLLM: Uses Ollama's native REST API
- ExaoneVllmLLM: Uses vLLM's OpenAI-compatible API
"""

import json
import logging
import time
import typing as T

import httpx
import openai
from pydantic import BaseModel, Field

from llm import LLM, LLMConfig
from llm.function_schemas import convert_function_calls_to_actions
from llm.output_model import CortexOutputModel
from providers.llm_history_manager import LLMHistoryManager

R = T.TypeVar("R", bound=BaseModel)


# =============================================================================
# ExaoneOllamaLLM - Ollama Backend
# =============================================================================


class ExaoneOllamaLLMConfig(LLMConfig):
    """
    Configuration for EXAONE LLM with Ollama backend.

    Parameters
    ----------
    base_url : str
        Base URL for Ollama API (default: http://localhost:11434)
    model : str
        Ollama model name (default: exaone3.5:7.8b)
    temperature : float
        Sampling temperature (0.0 - 1.0)
    num_ctx : int
        Context window size
    """

    base_url: T.Optional[str] = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API",
    )
    model: T.Optional[str] = Field(
        default="exaone3.5:7.8b",
        description="Ollama model name",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
    )
    num_ctx: int = Field(
        default=4096,
        description="Context window size",
    )
    timeout: T.Optional[int] = Field(
        default=120,
        description="Request timeout in seconds (longer for local inference)",
    )


class ExaoneOllamaLLM(LLM[R]):
    """
    EXAONE LLM implementation using Ollama backend.

    This class implements the LLM interface for local EXAONE models via Ollama,
    providing privacy-focused, cost-free, offline-capable inference.

    Config example:
        "cortex_llm": {
            "type": "ExaoneOllamaLLM",
            "config": {
                "model": "exaone3.5:7.8b",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "num_ctx": 4096,
                "timeout": 120,
                "history_length": 10
            }
        }

    Parameters
    ----------
    config : ExaoneOllamaLLMConfig
        Configuration object containing Ollama settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation.
    """

    def __init__(
        self,
        config: ExaoneOllamaLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the ExaoneOllamaLLM instance.

        Parameters
        ----------
        config : ExaoneOllamaLLMConfig
            Configuration settings for Ollama.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        self._config: ExaoneOllamaLLMConfig = config

        self._base_url = (self._config.base_url or "http://localhost:11434").rstrip("/")
        self._chat_url = f"{self._base_url}/api/chat"

        self._client = httpx.AsyncClient(timeout=config.timeout)

        # Initialize history manager with Ollama-compatible OpenAI client
        self._openai_client = openai.AsyncClient(
            base_url=f"{config.base_url}/v1",
            api_key="ollama",
        )
        self.history_manager = LLMHistoryManager(self._config, self._openai_client)

        logging.info(f"ExaoneOllamaLLM initialized with model: {config.model}")
        logging.info(f"Ollama endpoint: {self._chat_url}")

    def _build_json_response_instruction(self) -> str:
        """
        Build instruction for JSON-formatted response.

        Since EXAONE on Ollama doesn't support native tool calling,
        we instruct the model to respond with JSON format.

        Returns
        -------
        str
            JSON response instruction to append to prompt
        """
        if not self.function_schemas:
            return ""

        actions_desc = []
        for schema in self.function_schemas:
            func = schema.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})

            param_desc = []
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "string")
                param_enum = param_info.get("enum", [])
                if param_enum:
                    param_desc.append(f'"{param_name}": one of {param_enum}')
                else:
                    param_desc.append(f'"{param_name}": {param_type}')

            actions_desc.append(f'  - {name}: {desc} (params: {{{", ".join(param_desc)}}})')

        instruction = """

Respond ONLY with a JSON array of actions. No other text.
IMPORTANT: Always include BOTH a "move" action AND a "speak" action. Never omit either one.
Available actions:
""" + "\n".join(actions_desc) + """

Response format (JSON array only):
[{"action": "action_name", "params": {"param_name": "value"}}]

Example:
[{"action": "move", "params": {"action": "go to L1"}}, {"action": "speak", "params": {"action": "Sure, I will guide you to L1."}}]
"""
        return instruction

    def _parse_json_response(self, content: str) -> T.List[T.Dict]:
        """
        Parse JSON response from model content.

        Parameters
        ----------
        content : str
            Raw response content from the model

        Returns
        -------
        list
            List of parsed action dictionaries
        """
        if not content:
            return []

        # Try to extract JSON from response
        content = content.strip()

        # Find JSON array in response
        start_idx = content.find("[")
        end_idx = content.rfind("]")

        if start_idx == -1 or end_idx == -1:
            logging.warning(f"No JSON array found in response: {content[:200]}")
            return []

        json_str = content[start_idx : end_idx + 1]

        try:
            actions = json.loads(json_str)
            if not isinstance(actions, list):
                actions = [actions]
            return actions
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse JSON response: {e}")
            logging.debug(f"Raw JSON string: {json_str}")
            return []

    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> T.Optional[R]:
        """
        Send a prompt to Ollama EXAONE and get a structured response.

        Uses prompt-based JSON response instead of native tool calling,
        since EXAONE on Ollama doesn't support native function calling.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.
        messages : List[Dict[str, str]]
            List of message dictionaries for conversation history.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails.
        """
        try:
            logging.info(f"ExaoneOllama input: {prompt}")
            logging.debug(f"ExaoneOllama messages: {messages}")

            if self.io_provider is not None:
                self.io_provider.llm_start_time = time.time()
                self.io_provider.set_llm_prompt(prompt)

            # Add JSON response instruction to prompt
            json_instruction = self._build_json_response_instruction()
            full_prompt = prompt + json_instruction

            formatted_messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]
            formatted_messages.append({"role": "user", "content": full_prompt})

            payload = {
                "model": self._config.model,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": self._config.temperature,
                    "num_ctx": self._config.num_ctx,
                },
            }

            # Note: No 'tools' parameter - using prompt-based JSON response instead

            logging.debug(f"Ollama request payload: {json.dumps(payload, indent=2)}")

            response = await self._client.post(
                self._chat_url,
                json=payload,
            )

            if response.status_code != 200:
                logging.error(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                return None

            result = response.json()
            if self.io_provider is not None:
                self.io_provider.llm_end_time = time.time()

            logging.debug(f"Ollama response: {json.dumps(result, indent=2)}")

            message = result.get("message", {})
            content = message.get("content", "")

            if content:
                logging.info(f"ExaoneOllama response: {content[:300]}...")

                # Parse JSON response from content
                parsed_actions = self._parse_json_response(content)

                if parsed_actions:
                    logging.info(f"Parsed {len(parsed_actions)} actions from response")
                    logging.info(f"Actions: {parsed_actions}")

                    # Convert to function call format for compatibility
                    function_call_data = []
                    for action in parsed_actions:
                        action_name = action.get("action", "")
                        params = action.get("params", {})
                        function_call_data.append(
                            {
                                "function": {
                                    "name": action_name,
                                    "arguments": json.dumps(params),
                                }
                            }
                        )

                    if function_call_data:
                        actions = convert_function_calls_to_actions(function_call_data)
                        result_model = CortexOutputModel(actions=actions)
                        return T.cast(R, result_model)

            return None

        except httpx.ConnectError as e:
            logging.error(
                f"Cannot connect to Ollama at {self._base_url}. Is Ollama running?"
            )
            logging.error("Start Ollama with: ollama serve")
            logging.error(f"Error: {e}")
            return None
        except httpx.TimeoutException as e:
            logging.error(f"Ollama request timed out after {self._config.timeout}s")
            logging.error("Try increasing timeout or using a smaller model")
            logging.error(f"Error: {e}")
            return None
        except Exception as e:
            logging.error(f"ExaoneOllama API error: {e}")
            return None

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


# =============================================================================
# ExaoneVllmLLM - vLLM Backend (OpenAI-compatible API)
# =============================================================================


class ExaoneVllmLLMConfig(LLMConfig):
    """
    Configuration for EXAONE LLM with vLLM backend.

    Parameters
    ----------
    base_url : str
        Base URL for vLLM API (default: http://127.0.0.1:8000/v1)
    model : str
        vLLM model name (default: LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct)
    temperature : float
        Sampling temperature (0.0 - 1.0)
    max_tokens : int
        Maximum number of tokens to generate
    """

    base_url: T.Optional[str] = Field(
        default="http://127.0.0.1:8000/v1",
        description="Base URL for vLLM OpenAI-compatible API",
    )
    model: T.Optional[str] = Field(
        default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        description="vLLM model name",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate",
    )
    timeout: T.Optional[int] = Field(
        default=120,
        description="Request timeout in seconds",
    )


class ExaoneVllmLLM(LLM[R]):
    """
    EXAONE LLM implementation using vLLM OpenAI-compatible backend.

    This class implements the LLM interface for local EXAONE models via vLLM,
    using the OpenAI-compatible API for seamless integration.

    Config example:
        "cortex_llm": {
            "type": "ExaoneVllmLLM",
            "config": {
                "model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
                "base_url": "http://127.0.0.1:8000/v1",
                "temperature": 0.7,
                "max_tokens": 2048,
                "timeout": 120,
                "history_length": 10
            }
        }

    Parameters
    ----------
    config : ExaoneVllmLLMConfig
        Configuration object containing vLLM settings.
    available_actions : list[AgentAction], optional
        List of available actions for function call generation.
    """

    def __init__(
        self,
        config: ExaoneVllmLLMConfig,
        available_actions: T.Optional[T.List] = None,
    ):
        """
        Initialize the ExaoneVllmLLM instance.

        Parameters
        ----------
        config : ExaoneVllmLLMConfig
            Configuration settings for vLLM.
        available_actions : list[AgentAction], optional
            List of available actions for function calling.
        """
        super().__init__(config, available_actions)

        self._config: ExaoneVllmLLMConfig = config

        if not config.model:
            self._config.model = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

        base_url = (config.base_url or "http://127.0.0.1:8000/v1").rstrip("/")

        self._client = openai.AsyncClient(
            base_url=base_url,
            api_key="placeholder_key",  # vLLM doesn't require real API key
        )

        self.history_manager = LLMHistoryManager(self._config, self._client)

        self._skip_state_management = False

        logging.info(f"ExaoneVllmLLM initialized with model: {config.model}")
        logging.info(f"vLLM endpoint: {base_url}")

    @LLMHistoryManager.update_history()
    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, T.Any]] = []
    ) -> T.Optional[R]:
        """
        Send prompt to vLLM EXAONE model and get structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send.
        messages : list of dict, optional
            Conversation history (default: []).

        Returns
        -------
        R or None
            Parsed response with actions, or None if parsing fails.
        """
        try:
            logging.info(f"ExaoneVllm input: {prompt}")
            logging.debug(f"ExaoneVllm messages: {messages}")

            if self.io_provider is not None:
                self.io_provider.llm_start_time = time.time()
                self.io_provider.set_llm_prompt(prompt)

            formatted = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in messages
            ]
            formatted.append({"role": "user", "content": prompt})

            model = self._config.model or "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

            request_params: T.Dict[str, T.Any] = {
                "model": model,
                "messages": formatted,
                "timeout": self._config.timeout,
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
            }

            if self.function_schemas:
                request_params["tools"] = self.function_schemas
                request_params["tool_choice"] = "auto"

            logging.debug(f"vLLM request params: {request_params}")

            response = await self._client.chat.completions.create(**request_params)

            if not response.choices:
                logging.warning("vLLM API returned empty choices")
                return None

            message = response.choices[0].message
            if self.io_provider is not None:
                self.io_provider.llm_end_time = time.time()

            tool_calls = list(message.tool_calls or [])

            if tool_calls:
                logging.info(f"Received {len(tool_calls)} function calls from vLLM")
                logging.info(f"Function calls: {tool_calls}")

                function_call_data = [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls
                ]
                actions = convert_function_calls_to_actions(function_call_data)
                result = CortexOutputModel(actions=actions)
                return T.cast(R, result)

            # If no tool calls, check for content and try to parse as action
            if message.content:
                logging.info(f"ExaoneVllm response content: {message.content}")

            return None

        except openai.APIConnectionError as e:
            logging.error(
                f"Cannot connect to vLLM at {self._config.base_url}. Is vLLM running?"
            )
            logging.error(f"Error: {e}")
            return None
        except openai.APITimeoutError as e:
            logging.error(f"vLLM request timed out after {self._config.timeout}s")
            logging.error(f"Error: {e}")
            return None
        except Exception as e:
            logging.error(f"ExaoneVllm LLM error: {e}")
            return None

    async def close(self):
        """Close the OpenAI client."""
        await self._client.close()
