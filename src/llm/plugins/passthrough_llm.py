# passthrough_llm.py

import typing as T

from llm import LLM, LLMConfig
from llm.output_model import CortexOutputModel


class PassthroughLLM(LLM[CortexOutputModel]):
    """
    LLM 없이 하드웨어/background만 동작시킬 때 사용하는 더미 LLM.

    ask()가 항상 None을 반환하므로 cortex loop는 돌지만
    어떠한 액션도 실행되지 않는다.

    config 예시:
        "cortex_llm": { "type": "PassthroughLLM", "config": {} }
    """

    def __init__(self, config: LLMConfig, available_actions: T.Optional[list] = None):
        super().__init__(config=config, available_actions=available_actions)

    async def ask(
        self, prompt: str, messages: T.List[T.Dict[str, str]] = []
    ) -> None:
        return None
