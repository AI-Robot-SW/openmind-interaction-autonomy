import logging

from hooks.action_hooks import ActionHook
from llm.output_model import CortexOutputModel

logger = logging.getLogger(__name__)


class RepeatSpeakHook(ActionHook):
    """
    Hook 7: Blocks repeated identical speak output across consecutive ticks.

    Field tests (v1.0, 2026-04-09) confirmed the LLM repeating the same
    greeting every tick when no voice input was present:
        tick 1: speak("안내를 시작하겠습니다. 목적지를 말씀해주세요.")
        tick 2: speak("안내를 시작하겠습니다. 목적지를 말씀해주세요.")  ← repeat
        tick 3: speak("안내를 시작하겠습니다. 목적지를 말씀해주세요.")  ← repeat

    When voice input IS present, the same speak text is allowed — the user
    may have asked the same question again.
    """

    def __init__(self):
        self._last_speak: str | None = None

    def validate(
        self, output: CortexOutputModel, context: dict
    ) -> CortexOutputModel:
        has_voice = context.get("has_voice_input", False)

        speak_actions = [a for a in output.actions if a.type.lower() == "speak"]
        if not speak_actions:
            return output

        speak_text = speak_actions[0].value

        # Block repeat only when there is no new voice input
        if speak_text == self._last_speak and not has_voice:
            logger.warning(
                f"[ActionHook:RepeatSpeak] BLOCKED repeat speak "
                f"'{speak_text[:30]}...' (no new voice input)"
            )
            output.actions = []
            return output

        self._last_speak = speak_text
        return output

    def reset(self):
        """Reset state on config reload."""
        self._last_speak = None
