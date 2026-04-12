import logging

from hooks.action_hooks import ActionHook
from llm.output_model import CortexOutputModel

logger = logging.getLogger(__name__)


class NoVoiceMoveHook(ActionHook):
    """
    Hook 6: Blocks move actions when no voice input exists in the current tick.

    Field tests (v1.0, 2026-04-07) confirmed that the LLM self-generates
    move actions (resume, stop move) based on Navigation state even when
    the user gave no voice command.  This caused:
      - resume called twice → stored speed consumed, fallback to 0.2 m/s
      - stop move self-triggered → unexpected halt without user request

    This hook removes move actions when context.has_voice_input is False,
    keeping speak actions intact.
    """

    def validate(
        self, output: CortexOutputModel, context: dict
    ) -> CortexOutputModel:
        has_voice = context.get("has_voice_input", False)

        if has_voice:
            return output

        move_actions = [a for a in output.actions if a.type.lower() == "move"]
        if not move_actions:
            return output

        blocked = [a.value for a in move_actions]
        output.actions = [a for a in output.actions if a.type.lower() != "move"]
        logger.warning(
            f"[ActionHook:NoVoiceMove] BLOCKED move({', '.join(blocked)}) "
            f"— no voice input in this tick"
        )
        return output
