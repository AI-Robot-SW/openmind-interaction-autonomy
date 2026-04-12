import logging

from hooks.action_hooks import ActionHook
from llm.output_model import CortexOutputModel

logger = logging.getLogger(__name__)

MAX_ACTIONS_PER_TICK = 3


class ActionCountHook(ActionHook):
    """
    Hook 4: Limits the number of actions per tick.

    Pedestrian Companion Robots has only two registered action types (speak, move), so a normal
    tick produces at most 2 actions.  More than MAX_ACTIONS_PER_TICK
    indicates an LLM parsing error or hallucination — excess actions
    are dropped.
    """

    def __init__(self, max_actions: int = MAX_ACTIONS_PER_TICK):
        self._max_actions = max_actions

    def validate(
        self, output: CortexOutputModel, context: dict
    ) -> CortexOutputModel:
        count = len(output.actions)
        if count > self._max_actions:
            logger.warning(
                f"[ActionHook:ActionCount] TRIMMED {count} actions → "
                f"{self._max_actions} (exceeded max per tick)"
            )
            output.actions = output.actions[: self._max_actions]
        return output
