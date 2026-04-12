import logging
from typing import List

from hooks.action_hooks import ActionHook
from llm.output_model import CortexOutputModel

logger = logging.getLogger(__name__)

# Keywords in speak text that imply the robot intends to move.
MOVE_INTENT_KEYWORDS: List[str] = [
    "가자",
    "이동",
    "출발",
    "가겠습니다",
    "갈게요",
    "안내할게요",
    "따라와",
    "go to",
    "move to",
    "head to",
    "navigate",
]


class ConsistencyHook(ActionHook):
    """
    Hook 9: Checks semantic consistency between speak and move actions.

    Two checks (warning-only — no action modification):
      A. speak contains move-intent keywords but no move action exists.
      B. move action exists but no speak action to inform the user.

    This hook does NOT inject or remove actions because false positives
    are possible (e.g. "도서관에 가자고 말씀하셨는데, 해당 목적지는 지정되어
    있지 않습니다" contains "가자" but should NOT trigger a move).
    Warnings are logged for monitoring; promotion to blocking is deferred
    until field data confirms real inconsistency patterns.
    """

    def validate(
        self, output: CortexOutputModel, context: dict
    ) -> CortexOutputModel:
        speak_actions = [a for a in output.actions if a.type.lower() == "speak"]
        move_actions = [a for a in output.actions if a.type.lower() == "move"]

        speak_text = " ".join(a.value for a in speak_actions)

        # Rule A: speak has move intent but no move action
        if speak_text and not move_actions:
            matched = [kw for kw in MOVE_INTENT_KEYWORDS if kw in speak_text]
            if matched:
                logger.warning(
                    f"[ActionHook:Consistency] WARNING speak contains move "
                    f"intent '{matched[0]}' but no move action"
                )

        # Rule B: move exists but no speak to inform user
        if move_actions and not speak_actions:
            logger.warning(
                f"[ActionHook:Consistency] WARNING move({move_actions[0].value}) "
                f"without speak — user not informed"
            )

        return output
