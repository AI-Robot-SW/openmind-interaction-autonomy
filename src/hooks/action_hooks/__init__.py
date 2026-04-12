import logging
from abc import ABC, abstractmethod
from typing import List

from llm.output_model import CortexOutputModel


class ActionHook(ABC):
    """Base class for all Action Hooks."""

    @abstractmethod
    def validate(
        self, output: CortexOutputModel, context: dict
    ) -> CortexOutputModel:
        """
        Validate and optionally modify the LLM output before action dispatch.

        Parameters
        ----------
        output : CortexOutputModel
            LLM-generated action list.
        context : dict
            Current tick system state (has_voice_input, nav_mode, nav_goal, tick_number).

        Returns
        -------
        CortexOutputModel
            Validated/modified action list.
        """
        pass


class ActionHookChain:
    """Runs registered hooks in order against each LLM output."""

    def __init__(self, hooks: List[ActionHook]):
        self.hooks = hooks
        logging.info(
            f"ActionHookChain initialized with {len(hooks)} hooks: "
            f"{[type(h).__name__ for h in hooks]}"
        )

    def validate(
        self, output: CortexOutputModel, context: dict
    ) -> CortexOutputModel:
        for hook in self.hooks:
            output = hook.validate(output, context)
        return output
