import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.move_caution.interface import MoveCautionInput


class MoveCautionConfig(ActionConfig):
    """
    Configuration for MoveCaution action connector.
    """

    pass


class MoveCautionConnector(ActionConnector[MoveCautionConfig, MoveCautionInput]):
    def __init__(self, config: MoveCautionConfig):
        super().__init__(config)
        # TODO: self._nav_provider = NavProvider()
        self._nav_provider = None

    async def connect(self, output_interface: MoveCautionInput) -> None:
        """
        Connect the input protocol to the caution move action.
        Only slow_down and stop_move are allowed.

        Parameters
        ----------
        output_interface : MoveCautionInput
            The input protocol containing the action details.
        """
        action_val = output_interface.action.value
        try:
            if action_val == "slow down":
                pass  # TODO: NavProvider.step_slower()
            elif action_val == "stop move":
                pass  # TODO: NavProvider.set_goal(None)
            else:
                logging.warning("Unknown caution move type: %s", output_interface.action)
                raise ValueError(f"Unknown caution move type: {output_interface.action}")
        except ValueError:
            raise
        except Exception as e:
            logging.error("MoveCautionConnector connect failed for action %s: %s", action_val, e)
            raise

    def tick(self) -> None:
        time.sleep(0.1)
        move_cmd = self._nav_provider.get_next_move() if self._nav_provider else None
        if move_cmd is not None:
            # In caution mode, enforce reduced speed
            logging.info("Caution mode: executing move command with safety constraints")
