import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput
from providers.unitree_go2_provider import UnitreeGo2Provider


class MoveConfig(ActionConfig):
    """
    Configuration for Move action connector.

    Parameters
    ----------
    (Add fields as needed.)
    """

    pass


class MoveConnector(ActionConnector[MoveConfig, MoveInput]):
    def __init__(self, config: MoveConfig):
        super().__init__(config)
        self.unitree_go2_provider = UnitreeGo2Provider()
        # TODO: self._nav_provider = NavProvider()
        self._nav_provider = None
        self._already_stopped = False  # True after first stop_move() in a streak of Nones

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect the input protocol to the move action.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol containing the action details.
        """
        action_val = output_interface.action.value
        try:
            if action_val == "go to L1":
                pass  # TODO: NavProvider.set_goal("L1")
            elif action_val == "go to L2":
                pass  # TODO: NavProvider.set_goal("L2")
            elif action_val == "go to L3":
                pass  # TODO: NavProvider.set_goal("L3")
            elif action_val == "slow down":
                pass  # TODO: NavProvider.step_slower()
            elif action_val == "speed up":
                pass  # TODO: NavProvider.step_faster()
            elif action_val == "stand up":
                self.unitree_go2_provider.stand_up()
            elif action_val == "stand down":
                self.unitree_go2_provider.stand_down()
            elif action_val == "damp":
                self.unitree_go2_provider.damp()
            elif action_val == "stop move":
                pass  # TODO: NavProvider.set_goal(None)
            else:
                logging.warning("Unknown move type: %s", output_interface.action)
                raise ValueError(f"Unknown move type: {output_interface.action}")
        except ValueError:
            raise
        except Exception as e:
            logging.error("MoveConnector connect failed for action %s: %s", action_val, e)
            raise

    def tick(self) -> None:
        time.sleep(0.1)
        move_cmd = self._nav_provider.get_next_move() if self._nav_provider else None
        if move_cmd is not None:
            self.unitree_go2_provider.move(move_cmd.vx, move_cmd.vy, move_cmd.vyaw)
            self._already_stopped = False
        else:
            if not self._already_stopped:
                self.unitree_go2_provider.stop_move()
                self._already_stopped = True

