import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MovementAction, MoveInput
from providers.navigation_provider import NavigationProvider
from providers.unitree_go2_provider import UnitreeGo2Provider

# place_id for each navigation destination.
# Add an entry here when a new destination place is registered in the graph.
_DESTINATION_PLACES: dict[MovementAction, str] = {
    MovementAction.GO_TO_L8: "l8",
    MovementAction.GO_TO_NG: "north_gate",
    MovementAction.GO_TO_L8_F1_MAIN_ENTRANCE: "l8_f1_main_entrance",
    MovementAction.GO_TO_L8_F1_STAIR_1: "l8_f1_stair_1",
    MovementAction.GO_TO_L8_F1_ELEVATOR_1: "l8_f1_elevator_1",
}


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
        self._unitree_provider = UnitreeGo2Provider()
        self._nav_provider = NavigationProvider()
        logging.info("MoveConnector initialized")

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect the input protocol to the move action.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol containing the action details.
        """
        action = output_interface.action
        logging.info("MoveConnector received action: %s", action)

        if action in _DESTINATION_PLACES:
            place_id = _DESTINATION_PLACES[action]
            self._nav_provider.set_goal(place_id)
            logging.info("MoveConnector forwarded destination to NavigationProvider.set_goal: %s", place_id)

        elif action == MovementAction.SLOW_DOWN:
            logging.info("MoveConnector forwarding speed change: slower")
            self._nav_provider.step_slower()

        elif action == MovementAction.SPEED_UP:
            logging.info("MoveConnector forwarding speed change: faster")
            self._nav_provider.step_faster()

        elif action == MovementAction.STAND_UP:
            logging.info("MoveConnector forwarding posture command: stand_up")
            self._unitree_provider.stand_up()

        elif action == MovementAction.STAND_DOWN:
            logging.info("MoveConnector forwarding posture command: stand_down")
            self._unitree_provider.stand_down()

        elif action == MovementAction.DAMP:
            logging.info("MoveConnector forwarding posture command: damp")
            self._unitree_provider.damp()

        elif action == MovementAction.STOP_MOVE:
            logging.info("MoveConnector forwarding stop command")
            self._nav_provider.pause()
            self._unitree_provider.stop_move()

        elif action == MovementAction.RESUME:
            logging.info("MoveConnector forwarding resume command")
            self._nav_provider.resume()

        else:
            raise ValueError(f"Unknown move action: {action}")

    def tick(self) -> None:
        time.sleep(0.1)
        if not (self._nav_provider.running and self._nav_provider.get_active_goal() is not None):
            return

        state = self._nav_provider.get_state()
        vx, vy, vyaw = state.vx, state.vy, state.vyaw

        if any(abs(v) > 1e-6 for v in (vx, vy, vyaw)):
            vy = 0.0
            logging.info(
                "MoveConnector forwarding move command: vx=%.3f vy=%.3f vyaw=%.3f (mode=%s)",
                vx,
                vy,
                vyaw,
                state.mode,
            )
            self._unitree_provider.move(vx, vy, vyaw)
