import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MovementAction, MoveInput
from providers.navigation_provider import NavigationProvider
from providers.unitree_go2_provider import UnitreeGo2Provider

# Waypoint path file for each navigation destination (relative to providers/).
# Add an entry here when a new destination path file is available.
_DESTINATION_PATHS: dict[MovementAction, str] = {
    MovementAction.GO_TO_L8: "utils/paths/EntL8-go2-0.75m.txt",
    MovementAction.GO_TO_NG: "utils/paths/EntNG-go2-0.75m.txt",
}

# Fixed forward speeds (m/s).  vy is always forced to 0.
# During initial heading calibration NavigationProvider emits GnssRoute commands
# (not DWA), so a slower speed is safer.
_VX_CALIBRATING = 0.5
_VX_RUN = 0.8


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
        self._already_stopped = False  # True after first stop_move() in a streak of idles
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

        if action in _DESTINATION_PATHS:
            path = _DESTINATION_PATHS[action]
            self._nav_provider.set_path(path)
            logging.info("MoveConnector forwarded destination to NavigationProvider.set_path: %s", path)
            self._already_stopped = False

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
            self._nav_provider.clear_path()
            self._unitree_provider.stop_move()
            self._already_stopped = True

        else:
            raise ValueError(f"Unknown move action: {action}")

    def tick(self) -> None:
        time.sleep(0.1)
        if self._nav_provider.running and self._nav_provider.get_active_path() is not None:
            state = self._nav_provider.get_state()
            vx, vy, vyaw = state.vx, state.vy, state.vyaw
        else:
            state = None

        if state is not None and any(abs(v) > 1e-6 for v in (vx, vy, vyaw)):
            # vy is always 0 (NavigationProvider already guarantees this, but be explicit)
            vy = 0.0
            # Override vx with a fixed speed so the robot moves at a predictable pace.
            # vx == 0 means turn-in-place — respect it and don't force forward motion.
            if abs(vx) > 1e-6:
                vx = _VX_CALIBRATING if state.mode == "CALIBRATING" else _VX_RUN
            logging.info(
                "MoveConnector forwarding move command: vx=%.3f vy=%.3f vyaw=%.3f (mode=%s)",
                vx,
                vy,
                vyaw,
                state.mode,
            )
            self._unitree_provider.move(vx, vy, vyaw)
            self._already_stopped = False
        else:
            if not self._already_stopped:
                logging.info("MoveConnector forwarding stop command from tick")
                self._unitree_provider.stop_move()
                self._already_stopped = True
