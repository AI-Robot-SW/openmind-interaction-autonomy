import logging
import time

from actions.base import ActionConfig, ActionConnector
from actions.move.interface import MoveInput
from providers.dwa_route_provider import DwaRouteProvider
from providers.gnss_route_provider import GnssRouteProvider
from providers.navigation_provider import NavigationProvider
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
        try:
            gnss_provider = GnssRouteProvider(waypoints=[])
            dwa_provider = DwaRouteProvider(gnss_route_provider=gnss_provider)
            self._nav_provider = NavigationProvider(
                gnss=gnss_provider,
                dwa=dwa_provider,
            )
        except Exception as e:
            logging.warning("MoveConnector navigation provider init failed: %s", e)
            self._nav_provider = None
        self._already_stopped = False  # True after first stop_move() in a streak of Nones
        logging.info(
            "MoveConnector initialized (navigation_provider=%s)",
            "ready" if self._nav_provider is not None else "unavailable",
        )

    async def connect(self, output_interface: MoveInput) -> None:
        """
        Connect the input protocol to the move action.

        Parameters
        ----------
        output_interface : MoveInput
            The input protocol containing the action details.
        """
        action_val = output_interface.action.value
        logging.info("MoveConnector received action: %s", action_val)
        try:
            if action_val == "go to L8":
                path = "EntL8-go2-0.75m.txt"
                logging.info("MoveConnector destination path requested: %s", path)
                if self._nav_provider is None:
                    logging.warning(
                        "MoveConnector destination path %s ignored: navigation provider not configured",
                        path,
                    )
                else:
                    self._nav_provider.set_path(path)
                    logging.info(
                        "MoveConnector forwarded destination path to NavigationProvider.set_path: %s",
                        path,
                    )
                    self._already_stopped = False
            elif action_val == "slow down":
                if self._nav_provider is None:
                    logging.warning("MoveConnector slow down ignored: navigation provider not configured")
                else:
                    logging.info("MoveConnector forwarding speed change: slower")
                    self._nav_provider.step_slower()
            elif action_val == "speed up":
                if self._nav_provider is None:
                    logging.warning("MoveConnector speed up ignored: navigation provider not configured")
                else:
                    logging.info("MoveConnector forwarding speed change: faster")
                    self._nav_provider.step_faster()
            elif action_val == "stand up":
                logging.info("MoveConnector forwarding posture command: stand_up")
                self.unitree_go2_provider.stand_up()
            elif action_val == "stand down":
                logging.info("MoveConnector forwarding posture command: stand_down")
                self.unitree_go2_provider.stand_down()
            elif action_val == "damp":
                logging.info("MoveConnector forwarding posture command: damp")
                self.unitree_go2_provider.damp()
            elif action_val == "stop move":
                # TODO: When nav provider supports goal cancellation, clear the active path here too.
                # 로봇 속도 0으로 만드는 건데 현재 목표를 취소할건지 경로 추종을 중단하는 멈추기를 할건지?
                logging.info("MoveConnector forwarding stop command")
                self.unitree_go2_provider.stop_move()
                self._already_stopped = True
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
        move_cmd = None
        if (
            self._nav_provider is not None
            and self._nav_provider.running
            and self._nav_provider.get_active_path() is not None
        ):
            move_cmd = self._nav_provider.get_next_move()
        if move_cmd is not None:
            if isinstance(move_cmd, tuple):
                vx, vy, vyaw = move_cmd
            else:
                vx, vy, vyaw = move_cmd.vx, move_cmd.vy, move_cmd.vyaw
            logging.info(
                "MoveConnector forwarding move command: vx=%.3f vy=%.3f vyaw=%.3f",
                vx,
                vy,
                vyaw,
            )
            self.unitree_go2_provider.move(vx, vy, vyaw)
            self._already_stopped = False
        else:
            if not self._already_stopped:
                logging.info("MoveConnector forwarding stop command from tick")
                self.unitree_go2_provider.stop_move()
                self._already_stopped = True
