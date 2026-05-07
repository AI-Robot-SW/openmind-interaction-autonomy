from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """Navigation goals (L1/L2/L3/L8/NG + L8 indoor), speed adjustment, and posture (sit, stand, damp, stop)."""
    GO_TO_L1 = "go to L1"
    GO_TO_L2 = "go to L2"
    GO_TO_L3 = "go to L3"
    GO_TO_A0 = "go to A0"
    GO_TO_L8 = "go to L8"
    GO_TO_NG = "go to NG"
    GO_TO_L8_F1_MAIN_ENTRANCE = "go to L8 main entrance"
    GO_TO_L8_F1_STAIR_1 = "go to L8 stair"
    GO_TO_L8_F1_ELEVATOR_1 = "go to L8 elevator"
    SLOW_DOWN = "slow down"
    SPEED_UP = "speed up"
    STAND_UP = "stand up"
    STAND_DOWN = "stand down"
    DAMP = "damp"
    STOP_MOVE = "stop move"
    RESUME = "resume"


@dataclass
class MoveInput:
    """
    Input payload for move action.

    Parameters
    ----------
    action : MovementAction
        One of: go to L1/L2/L3/A0/L8/NG, slow down, speed up, stand up/down, damp, stop move, sit.
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    Move action interface.

    Navigation goals (L1/L2/L3/A0/L8/NG), speed control (slow down / speed up), and posture (sit, stand up/down, damp, stop move).
    """

    input: MoveInput
    output: MoveInput
