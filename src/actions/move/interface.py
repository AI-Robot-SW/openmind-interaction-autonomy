from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """Navigation goals (L1/L2/L3), speed adjustment, and posture (sit, stand, damp, stop)."""
    GO_TO_L1 = "go to L1"
    GO_TO_L2 = "go to L2"
    GO_TO_L3 = "go to L3"
    SLOW_DOWN = "slow down"
    SPEED_UP = "speed up"
    STAND_UP = "stand up"
    STAND_DOWN = "stand down"
    DAMP = "damp"
    STOP_MOVE = "stop move"


@dataclass
class MoveInput:
    """
    Input payload for move action.

    Parameters
    ----------
    action : MovementAction
        One of: go to L1/L2/L3, slow down, speed up, stand up/down, damp, stop move, sit.
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    Move action interface.

    Navigation goals (L1/L2/L3), speed control (slow down / speed up), and posture (sit, stand up/down, damp, stop move).
    """

    input: MoveInput
    output: MoveInput
