from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class MovementAction(str, Enum):
    """Navigation goals (L8/NG), speed adjustment, and posture (sit, stand, damp, stop)."""
    GO_TO_L8 = "go to L8"
    GO_TO_NG = "go to NG"
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
        One of: go to L8/NG, slow down, speed up, stand up/down, damp, stop move, resume.
    """

    action: MovementAction


@dataclass
class Move(Interface[MoveInput, MoveInput]):
    """
    Move action interface.

    Navigation goals (L8/NG), speed control (slow down / speed up / resume), and posture (sit, stand up/down, damp, stop move).
    """

    input: MoveInput
    output: MoveInput
