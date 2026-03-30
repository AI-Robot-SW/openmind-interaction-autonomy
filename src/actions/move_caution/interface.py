from dataclasses import dataclass
from enum import Enum

from actions.base import Interface


class CautionMovementAction(str, Enum):
    """Restricted movement actions for Caution mode: only slow down and stop."""
    SLOW_DOWN = "slow down"
    STOP_MOVE = "stop move"


@dataclass
class MoveCautionInput:
    """
    Input payload for move_caution action (Caution mode only).

    Parameters
    ----------
    action : CautionMovementAction
        One of: slow down, stop move.
    """

    action: CautionMovementAction


@dataclass
class MoveCaution(Interface[MoveCautionInput, MoveCautionInput]):
    """
    Caution move action interface.

    Restricted movement for Caution mode: only slow down and stop move are allowed.
    """

    input: MoveCautionInput
    output: MoveCautionInput
