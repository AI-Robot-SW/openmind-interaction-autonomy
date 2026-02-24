from dataclasses import dataclass

from actions.base import Interface


@dataclass
class IDLEInput:
    action: str


@dataclass
class GameController(Interface[IDLEInput, IDLEInput]):
    """
    Game controller interface
    """

    input: IDLEInput
    output: IDLEInput
