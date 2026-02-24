from dataclasses import dataclass

from actions.base import Interface


@dataclass
class SpeakInput:
    """
    Input interface for the Speak action.

    Parameters
    ----------
    action : str
        The text to be spoken.
    interrupt : bool
        현재 재생을 중단하고 새로 시작할지 여부
    """

    action: str
    interrupt: bool = True


@dataclass
class Speak(Interface[SpeakInput, SpeakInput]):
    """
    This action allows you to speak.
    """

    # TODO : LLM interaction case-study 이후 speak 관련 액션 분기 생성
 
    input: SpeakInput
    output: SpeakInput

