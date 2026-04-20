"""
action_hooks — LLM 출력(actions)을 Action Orchestrator에 전달하기 전에
               룰 기반으로 검증/차단/보정하는 Post-LLM 가드 레이어.

새로운 검증 규칙이 필요하면 이 파일에 함수를 추가하고,
ActionHookChain.DEFAULT_HOOKS에 등록하면 됩니다.

See docs/action_hooks.md for full specification.
"""

import logging
from typing import Callable, Dict, List, Optional

from llm.output_model import CortexOutputModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Hook 함수 시그니처: (output, context, state) -> output
#   output : CortexOutputModel — LLM이 생성한 action 리스트
#   context: dict — 현재 tick 시스템 상태 (has_voice_input, tick_number, ...)
#   state  : dict — Hook 간 공유 상태 (직전 speak 텍스트 등, chain이 관리)
HookFn = Callable[[CortexOutputModel, dict, dict], CortexOutputModel]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ACTIONS_PER_TICK = 3

MOVE_INTENT_KEYWORDS: List[str] = [
    "가자", "이동", "출발", "가겠습니다", "갈게요",
    "안내할게요", "따라와",
    "go to", "move to", "head to", "navigate",
]

# ---------------------------------------------------------------------------
# Hook functions
# ---------------------------------------------------------------------------


def action_count_hook(
    output: CortexOutputModel, context: dict, state: dict
) -> CortexOutputModel:
    """비정상적으로 많은 action이 생성된 경우 초과분을 drop한다. (max 3)"""
    count = len(output.actions)
    if count > MAX_ACTIONS_PER_TICK:
        logger.warning(
            f"[ActionHook:ActionCount] TRIMMED {count} actions → "
            f"{MAX_ACTIONS_PER_TICK} (exceeded max per tick)"
        )
        output.actions = output.actions[:MAX_ACTIONS_PER_TICK]
    return output


def no_voice_move_hook(
    output: CortexOutputModel, context: dict, state: dict
) -> CortexOutputModel:
    """Voice 입력이 없는 tick에서 LLM이 자체 판단으로 생성한 move를 차단한다.

    필드 테스트 확인 오류 (2026-04-07):
      - resume 중복 호출 → 속도 0.80 → 0.20 fallback
      - stop move 자체 판단 → 사용자 미요청 정지
    """
    if context.get("has_voice_input", False):
        return output

    move_actions = [a for a in output.actions if a.type.lower() == "move"]
    if not move_actions:
        return output

    blocked = [a.value for a in move_actions]
    output.actions = [a for a in output.actions if a.type.lower() != "move"]
    logger.warning(
        f"[ActionHook:NoVoiceMove] BLOCKED move({', '.join(blocked)}) "
        f"— no voice input in this tick"
    )
    return output


def repeat_speak_hook(
    output: CortexOutputModel, context: dict, state: dict
) -> CortexOutputModel:
    """직전 tick과 동일한 speak을 반복하는 경우 차단한다.

    필드 테스트 확인 오류 (2026-04-09):
      - "안내를 시작하겠습니다. 목적지를 말씀해주세요." 무한 반복
    Voice 입력이 있는 경우에는 같은 응답이어도 허용한다.
    """
    speak_actions = [a for a in output.actions if a.type.lower() == "speak"]
    if not speak_actions:
        return output

    speak_text = speak_actions[0].value
    has_voice = context.get("has_voice_input", False)

    if speak_text == state.get("last_speak") and not has_voice:
        logger.warning(
            f"[ActionHook:RepeatSpeak] BLOCKED repeat speak "
            f"'{speak_text[:30]}...' (no new voice input)"
        )
        output.actions = []
        return output

    state["last_speak"] = speak_text
    return output


def consistency_hook(
    output: CortexOutputModel, context: dict, state: dict
) -> CortexOutputModel:
    """speak-move 의미 일관성을 검증한다. (warning only — 차단/주입 없음)

    Rule A: speak에 이동 의도 키워드가 있는데 move action이 없음
    Rule B: move action이 있는데 speak action이 없음 (사용자 미고지)
    """
    speak_actions = [a for a in output.actions if a.type.lower() == "speak"]
    move_actions = [a for a in output.actions if a.type.lower() == "move"]
    speak_text = " ".join(a.value for a in speak_actions)

    # Rule A
    if speak_text and not move_actions:
        matched = [kw for kw in MOVE_INTENT_KEYWORDS if kw in speak_text]
        if matched:
            logger.warning(
                f"[ActionHook:Consistency] WARNING speak contains move "
                f"intent '{matched[0]}' but no move action"
            )

    # Rule B
    if move_actions and not speak_actions:
        logger.warning(
            f"[ActionHook:Consistency] WARNING move({move_actions[0].value}) "
            f"without speak — user not informed"
        )

    return output


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

# 기본 Hook 실행 순서: 확실한 차단부터, 의미 검증은 마지막.
DEFAULT_HOOKS: List[HookFn] = [
    action_count_hook,
    no_voice_move_hook,
    repeat_speak_hook,
    consistency_hook,
]


class ActionHookChain:
    """등록된 Hook 함수를 순서대로 실행하는 러너.

    새로운 규칙을 추가하려면:
      1. 이 파일에 hook 함수를 작성 (시그니처: output, context, state -> output)
      2. DEFAULT_HOOKS 리스트에 추가
    """

    def __init__(self, hooks: Optional[List[HookFn]] = None):
        self.hooks = hooks if hooks is not None else list(DEFAULT_HOOKS)
        self._state: Dict[str, object] = {}
        hook_names = [h.__name__ for h in self.hooks]
        logger.info(
            f"ActionHookChain initialized with {len(self.hooks)} hooks: {hook_names}"
        )

    def validate(
        self, output: CortexOutputModel, context: dict
    ) -> CortexOutputModel:
        for hook in self.hooks:
            output = hook(output, context, self._state)
        return output

    def reset_state(self):
        """Config reload 시 호출하여 내부 상태를 초기화한다."""
        self._state.clear()
