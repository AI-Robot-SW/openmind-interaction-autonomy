# debug_viz_bg.py

import logging
import threading
import time

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.debug_viz_provider import DebugVizProvider


class DebugVizBgConfig(BackgroundConfig):
    panel_w: int = Field(default=320, description="Width of each panel in pixels")
    panel_h: int = Field(default=240, description="Height of each panel in pixels")
    target_fps: float = Field(default=15.0, description="Visualization target FPS")


class DebugVizBg(Background[DebugVizBgConfig]):
    """
    HW 디버그용 실시간 시각화 Background.

    DebugVizProvider를 시작하여 6-패널 OpenCV 창을 표시한다.
    PassthroughLLM과 함께 사용하면 LLM 없이 하드웨어 상태를 시각적으로 디버깅할 수 있다.
    """

    def __init__(self, config: DebugVizBgConfig):
        super().__init__(config)
        self.debug_viz_provider = DebugVizProvider(
            panel_w=config.panel_w,
            panel_h=config.panel_h,
            target_fps=config.target_fps,
        )
        self.debug_viz_provider.start()
        logging.info(
            "DebugVizProvider initialized and started in background "
            f"(panel={config.panel_w}×{config.panel_h}, fps={config.target_fps})"
        )

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.debug_viz_provider.stop()
            logging.info("DebugVizProvider stopped")
            return
        time.sleep(1.0)
