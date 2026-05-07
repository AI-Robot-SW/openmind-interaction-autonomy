# kf_position_bg.py

import time
import logging
import threading

from backgrounds.base import Background, BackgroundConfig
from providers.kf_position_provider import KfPositionProvider


class KfPositionBgConfig(BackgroundConfig):
    pass


class KfPositionBg(Background[KfPositionBgConfig]):
    """
    KF Position Provider Background.

    KfPositionProvider를 시작하고 종료한다.
    """

    def __init__(self, config: KfPositionBgConfig):
        super().__init__(config)

        self.kf_position_provider = KfPositionProvider()
        self.kf_position_provider.start()

        logging.info("KfPositionBg started")

    def run(self) -> None:
        evt = (
            self._orchestrator_stop_event
            if self._orchestrator_stop_event is not None
            else threading.Event()
        )
        if evt.is_set():
            self.kf_position_provider.stop()
            logging.info("KfPositionBg stopped")
            return
        time.sleep(1.0)
