# path_follow_bg.py

import time
import logging
import threading

from backgrounds.base import Background, BackgroundConfig
from providers.path_follow_provider import PathFollowProvider


class PathFollowBgConfig(BackgroundConfig):
    pass


class PathFollowBg(Background[PathFollowBgConfig]):
    """
    Path Follow Provider Background.

    PathFollowProvider를 시작하고 종료한다.
    """

    def __init__(self, config: PathFollowBgConfig):
        super().__init__(config)

        self.path_follow_provider = PathFollowProvider()
        self.path_follow_provider.start()

        logging.info("PathFollowBg started")

    def run(self) -> None:
        evt = (
            self._orchestrator_stop_event
            if self._orchestrator_stop_event is not None
            else threading.Event()
        )
        if evt.is_set():
            self.path_follow_provider.stop()
            logging.info("PathFollowBg stopped")
            return
        time.sleep(1.0)
