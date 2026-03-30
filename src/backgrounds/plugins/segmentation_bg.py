# segmentation_bg.py

import time
import logging
import threading

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.segmentation_provider import SegmentationProvider


class SegmentationConfig(BackgroundConfig):
    pass


class SegmentationBg(Background[SegmentationConfig]):
    """
    Segmentation Background.

    Initializes and starts the SegmentationProvider in the background.
    """

    def __init__(self, config: SegmentationConfig):
        super().__init__(config)

        self.segmentation_provider = SegmentationProvider()
        self.segmentation_provider.start()
        logging.info("Segmentation Provider initialized and started in background")

    def run(self) -> None:
        evt = self._orchestrator_stop_event if self._orchestrator_stop_event is not None else threading.Event()
        if evt.is_set():
            self.segmentation_provider.stop()
            logging.info("Segmentation Provider stopped")
            return
        time.sleep(1.0)
