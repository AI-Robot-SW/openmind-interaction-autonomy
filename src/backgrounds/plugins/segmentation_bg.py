import logging
import time
from pathlib import Path
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers.segmentation_provider import SegmentationProvider


class SegmentationConfig(BackgroundConfig):
    """
    Configuration for Segmentation Background.

    Parameters
    ----------
    engine_path : Optional[str]
        Path to TensorRT engine file (default: "").
    auto_start_camera : bool
        Whether to start the camera automatically (default: False).
    """

    engine_path: Optional[str] = Field(
        default=str(
            Path(__file__).resolve().parents[2]
            / "providers"
            / "engines"
            / "trt"
            / "ddrnet23_fp16_kist-v1-80k_1x480x640.engine"
        ),
        description="Path to TensorRT engine file",
    )
    auto_start_camera: bool = Field(
        default=False, description="Start camera automatically"
    )


class SegmentationBg(Background[SegmentationConfig]):
    """
    Segmentation Background.

    Initializes and starts the SegmentationProvider in the background.
    """

    def __init__(self, config: SegmentationConfig):
        """
        Initialize the Segmentation Background.

        Parameters
        ----------
        config : SegmentationConfig
            Configuration for the background task.
        """
        super().__init__(config)

        engine_path = self.config.engine_path or ""

        # Initialize Provider (singleton, so same instance shared)
        self.segmentation_provider = SegmentationProvider(
            engine_path=engine_path,
            auto_start_camera=bool(self.config.auto_start_camera),
        )
        logging.info(
            f"Segmentation Provider initialized in background (engine_path: {engine_path})"
        )

    def run(self) -> None:
        """
        Start the SegmentationProvider.

        The provider manages its own internal loop; this method just ensures it is running.
        """
        if self.segmentation_provider.running:
            return
        try:
            self.segmentation_provider.start()
            logging.info("Segmentation Provider started by background run()")
            while self.segmentation_provider.running:
                time.sleep(1.0)
        finally:
            self.segmentation_provider.stop()
            logging.info("Segmentation Provider stopped by background run()")
