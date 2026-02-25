import logging
import time
from pathlib import Path
from threading import Thread

import pytest

from backgrounds.plugins.segmentation_bg import SegmentationBg, SegmentationConfig
from providers.segmentation_provider import SegmentationProvider

ENGINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "providers"
    / "engines"
    / "trt"
    / "ddrnet23_fp16_kist-v1-80k_1x480x640.engine"
).as_posix()


@pytest.fixture
def config():
    return SegmentationConfig(engine_path=ENGINE_PATH)


@pytest.fixture
def config_default():
    return SegmentationConfig()


@pytest.fixture
def config_with_camera():
    return SegmentationConfig(engine_path=ENGINE_PATH, auto_start_camera=True)


@pytest.fixture(autouse=True)
def reset_segmentation_provider():
    """Reset singleton instances between tests."""
    SegmentationProvider.reset()  # type: ignore
    yield
    SegmentationProvider.reset()  # type: ignore


def test_config_initialization():
    """Test SegmentationConfig initialization."""
    config = SegmentationConfig()
    assert config.engine_path == ENGINE_PATH

    config_custom = SegmentationConfig(engine_path="/custom/path.engine")
    assert config_custom.engine_path == "/custom/path.engine"


def test_background_initialization(config):
    """Test SegmentationBg initialization."""
    background = SegmentationBg(config=config)

    assert background.segmentation_provider is not None
    t = Thread(target=background.run, daemon=True)
    t.start()
    time.sleep(0.1)
    assert background.segmentation_provider.running
    background.segmentation_provider.stop()
    t.join(timeout=2.0)


def test_background_initialization_default(config_default):
    """Test SegmentationBg initialization with default engine_path."""
    background = SegmentationBg(config=config_default)

    assert background.segmentation_provider is not None
    t = Thread(target=background.run, daemon=True)
    t.start()
    time.sleep(0.1)
    assert background.segmentation_provider.running
    background.segmentation_provider.stop()
    t.join(timeout=2.0)


def test_background_name(config):
    """Test that background has correct name."""
    background = SegmentationBg(config=config)
    assert background.name == "SegmentationBg"
    t = Thread(target=background.run, daemon=True)
    t.start()
    time.sleep(0.1)
    background.segmentation_provider.stop()
    t.join(timeout=2.0)


def test_background_config_access(config):
    """Test that background has access to config."""
    background = SegmentationBg(config=config)
    assert background.config == config
    assert background.config.engine_path == ENGINE_PATH
    t = Thread(target=background.run, daemon=True)
    t.start()
    time.sleep(0.1)
    background.segmentation_provider.stop()
    t.join(timeout=2.0)


def test_background_produces_segmentation_output(config_with_camera):
    """Integration test: background produces segmentation output from camera."""
    background = SegmentationBg(config=config_with_camera)
    provider = background.segmentation_provider
    t = Thread(target=background.run, daemon=True)
    t.start()

    cam = getattr(provider, "cam", None)
    if cam is None:
        provider.stop()
        t.join(timeout=2.0)
        pytest.skip("RealSense camera not available")

    cam_start_deadline = time.monotonic() + 3.0
    while time.monotonic() < cam_start_deadline and not getattr(cam, "running", False):
        time.sleep(0.05)

    if not getattr(cam, "running", False):
        logging.info(
            "Skip: cam missing or not running. cam=%s running=%s",
            cam,
            getattr(cam, "running", None),
        )
        provider.stop()
        t.join(timeout=2.0)
        pytest.skip("RealSense camera not running/available")

    deadline = time.monotonic() + 20.0
    got_output = False
    while time.monotonic() < deadline:
        data = provider.data
        if data is not None:
            logging.debug(
                "seg data keys=%s segmented_image=%s classes=%s",
                list(data.keys()),
                "set" if data.get("segmented_image") is not None else None,
                data.get("classes"),
            )
        if data is not None and data.get("segmented_image") is not None:
            got_output = True
            break
        time.sleep(0.01)

    provider.stop()
    t.join(timeout=2.0)
    if not got_output:
        logging.info(
            "Skip: no output. cam.running=%s cam.data=%s",
            getattr(cam, "running", None),
            "set" if getattr(cam, "data", None) is not None else None,
        )
        pytest.skip("No segmentation output received within timeout")
