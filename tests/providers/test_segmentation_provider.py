import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from providers.segmentation_provider import SegmentationProvider


ENGINE_PATH = Path(
    "/home/nvidia/pedestrian-companion-robot/src/providers/engines/trt/ddrnet23_fp16_kist-v1-80k_1x480x640.engine"
).as_posix()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    SegmentationProvider.reset()  # type: ignore
    yield
    SegmentationProvider.reset()  # type: ignore

@pytest.fixture
def engine_path():
    return ENGINE_PATH


def test_initialization(engine_path):
    """Test SegmentationProvider initialization."""
    provider = SegmentationProvider(engine_path=engine_path)

    assert provider.engine_path == engine_path
    assert provider._segmented_image is None
    assert provider._classes is None
    assert provider._data is None
    assert not provider.running
    assert provider._thread is None


def test_singleton_pattern(engine_path):
    """Test that SegmentationProvider follows singleton pattern."""
    provider1 = SegmentationProvider(engine_path=engine_path)
    provider2 = SegmentationProvider(engine_path=engine_path)

    assert provider1 is provider2


def test_start(engine_path):
    """Test starting the provider."""
    provider = SegmentationProvider(engine_path=engine_path)
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    # Cleanup
    provider.stop()


def test_stop(engine_path):
    """Test stopping the provider."""
    provider = SegmentationProvider(engine_path=engine_path)
    provider.start()
    assert provider.running

    provider.stop()

    assert not provider.running
    time.sleep(0.1)
    if provider._thread:
        assert not provider._thread.is_alive()


def test_data_property(engine_path):
    """Test data property access."""
    provider = SegmentationProvider(engine_path=engine_path)

    assert provider.data is None

    test_data = {
        "segmented_image": None,
        "classes": None,
        "timestamp": time.time(),
    }
    provider._data = test_data
    assert provider.data == test_data


def test_update_data_placeholder(engine_path):
    """Test _update_data method (placeholder implementation)."""
    provider = SegmentationProvider(engine_path=engine_path)
    provider._update_data()

    assert provider._data is not None
    assert "segmented_image" in provider._data
    assert "classes" in provider._data
    assert "timestamp" in provider._data


def test_error_handling_in_run(engine_path):
    """Test error handling in _run method."""
    provider = SegmentationProvider(engine_path=engine_path)

    with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
        provider.start()
        time.sleep(0.1)
        provider.stop()

    assert not provider.running


def test_process_frame_logs_engine_output(engine_path, caplog):
    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.zeros((1, 3, 2, 3))}

    class DummyEngine:
        def infer(self, pixel_values):
            return [np.zeros((1, 5, 2, 3), dtype=np.float32)]

    provider = SegmentationProvider(engine_path=engine_path)
    provider._processor = DummyProcessor()
    provider._engine = DummyEngine()

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    caplog.set_level(logging.DEBUG)
    out = provider.process_frame(frame)

    assert out["predicted_map"].shape == (2, 3)
    assert "Segmentation TRT output[0] shape=" in caplog.text


# @pytest.mark.integration
def test_segmentation_provider_fps_30(engine_path):
    TARGET_FPS = 30
    MIN_FPS = 15
    DURATION = 3.0

    provider = SegmentationProvider(
        engine_path=engine_path,
        auto_start_camera=True,
    )
    provider.start()

    # Wait for first update
    t_deadline = time.monotonic() + 2.0
    last_ts = None
    while time.monotonic() < t_deadline:
        d = provider.data
        if d is not None and d.get("segmented_image") is not None:
            last_ts = d["timestamp"]
            break
        time.sleep(0.01)

    if last_ts is None:
        provider.stop()
        pytest.skip("No segmentation outputs received")

    # Measure update rate using timestamp changes
    t0 = time.monotonic()
    changes = 0
    while (time.monotonic() - t0) < DURATION:
        d = provider.data
        if d is None or d.get("segmented_image") is None:
            time.sleep(0.001)
            continue
        ts = d["timestamp"]
        if ts != last_ts:
            changes += 1
            last_ts = ts
        time.sleep(0.001)

    provider.stop()

    measured_fps = changes / DURATION
    print(f"Measured segmentation FPS: {measured_fps:.2f} (target {TARGET_FPS})")
    assert measured_fps >= MIN_FPS, f"Measured FPS too low: {measured_fps:.1f}"
