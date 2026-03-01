import time
from unittest.mock import patch

import pytest
import numpy as np

from providers.pointcloud_provider import PointCloudProvider
from providers.segmentation_provider import SegmentationProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    PointCloudProvider.reset()  # type: ignore
    yield
    PointCloudProvider.reset()  # type: ignore


@pytest.fixture
def provider_params():
    return {
        "range_max": 3.0,
        "stride": 2,
        "sync_slop": 0.05,
    }


def test_initialization(provider_params):
    """Test PointCloudProvider initialization."""
    provider = PointCloudProvider(**provider_params)

    assert provider.range_max == provider_params["range_max"]
    assert provider.stride == provider_params["stride"]
    assert provider.sync_slop == provider_params["sync_slop"]
    assert provider._pointcloud_data is None
    assert provider._data is None
    assert not provider.running
    assert provider._thread is None


def test_singleton_pattern(provider_params):
    """Test that PointCloudProvider follows singleton pattern."""
    provider1 = PointCloudProvider(**provider_params)
    provider2 = PointCloudProvider(**provider_params)

    assert provider1 is provider2


def test_start(provider_params):
    """Test starting the provider."""
    provider = PointCloudProvider(**provider_params)
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    provider.stop()


def test_stop(provider_params):
    """Test stopping the provider."""
    provider = PointCloudProvider(**provider_params)
    provider.start()
    assert provider.running

    provider.stop()

    assert not provider.running
    time.sleep(0.1)
    if provider._thread:
        assert not provider._thread.is_alive()


def test_data_property(provider_params):
    """Test data property access."""
    provider = PointCloudProvider(**provider_params)

    assert provider.data is None

    test_data = {"pointcloud": None, "timestamp": time.time()}
    provider._data = test_data
    assert provider.data == test_data


class _DummyProvider:
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data


def _make_rs_data(ts: float, h: int = 2, w: int = 3):
    depth = np.full((h, w), 1.0, dtype=np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    camera_info = {
        "color": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}
    }
    return {
        "depth": {"image": depth, "encoding": "32FC1", "depth_scale": 0.001},
        "rgb": {"image": rgb, "encoding": "bgr8"},
        "camera_info": camera_info,
        "timestamp": ts,
    }


def _make_seg_data(ts: float, h: int = 2, w: int = 3):
    seg = np.zeros((h, w, 3), dtype=np.uint8)
    return {"segmented_image": seg, "timestamp": ts}


def test_update_data(provider_params):
    """Test _update_data with mocked providers."""
    provider = PointCloudProvider(**provider_params)
    ts = time.time()
    provider._realsense_provider = _DummyProvider(_make_rs_data(ts))
    provider._segmentation_provider = _DummyProvider(_make_seg_data(ts))

    provider._update_data()

    assert provider._data is not None
    assert "pointcloud" in provider._data
    assert "timestamp" in provider._data


def test_try_sync(provider_params):
    provider = PointCloudProvider(**provider_params)
    ts = time.time()
    provider._last_depth_data = {"image": np.ones((2, 2), dtype=np.float32), "timestamp": ts}
    provider._last_rgb_data = {"image": np.zeros((2, 2, 3), dtype=np.uint8), "timestamp": ts}
    provider._last_camera_info = {
        "camera_info": {"color": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}},
        "timestamp": ts,
    }

    with patch.object(provider, "_process_triplet") as mock_proc:
        provider._try_sync()
        assert mock_proc.called


def test_pc_gen(provider_params):
    provider = PointCloudProvider(**provider_params)
    ts = time.time()
    depth = {"image": np.full((2, 3), 1.0, dtype=np.float32), "encoding": "32FC1"}
    rgb = {"image": np.zeros((2, 3, 3), dtype=np.uint8), "encoding": "bgr8"}
    info = {"camera_info": {"color": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0}}}

    provider._process_triplet(depth, rgb, info)
    assert provider._pointcloud_data is not None
    assert "data" in provider._pointcloud_data
    assert provider._pointcloud_data["point_step"] == 20


def test_pc_hz(provider_params):
    provider = PointCloudProvider(**provider_params)
    t0 = time.monotonic()
    duration = 0.5
    updates = 0
    last_ts = None

    while (time.monotonic() - t0) < duration:
        ts = time.time()
        provider._realsense_provider = _DummyProvider(_make_rs_data(ts))
        provider._segmentation_provider = _DummyProvider(_make_seg_data(ts))
        provider._update_data()
        d = provider.data
        if d is not None and d.get("pointcloud") is not None:
            if last_ts != d["timestamp"]:
                updates += 1
                last_ts = d["timestamp"]

    hz = updates / max(duration, 1e-6)
    assert hz > 0.0


# @pytest.mark.integration
def test_pc_real_hz(provider_params):
    TARGET_HZ = 10
    MIN_HZ = 3
    DURATION = 3.0
    TRIALS = 3

    if hasattr(SegmentationProvider, "reset"):
        SegmentationProvider.reset()  # type: ignore

    seg = SegmentationProvider(auto_start_camera=True)
    seg.start()

    pc = PointCloudProvider(**provider_params)
    pc.start()

    t_deadline = time.monotonic() + 5.0
    last_ts = None
    while time.monotonic() < t_deadline:
        d = pc.data
        if d is not None and d.get("pointcloud") is not None:
            last_ts = d["timestamp"]
            break
        time.sleep(0.01)

    if last_ts is None:
        pc.stop()
        seg.stop()
        pytest.skip("No pointcloud outputs received")

    hz_list = []
    for _ in range(TRIALS):
        t0 = time.monotonic()
        changes = 0
        while (time.monotonic() - t0) < DURATION:
            d = pc.data
            if d is None or d.get("pointcloud") is None:
                time.sleep(0.001)
                continue
            ts = d["timestamp"]
            if ts != last_ts:
                changes += 1
                last_ts = ts
            time.sleep(0.001)
        hz_list.append(changes / DURATION)

    pc.stop()
    seg.stop()

    for i, hz in enumerate(hz_list, start=1):
        print(f"Measured real pointcloud Hz (trial {i}/{TRIALS}): {hz:.2f} (target {TARGET_HZ})")
    assert all(hz >= MIN_HZ for hz in hz_list), f"Measured Hz too low: {min(hz_list):.1f}"


def test_error_handling_in_run(provider_params):
    """Test error handling in _run method."""
    provider = PointCloudProvider(**provider_params)

    with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
        provider.start()
        time.sleep(0.1)
        provider.stop()

    assert not provider.running
