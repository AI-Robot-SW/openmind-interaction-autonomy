import time
from unittest.mock import patch

import pytest

from providers.realsense_camera_provider import RealSenseCameraProvider


#realsense test requires actual device
def has_realsense():
    try:
        import pyrealsense2 as rs
        return len(rs.context().query_devices()) > 0
    except Exception:
        return False


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    RealSenseCameraProvider.reset()  # type: ignore
    yield
    RealSenseCameraProvider.reset()  # type: ignore


@pytest.fixture
def camera_index():
    return 0


def test_initialization(camera_index):
    """Test RealSenseCameraProvider initialization."""
    provider = RealSenseCameraProvider(camera_index=camera_index)

    assert provider.camera_index == camera_index
    assert provider._rgb_data is None
    assert provider._depth_data is None
    assert provider._camera_info is None
    assert provider._data is None
    assert not provider.running
    assert provider._thread is None


def test_singleton_pattern(camera_index):
    """Test that RealSenseCameraProvider follows singleton pattern."""
    provider1 = RealSenseCameraProvider(camera_index=camera_index)
    provider2 = RealSenseCameraProvider(camera_index=camera_index)

    assert provider1 is provider2


@pytest.mark.skipif(not has_realsense(), reason="No RealSense device attached")
def test_start(camera_index):
    """Test starting the provider."""
    provider = RealSenseCameraProvider(camera_index=camera_index)
    provider.start()

    assert provider.running
    assert provider._thread is not None
    assert provider._thread.is_alive()

    provider.stop()
    assert not provider.running
    assert provider._thread is None or not provider._thread.is_alive()


@pytest.mark.skipif(not has_realsense(), reason="No RealSense device attached")
def test_stop(camera_index):
    """Test stopping the provider."""
    provider = RealSenseCameraProvider(camera_index=camera_index)
    provider.start()
    assert provider.running

    provider.stop()
    assert not provider.running
    assert provider._thread is None or not provider._thread.is_alive()


def test_data_property(camera_index):
    """Test data property access."""
    provider = RealSenseCameraProvider(camera_index=camera_index)

    # Initially should be None
    assert provider.data is None

    # After setting _data, should return it
    test_data = {"rgb": None, "depth": None, "camera_info": None, "timestamp": time.time()}
    provider._data = test_data
    assert provider.data == test_data


# def test_update_data_placeholder(camera_index):
#     """Test _update_data method (placeholder implementation)."""
#     provider = RealSenseCameraProvider(camera_index=camera_index)
#     provider._update_data()

#     # Should update _data with structure
#     assert provider._data is not None
#     assert "rgb" in provider._data
#     assert "depth" in provider._data
#     assert "camera_info" in provider._data
#     assert "timestamp" in provider._data

@pytest.mark.skipif(not has_realsense(), reason="No RealSense device attached")
def test_update_data_fills_data(camera_index):
    provider = RealSenseCameraProvider(camera_index=camera_index)
    provider.start()
    time.sleep(0.2)

    d = provider.data
    assert d is not None
    assert "rgb" in d and d["rgb"] is not None
    assert "depth" in d and d["depth"] is not None
    assert "camera_info" in d
    assert "timestamp" in d

    provider.stop()


# def test_error_handling_in_run(camera_index):
#     """Test error handling in _run method."""
#     provider = RealSenseCameraProvider(camera_index=camera_index)

#     # Simulate error in _update_data
#     with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
#         provider.start()
#         time.sleep(0.1)  # Let thread run briefly
#         provider.stop()

#     # Should handle error gracefully
#     assert not provider.running

def test_error_handling_in_run(camera_index):
    provider = RealSenseCameraProvider(camera_index=camera_index)

    with patch.object(provider, "_start_sdk", return_value=None):
        with patch.object(provider, "_update_data", side_effect=Exception("Test error")):
            provider.start()
            time.sleep(0.1)
            provider.stop()

    assert not provider.running


# @pytest.mark.integration
def test_realsense_camera_provider_fps_30():
    TARGET_FPS = 30
    MIN_FPS = 25          
    DURATION = 3.0      

    p = RealSenseCameraProvider(width=640, height=480, fps=TARGET_FPS)
    p.start()

    # 첫 프레임 나올 때까지 대기
    t_deadline = time.monotonic() + 2.0
    last_ts = None
    while time.monotonic() < t_deadline:
        d = p.data
        if d is not None:
            last_ts = d["timestamp"]
            break
        time.sleep(0.01)

    if last_ts is None:
        p.stop()
        pytest.skip("No frames received (camera not connected or pipeline not running)")

    # FPS 측정
    t0 = time.monotonic()
    changes = 0
    while (time.monotonic() - t0) < DURATION:
        d = p.data
        if d is None:
            time.sleep(0.001)
            continue
        ts = d["timestamp"]
        if ts != last_ts:
            changes += 1
            last_ts = ts
        time.sleep(0.001)

    p.stop()

    measured_fps = changes / DURATION
    print(f"Measured FPS: {measured_fps:.2f}")
    assert measured_fps >= MIN_FPS, f"Measured FPS too low: {measured_fps:.1f}"
