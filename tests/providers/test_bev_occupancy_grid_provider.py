"""
Pytest tests for BEVOccupancyGridProvider.

Follows provider-testing SKILL: singleton reset, mock external deps (PointCloudProvider).

Run modes (from project root):

  # 1) Full: all tests (no_pycuda + cuda). On GPU-less machine, cuda tests are skipped.
  uv run pytest tests/providers/test_bev_occupancy_grid_provider.py -v

  # 2) CUDA unnecessary only (no GPU required).
  uv run pytest tests/providers/test_bev_occupancy_grid_provider.py -m no_pycuda -v

  # 3) CUDA required only (need GPU + pycuda). Skip on GPU-less machine.
  uv run pytest tests/providers/test_bev_occupancy_grid_provider.py -m cuda -v
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from providers.bev_occupancy_grid_provider import BEVOccupancyGridProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    BEVOccupancyGridProvider.reset()  # type: ignore
    yield
    BEVOccupancyGridProvider.reset()  # type: ignore


@pytest.fixture
def provider_params():
    """Constructor params for BEVOccupancyGridProvider."""
    return {
        "res": 0.05,
        "width": 50,
        "height": 60,
        "origin_x": 0.0,
        "origin_y": -1.5,
        "dx": -0.34,
        "dy": 0.0,
        "closing_kernel_size": 1,
    }


def _make_pointcloud_buffer(n_points: int, point_step: int = 20):
    """Build raw buffer (A안): x,y,z at 0:4,4:8,8:12, rgb at 12:16 (float32 packed)."""
    buf = np.zeros((n_points, point_step), dtype=np.uint8)
    for i in range(n_points):
        buf[i, 0:4] = np.frombuffer(np.float32(1.0 + i).tobytes(), dtype=np.uint8)
        buf[i, 4:8] = np.frombuffer(np.float32(2.0).tobytes(), dtype=np.uint8)
        buf[i, 8:12] = np.frombuffer(np.float32(3.0).tobytes(), dtype=np.uint8)
        rgb_int = (255 << 16) | (0 << 8) | 0
        buf[i, 12:16] = np.frombuffer(
            np.array([rgb_int], dtype=np.uint32).view(np.float32).tobytes(),
            dtype=np.uint8,
        )
    return buf


# --- Initialization ---


@pytest.mark.no_pycuda
class TestBEVOccupancyGridProviderInitialization:
    """Initialization and internal state."""

    def test_default_initialization(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        assert provider.res == provider_params["res"]
        assert provider.width == provider_params["width"]
        assert provider.height == provider_params["height"]
        assert provider.origin_x == provider_params["origin_x"]
        assert provider.origin_y == provider_params["origin_y"]
        assert provider.dx == provider_params["dx"]
        assert provider.dy == provider_params["dy"]
        assert provider.closing_kernel_size == provider_params["closing_kernel_size"]
        assert provider._bev_image is None
        assert provider._occupancy_grid is None
        assert provider._data is None
        assert provider.running is False
        assert provider._thread is None
        assert provider._closing_kernel.shape == (1, 1)

    def test_singleton_pattern(self, provider_params):
        p1 = BEVOccupancyGridProvider(**provider_params)
        p2 = BEVOccupancyGridProvider(
            res=0.1, width=80, height=80
        )
        assert p1 is p2


# --- Lifecycle ---


@pytest.mark.no_pycuda
class TestBEVOccupancyGridProviderLifecycle:
    """start(), stop(), idempotent start."""

    def test_start_starts_thread(self, provider_params):
        with patch("providers.bev_occupancy_grid_provider.PointCloudProvider") as mock_pc:
            mock_pc.return_value.data = None
            provider = BEVOccupancyGridProvider(**provider_params)
            provider.start()
            assert provider.running is True
            assert provider._thread is not None
            assert provider._thread.is_alive()
            provider.stop()

    def test_stop_clears_state(self, provider_params):
        with patch("providers.bev_occupancy_grid_provider.PointCloudProvider") as mock_pc:
            mock_pc.return_value.data = None
            provider = BEVOccupancyGridProvider(**provider_params)
            provider.start()
            provider.stop()
            assert provider.running is False
            assert provider._thread is None
            assert provider._cuda_kernel is None
            assert provider._cuda_mod is None
        time.sleep(0.1)

    def test_start_idempotent(self, provider_params):
        with patch("providers.bev_occupancy_grid_provider.PointCloudProvider") as mock_pc:
            mock_pc.return_value.data = None
            provider = BEVOccupancyGridProvider(**provider_params)
            provider.start()
            t1 = provider._thread
            provider.start()
            assert provider._thread is t1
            provider.stop()


# --- Data property ---


@pytest.mark.no_pycuda
class TestBEVOccupancyGridProviderData:
    """data property before/after start and when set."""

    def test_data_initial_none(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        assert provider.data is None

    def test_data_returns_internal_data(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        fake = {"bev_image": None, "occupancy_grid": None, "timestamp": time.time()}
        provider._data = fake
        assert provider.data is fake


# --- _parse_pointcloud (A안 only) ---


@pytest.mark.no_pycuda
class TestBEVOccupancyGridProviderParsePointcloud:
    """_parse_pointcloud: valid A안 and error paths."""

    def test_parse_valid_buffer_bytes(self, provider_params):
        buf = _make_pointcloud_buffer(2, 20)
        pc = {"data": buf.tobytes(), "point_step": 20}
        provider = BEVOccupancyGridProvider(**provider_params)
        out = provider._parse_pointcloud(pc)
        assert out is not None
        x, y, z, r, g, b = out
        assert len(x) == 2
        assert len(y) == 2
        np.testing.assert_array_almost_equal(x, [1.0, 2.0])
        np.testing.assert_array_almost_equal(y, [2.0, 2.0])
        np.testing.assert_array_almost_equal(z, [3.0, 3.0])
        assert r[0] == 255 and g[0] == 0 and b[0] == 0

    def test_parse_valid_buffer_ndarray(self, provider_params):
        buf = _make_pointcloud_buffer(1, 20)
        pc = {"data": buf.ravel(), "point_step": 20}
        provider = BEVOccupancyGridProvider(**provider_params)
        out = provider._parse_pointcloud(pc)
        assert out is not None
        x, y, z, r, g, b = out
        assert len(x) == 1

    def test_parse_missing_data_returns_none(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        assert provider._parse_pointcloud({"point_step": 20}) is None
        assert provider._parse_pointcloud({"data": b""}) is None
        assert provider._parse_pointcloud({}) is None

    def test_parse_point_step_too_small_returns_none(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        pc = {"data": np.zeros(20, dtype=np.uint8).tobytes(), "point_step": 10}
        assert provider._parse_pointcloud(pc) is None

    def test_parse_empty_buffer_returns_none(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        pc = {"data": np.zeros(0, dtype=np.uint8).tobytes(), "point_step": 20}
        assert provider._parse_pointcloud(pc) is None


# --- _build_occupancy_grid ---


@pytest.mark.no_pycuda
class TestBEVOccupancyGridProviderBuildOccupancyGrid:
    """_build_occupancy_grid: shape, keys, values."""

    def test_build_occupancy_grid_returns_dict(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        n = 10
        x = np.zeros(n, dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        z = np.zeros(n, dtype=np.float32)
        r = np.zeros(n, dtype=np.uint8)
        g = np.full(n, 150, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)
        out = provider._build_occupancy_grid(x, y, z, r, g, b)
        assert out is not None
        assert out["resolution"] == provider.res
        assert out["width"] == provider.width
        assert out["height"] == provider.height
        assert out["origin_x"] == provider.origin_x
        assert out["origin_y"] == provider.origin_y
        assert "data" in out
        assert out["data"].shape == (provider.height, provider.width)
        assert out["data"].dtype == np.int8

    def test_build_occupancy_grid_values_in_range(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        n = 6
        x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        z = np.full(n, 1.0, dtype=np.float32)
        r = np.array([255, 0, 0, 255, 200, 0], dtype=np.uint8)
        g = np.array([0, 150, 0, 200, 200, 0], dtype=np.uint8)
        b = np.array([0, 0, 0, 0, 200, 255], dtype=np.uint8)
        out = provider._build_occupancy_grid(x, y, z, r, g, b)
        assert out is not None
        data = out["data"]
        assert np.isin(data, [0, 70, 88, 100]).all()
        assert 88 in data


# --- Error handling ---


@pytest.mark.no_pycuda
class TestBEVOccupancyGridProviderErrorHandling:
    """Error in _run: exception leaves provider stoppable and state cleared."""

    def test_run_exception_clears_data_and_stop_succeeds(self, provider_params):
        provider = BEVOccupancyGridProvider(**provider_params)
        with patch("providers.bev_occupancy_grid_provider.PointCloudProvider") as mock_pc:
            mock_pc.return_value.data = None
            provider.start()
            time.sleep(0.05)
            provider._data = {"bev_image": None, "occupancy_grid": None, "timestamp": 0}
            provider.stop()
        assert provider._data is None or not provider.running
        assert provider._thread is None

    def test_pointcloud_provider_raises_run_continues(self, provider_params, caplog):
        mock_pc = MagicMock()
        mock_pc.return_value.data = None
        with patch("providers.bev_occupancy_grid_provider.PointCloudProvider", mock_pc):
            mock_pc.return_value.data = None
            provider = BEVOccupancyGridProvider(**provider_params)
            provider.start()
            time.sleep(0.08)
            provider.stop()
        assert not provider.running


# --- CUDA kernel (requires GPU + pycuda) ---


@pytest.mark.cuda
class TestBEVOccupancyGridProviderCuda:
    """Tests that require pycuda/CUDA (run on machine with NVIDIA GPU)."""

    def test_run_bev_kernel_returns_shape(self, provider_params):
        """_run_bev_kernel returns (height, width, 3) BGR uint8 when pycuda available."""
        pytest.importorskip("pycuda")
        provider = BEVOccupancyGridProvider(**provider_params)
        if provider._cuda_kernel is None:
            pytest.skip("pycuda loaded but CUDA kernel not available (e.g. no cuda.h)")
        n = 10
        x = np.zeros(n, dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        z = np.zeros(n, dtype=np.float32)
        r = np.zeros(n, dtype=np.uint8)
        g = np.zeros(n, dtype=np.uint8)
        b = np.zeros(n, dtype=np.uint8)
        out = provider._run_bev_kernel(x, y, z, r, g, b, n)
        assert out is not None
        assert out.shape == (provider.height, provider.width, 3)
        assert out.dtype == np.uint8
