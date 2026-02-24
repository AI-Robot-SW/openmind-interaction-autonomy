"""
Tests for BEVOccupancyGridBg (BEV Occupancy Grid Background).

Follows .cursor/skills/background-testing/SKILL.md:
- Patch BEVOccupancyGridProvider in backgrounds.plugins.bev_occupancy_grid_bg (no HW/CUDA).
- Fixtures: config, config_default, config_with_custom.
- Test: config init, Background init (Provider args + start()), name, config access, run(), init failure.

Run: uv run pytest tests/backgrounds/test_bev_occupancy_grid_bg.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.bev_occupancy_grid_bg import (
    BEVOccupancyGridBg,
    BEVOccupancyGridConfig,
)


# ----- Fixtures -----


@pytest.fixture
def config():
    """Config with explicit default-like values."""
    return BEVOccupancyGridConfig(
        res=0.05,
        width=50,
        height=60,
        origin_x=0.0,
        origin_y=-1.5,
        dx=-0.34,
        dy=0.0,
        closing_kernel_size=1,
    )


@pytest.fixture
def config_default():
    """Config using class defaults (no args)."""
    return BEVOccupancyGridConfig()


@pytest.fixture
def config_with_custom():
    """Config with custom values for derivation tests."""
    return BEVOccupancyGridConfig(
        res=0.1,
        width=100,
        height=120,
        origin_x=1.0,
        origin_y=-2.0,
        dx=-0.5,
        dy=0.1,
        closing_kernel_size=2,
    )


# ----- Config -----


def test_config_initialization():
    """Config class default and custom values."""
    cfg = BEVOccupancyGridConfig()
    assert cfg.res == 0.05
    assert cfg.width == 50
    assert cfg.height == 60
    assert cfg.origin_x == 0.0
    assert cfg.origin_y == -1.5
    assert cfg.dx == -0.34
    assert cfg.dy == 0.0
    assert cfg.closing_kernel_size == 1

    cfg_custom = BEVOccupancyGridConfig(
        res=0.1,
        width=100,
        height=120,
        origin_x=1.0,
        origin_y=-2.0,
        dx=-0.5,
        dy=0.1,
        closing_kernel_size=2,
    )
    assert cfg_custom.res == 0.1
    assert cfg_custom.width == 100
    assert cfg_custom.height == 120
    assert cfg_custom.origin_x == 1.0
    assert cfg_custom.origin_y == -2.0
    assert cfg_custom.dx == -0.5
    assert cfg_custom.dy == 0.1
    assert cfg_custom.closing_kernel_size == 2


# ----- Initialization -----


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_background_initialization(mock_provider_class, config):
    """Background constructs Provider with args from config; stores instance; calls start() once."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = BEVOccupancyGridBg(config=config)

    mock_provider_class.assert_called_once_with(
        res=0.05,
        width=50,
        height=60,
        origin_x=0.0,
        origin_y=-1.5,
        dx=-0.34,
        dy=0.0,
        closing_kernel_size=1,
    )
    assert background.bev_occupancy_grid_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_background_initialization_default(mock_provider_class, config_default):
    """Background passes default config values to Provider."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = BEVOccupancyGridBg(config=config_default)

    mock_provider_class.assert_called_once_with(
        res=0.05,
        width=50,
        height=60,
        origin_x=0.0,
        origin_y=-1.5,
        dx=-0.34,
        dy=0.0,
        closing_kernel_size=1,
    )
    assert background.bev_occupancy_grid_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_background_initialization_custom(mock_provider_class, config_with_custom):
    """Background passes custom config values to Provider."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = BEVOccupancyGridBg(config=config_with_custom)

    mock_provider_class.assert_called_once_with(
        res=0.1,
        width=100,
        height=120,
        origin_x=1.0,
        origin_y=-2.0,
        dx=-0.5,
        dy=0.1,
        closing_kernel_size=2,
    )
    assert background.bev_occupancy_grid_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


# ----- Name -----


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_background_name(mock_provider_class, config):
    """background.name equals class name."""
    mock_provider_class.return_value = MagicMock()
    background = BEVOccupancyGridBg(config=config)
    assert background.name == "BEVOccupancyGridBg"


# ----- Config access -----


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_background_config_access(mock_provider_class, config):
    """background.config is injected config; attributes match."""
    mock_provider_class.return_value = MagicMock()
    background = BEVOccupancyGridBg(config=config)
    assert background.config is config
    assert background.config.res == 0.05
    assert background.config.width == 50
    assert background.config.height == 60


# ----- run() -----


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
@patch("backgrounds.plugins.bev_occupancy_grid_bg.time.sleep")
def test_run_sleeps(mock_sleep, mock_provider_class, config):
    """run() calls time.sleep(60)."""
    mock_provider_class.return_value = MagicMock()
    background = BEVOccupancyGridBg(config=config)

    background.run()

    mock_sleep.assert_called_once_with(60)


# ----- Init failure -----


@patch("backgrounds.plugins.bev_occupancy_grid_bg.BEVOccupancyGridProvider")
def test_init_raises_on_provider_failure(mock_provider_class, config):
    """__init__ propagates when Provider raises."""
    mock_provider_class.side_effect = RuntimeError("Provider init failed")

    with pytest.raises(RuntimeError, match="Provider init failed"):
        BEVOccupancyGridBg(config=config)
