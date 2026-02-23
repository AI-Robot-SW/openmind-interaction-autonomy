"""
Tests for UnitreeGo2Bg (Unitree Go2 Background).

Follows .cursor/skills/background-testing/SKILL.md:
- Patch UnitreeGo2Provider in backgrounds.plugins.unitree_go2_bg (no robot/SDK).
- Fixtures: config, config_with_ethernet, config_custom_topic.
- Test: config init, Background init (Provider args + start()), config derivation, name, config access, run(), init failure.

Run: uv run pytest tests/backgrounds/test_unitree_go2_bg.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.unitree_go2_bg import UnitreeGo2Bg, UnitreeGo2BgConfig


# ----- Fixtures -----


@pytest.fixture
def config():
    """Default config: no ethernet, default timeout, default state_topic."""
    return UnitreeGo2BgConfig()


@pytest.fixture
def config_with_ethernet():
    """Config with ethernet channel and custom timeout."""
    return UnitreeGo2BgConfig(unitree_ethernet="eth0", timeout=10.0)


@pytest.fixture
def config_custom_topic():
    """Config with custom state_topic."""
    return UnitreeGo2BgConfig(
        unitree_ethernet="eth0", 
        timeout=10.0, 
        state_topic="rt/custom_state"
    )


# ----- Config -----


def test_config_initialization():
    """Config class default and custom values."""
    cfg = UnitreeGo2BgConfig()
    assert cfg.unitree_ethernet is None
    assert cfg.timeout == 10.0  # Updated default
    assert cfg.state_topic == "rt/sportmodestate"

    cfg_custom = UnitreeGo2BgConfig(
        unitree_ethernet="eth0", 
        timeout=5.0,
        state_topic="rt/custom"
    )
    assert cfg_custom.unitree_ethernet == "eth0"
    assert cfg_custom.timeout == 5.0
    assert cfg_custom.state_topic == "rt/custom"


# ----- Initialization -----


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization(mock_provider_class, config):
    """Background constructs Provider with args from config; stores instance; calls start() once."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = UnitreeGo2Bg(config=config)

    mock_provider_class.assert_called_once_with(
        channel="", 
        timeout=10.0,
        state_topic="rt/sportmodestate"
    )
    assert background.unitree_go2_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization_with_ethernet(mock_provider_class, config_with_ethernet):
    """Background passes unitree_ethernet, timeout, and state_topic to Provider."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = UnitreeGo2Bg(config=config_with_ethernet)

    mock_provider_class.assert_called_once_with(
        channel="eth0", 
        timeout=10.0,
        state_topic="rt/sportmodestate"
    )
    assert background.unitree_go2_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization_with_custom_topic(mock_provider_class, config_custom_topic):
    """Background passes custom state_topic to Provider."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = UnitreeGo2Bg(config=config_custom_topic)

    mock_provider_class.assert_called_once_with(
        channel="eth0", 
        timeout=10.0,
        state_topic="rt/custom_state"
    )
    assert background.unitree_go2_provider is mock_provider_instance
    mock_provider_instance.start.assert_called_once()


# ----- Config derivation (optional/whitespace) -----


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_initialization_strips_ethernet_whitespace(mock_provider_class):
    """Provider is called with stripped unitree_ethernet."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance
    cfg = UnitreeGo2BgConfig(unitree_ethernet="  eth0  ", timeout=2.0)

    UnitreeGo2Bg(config=cfg)

    mock_provider_class.assert_called_once_with(
        channel="eth0", 
        timeout=2.0,
        state_topic="rt/sportmodestate"
    )


# ----- Name -----


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_name(mock_provider_class, config):
    """background.name equals class name."""
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config)
    assert background.name == "UnitreeGo2Bg"


# ----- Config access -----


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_background_config_access(mock_provider_class, config_with_ethernet):
    """background.config is injected config; attributes match."""
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config_with_ethernet)
    assert background.config is config_with_ethernet
    assert background.config.unitree_ethernet == "eth0"
    assert background.config.timeout == 10.0
    assert background.config.state_topic == "rt/sportmodestate"


# ----- run() -----


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
@patch("backgrounds.plugins.unitree_go2_bg.time.sleep")
def test_run_sleeps(mock_sleep, mock_provider_class, config):
    """run() calls time.sleep(60)."""
    mock_provider_class.return_value = MagicMock()
    background = UnitreeGo2Bg(config=config)

    background.run()

    mock_sleep.assert_called_once_with(60)


# ----- Init failure -----


@patch("backgrounds.plugins.unitree_go2_bg.UnitreeGo2Provider")
def test_init_raises_on_provider_failure(mock_provider_class, config):
    """__init__ propagates when Provider raises."""
    mock_provider_class.side_effect = RuntimeError("DDS init failed")

    with pytest.raises(RuntimeError, match="DDS init failed"):
        UnitreeGo2Bg(config=config)