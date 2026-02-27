"""
Tests for NavigationBg.

Follows .cursor/skills/background-testing/SKILL.md:
- Patch GnssRouteProvider, DwaRouteProvider, NavigationProvider in navigation_bg module.
- Fixtures: config (default), config_custom.
- Tests: config init, Background init (Provider args), name, config access, run(), init failure.

Run: uv run pytest tests/backgrounds/test_navigation_bg.py -v
"""

import math
from unittest.mock import MagicMock, call, patch

import pytest

from backgrounds.plugins.navigation_bg import NavigationBg, NavigationBgConfig


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def config():
    """Default config."""
    return NavigationBgConfig()


@pytest.fixture
def config_custom():
    """Custom config with waypoints and tweaked params."""
    return NavigationBgConfig(
        waypoints=[(37.6, 127.0), (37.7, 127.1)],
        gnss_reach_tol_m=3.0,
        gnss_max_vx=0.6,
        dwa_vx_fixed=0.5,
        dwa_v_max=0.7,
        dwa_theta_turn_deg=30.0,
        monitor_rate_hz=10.0,
    )


# ==============================================================================
# Config
# ==============================================================================

class TestConfig:
    def test_default_values(self):
        cfg = NavigationBgConfig()
        assert cfg.waypoints == []
        assert cfg.gnss_reach_tol_m == pytest.approx(5.0)
        assert cfg.gnss_max_vx == pytest.approx(0.8)
        assert cfg.gnss_max_vyaw == pytest.approx(math.radians(45))
        assert cfg.dwa_vx_fixed == pytest.approx(0.8)
        assert cfg.dwa_v_max == pytest.approx(0.9)
        assert cfg.dwa_theta_turn_deg == pytest.approx(40.0)
        assert cfg.dwa_safety_slowdown is True
        assert cfg.dwa_enable_turn_in_place is True
        assert cfg.dwa_allow_backward is False
        assert cfg.dwa_unknown_is_obstacle is False
        assert cfg.dwa_dist_method == "bfs_cuda"
        assert cfg.dwa_control_rate_hz == pytest.approx(10.0)
        assert cfg.dwa_log_csv_path is None
        assert cfg.monitor_rate_hz == pytest.approx(20.0)

    def test_custom_values(self, config_custom):
        assert config_custom.waypoints == [(37.6, 127.0), (37.7, 127.1)]
        assert config_custom.gnss_reach_tol_m == pytest.approx(3.0)
        assert config_custom.dwa_vx_fixed == pytest.approx(0.5)
        assert config_custom.monitor_rate_hz == pytest.approx(10.0)


# ==============================================================================
# Initialization
# ==============================================================================

@patch("backgrounds.plugins.navigation_bg.NavigationProvider")
@patch("backgrounds.plugins.navigation_bg.DwaRouteProvider")
@patch("backgrounds.plugins.navigation_bg.GnssRouteProvider")
class TestInitialization:
    def test_gnss_constructed_with_config(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        mock_nav_cls.return_value = MagicMock()

        NavigationBg(config=config)

        mock_gnss_cls.assert_called_once_with(
            waypoints=config.waypoints,
            reach_tol_m=config.gnss_reach_tol_m,
            max_vx=config.gnss_max_vx,
            max_vyaw=config.gnss_max_vyaw,
        )

    def test_dwa_constructed_with_gnss_instance(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        gnss_instance = MagicMock()
        mock_gnss_cls.return_value = gnss_instance
        mock_dwa_cls.return_value = MagicMock()
        mock_nav_cls.return_value = MagicMock()

        NavigationBg(config=config)

        call_kwargs = mock_dwa_cls.call_args.kwargs
        assert call_kwargs["gnss_route_provider"] is gnss_instance

    def test_dwa_constructed_with_config_params(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config_custom):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        mock_nav_cls.return_value = MagicMock()

        NavigationBg(config=config_custom)

        kw = mock_dwa_cls.call_args.kwargs
        assert kw["vx_fixed"] == pytest.approx(0.5)
        assert kw["v_max"] == pytest.approx(0.7)
        assert kw["theta_turn_deg"] == pytest.approx(30.0)
        assert kw["control_rate_hz"] == pytest.approx(10.0)

    def test_navigation_provider_receives_gnss_and_dwa(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        gnss_instance = MagicMock()
        dwa_instance = MagicMock()
        mock_gnss_cls.return_value = gnss_instance
        mock_dwa_cls.return_value = dwa_instance
        mock_nav_cls.return_value = MagicMock()

        NavigationBg(config=config)

        kw = mock_nav_cls.call_args.kwargs
        assert kw["gnss"] is gnss_instance
        assert kw["dwa"] is dwa_instance

    def test_navigation_provider_tick_dt_from_monitor_rate(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        mock_nav_cls.return_value = MagicMock()

        NavigationBg(config=config)

        kw = mock_nav_cls.call_args.kwargs
        expected_tick_dt = 1.0 / config.monitor_rate_hz
        assert kw["tick_dt"] == pytest.approx(expected_tick_dt)

    def test_navigation_provider_stored_on_bg(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        nav_instance = MagicMock()
        mock_nav_cls.return_value = nav_instance

        bg = NavigationBg(config=config)

        assert bg.navigation_provider is nav_instance


# ==============================================================================
# run()
# ==============================================================================

@patch("backgrounds.plugins.navigation_bg.NavigationProvider")
@patch("backgrounds.plugins.navigation_bg.DwaRouteProvider")
@patch("backgrounds.plugins.navigation_bg.GnssRouteProvider")
@patch("backgrounds.plugins.navigation_bg.time.sleep")
class TestRun:
    def test_run_calls_start(self, mock_sleep, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        nav_instance = MagicMock()
        mock_nav_cls.return_value = nav_instance

        mock_sleep.side_effect = KeyboardInterrupt  # run() 루프 탈출

        bg = NavigationBg(config=config)
        with pytest.raises(KeyboardInterrupt):
            bg.run()

        nav_instance.start.assert_called_once()

    def test_run_calls_stop_on_exit(self, mock_sleep, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        nav_instance = MagicMock()
        mock_nav_cls.return_value = nav_instance

        mock_sleep.side_effect = KeyboardInterrupt

        bg = NavigationBg(config=config)
        with pytest.raises(KeyboardInterrupt):
            bg.run()

        nav_instance.stop.assert_called_once()

    def test_run_stop_called_even_on_exception(self, mock_sleep, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        nav_instance = MagicMock()
        mock_nav_cls.return_value = nav_instance

        mock_sleep.side_effect = RuntimeError("unexpected")

        bg = NavigationBg(config=config)
        with pytest.raises(RuntimeError):
            bg.run()

        nav_instance.stop.assert_called_once()


# ==============================================================================
# Init failure
# ==============================================================================

@patch("backgrounds.plugins.navigation_bg.NavigationProvider")
@patch("backgrounds.plugins.navigation_bg.DwaRouteProvider")
@patch("backgrounds.plugins.navigation_bg.GnssRouteProvider")
class TestInitFailure:
    def test_gnss_init_failure_propagates(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.side_effect = RuntimeError("gnss init failed")

        with pytest.raises(RuntimeError, match="gnss init failed"):
            NavigationBg(config=config)

    def test_dwa_init_failure_propagates(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.side_effect = RuntimeError("dwa init failed")

        with pytest.raises(RuntimeError, match="dwa init failed"):
            NavigationBg(config=config)

    def test_nav_init_failure_propagates(self, mock_gnss_cls, mock_dwa_cls, mock_nav_cls, config):
        mock_gnss_cls.return_value = MagicMock()
        mock_dwa_cls.return_value = MagicMock()
        mock_nav_cls.side_effect = RuntimeError("nav init failed")

        with pytest.raises(RuntimeError, match="nav init failed"):
            NavigationBg(config=config)
