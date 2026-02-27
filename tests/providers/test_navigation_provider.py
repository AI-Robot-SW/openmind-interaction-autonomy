"""
Tests for NavigationProvider.

Follows .cursor/skills/provider-testing/SKILL.md:
- reset_singleton (autouse) for NavigationProvider (@singleton).
- Mock GnssRouteProvider, DwaRouteProvider, LocationProvider (no HW required).
- Tests: initialization, singleton, lifecycle, state API, control API, worker loop.

Run: uv run pytest tests/providers/test_navigation_provider.py -v
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from providers.navigation_provider import NavigationProvider, NavigationState


# ==============================================================================
# Helpers
# ==============================================================================

def _make_gnss_mock(heading_calibrated=True, reached_goal=False, vx=0.5, vyaw=0.3):
    gnss = MagicMock()
    rec = MagicMock()
    rec.heading_calibrated = heading_calibrated
    rec.reached_goal = reached_goal
    rec.vx = vx
    rec.vyaw = vyaw
    gnss.get_record.return_value = rec
    gnss.waypoints = [(37.0, 127.0), (37.1, 127.1)]
    gnss.reach_tol_m = 5.0
    gnss.v_max = 0.9
    return gnss


def _make_dwa_mock(mode="DWA", vx_cmd=0.6, vyaw_cmd=0.2,
                   heading_calibrated=True, reached_goal=False, vx_fixed=0.8, v_max=0.9):
    dwa = MagicMock()
    rec = MagicMock()
    rec.mode = mode
    rec.vx_cmd = vx_cmd
    rec.vyaw_cmd = vyaw_cmd
    rec.heading_calibrated = heading_calibrated
    rec.reached_goal = reached_goal
    dwa.get_record.return_value = rec
    dwa.vx_fixed = vx_fixed
    dwa.v_max = v_max
    return dwa


def _make_provider(gnss=None, dwa=None, **kwargs):
    gnss = gnss or _make_gnss_mock()
    dwa = dwa or _make_dwa_mock()
    return NavigationProvider(gnss=gnss, dwa=dwa, **kwargs)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    NavigationProvider.reset()  # type: ignore
    yield
    NavigationProvider.reset()  # type: ignore


# ==============================================================================
# Initialization
# ==============================================================================

class TestInitialization:
    def test_stores_dependencies(self):
        gnss = _make_gnss_mock()
        dwa = _make_dwa_mock()
        p = NavigationProvider(gnss=gnss, dwa=dwa)
        assert p._gnss is gnss
        assert p._dwa is dwa

    def test_default_speed_params(self):
        dwa = _make_dwa_mock(v_max=0.9)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa)
        assert p._speed_step == 0.1
        assert p._speed_min == 0.2
        assert p._speed_max == pytest.approx(0.9)  # dwa.v_max 연동

    def test_speed_max_overridable(self):
        dwa = _make_dwa_mock(v_max=0.9)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, speed_max=1.2)
        assert p._speed_max == pytest.approx(1.2)

    def test_initial_state_is_idle(self):
        p = _make_provider()
        st = p.get_state()
        assert st.vx == 0.0
        assert st.mode == "IDLE"
        assert not p.running

    def test_active_path_initially_none(self):
        p = _make_provider()
        assert p.get_active_path() is None

    def test_singleton_pattern(self):
        gnss = _make_gnss_mock()
        dwa = _make_dwa_mock()
        p1 = NavigationProvider(gnss=gnss, dwa=dwa)
        p2 = NavigationProvider(gnss=MagicMock(), dwa=MagicMock())
        assert p1 is p2


# ==============================================================================
# Lifecycle
# ==============================================================================

class TestLifecycle:
    def test_start_calls_sub_providers(self):
        gnss = _make_gnss_mock()
        dwa = _make_dwa_mock()
        p = NavigationProvider(gnss=gnss, dwa=dwa)
        p.start()
        gnss.start.assert_called_once()
        dwa.start.assert_called_once()
        assert p.running
        p.stop()

    def test_stop_calls_sub_providers(self):
        gnss = _make_gnss_mock()
        dwa = _make_dwa_mock()
        p = NavigationProvider(gnss=gnss, dwa=dwa)
        p.start()
        p.stop()
        dwa.stop.assert_called_once()
        gnss.stop.assert_called_once()
        assert not p.running

    def test_stop_state_is_STOP(self):
        p = _make_provider()
        p.start()
        p.stop()
        assert p.get_state().mode == "STOP"

    def test_start_idempotent(self):
        gnss = _make_gnss_mock()
        dwa = _make_dwa_mock()
        p = NavigationProvider(gnss=gnss, dwa=dwa)
        p.start()
        p.start()
        gnss.start.assert_called_once()
        p.stop()

    def test_stop_without_start_is_safe(self):
        p = _make_provider()
        p.stop()  # should not raise


# ==============================================================================
# Worker loop — _run()
# ==============================================================================

class TestWorkerLoop:
    def _wait_state(self, p, condition, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition(p.get_state()):
                return True
            time.sleep(0.01)
        return False

    def test_dwa_mode_propagates(self):
        dwa = _make_dwa_mock(mode="DWA", vx_cmd=0.7, vyaw_cmd=0.15)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, tick_dt=0.01)
        p.start()
        assert self._wait_state(p, lambda s: s.mode == "DWA")
        st = p.get_state()
        assert st.vx == pytest.approx(0.7)
        assert st.vyaw == pytest.approx(0.15)
        p.stop()

    def test_calibrating_mode_uses_gnss_velocity(self):
        gnss = _make_gnss_mock(heading_calibrated=False, vx=0.5, vyaw=0.3)
        dwa = _make_dwa_mock(mode="STOP", heading_calibrated=False)
        p = NavigationProvider(gnss=gnss, dwa=dwa, tick_dt=0.01)
        p.start()
        assert self._wait_state(p, lambda s: s.mode == "CALIBRATING")
        st = p.get_state()
        assert st.vx == pytest.approx(0.5)
        assert st.vyaw == pytest.approx(0.3)
        p.stop()

    def test_stop_mode_vx_zero(self):
        gnss = _make_gnss_mock(heading_calibrated=True)
        dwa = _make_dwa_mock(mode="STOP", heading_calibrated=True)
        p = NavigationProvider(gnss=gnss, dwa=dwa, tick_dt=0.01)
        p.start()
        assert self._wait_state(p, lambda s: s.mode == "STOP")
        assert p.get_state().vx == 0.0
        p.stop()

    def test_idle_when_no_dwa_record(self):
        dwa = _make_dwa_mock()
        dwa.get_record.return_value = None
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, tick_dt=0.01)
        p.start()
        assert self._wait_state(p, lambda s: s.mode == "IDLE")
        p.stop()

    def test_reached_goal_propagates(self):
        dwa = _make_dwa_mock(mode="STOP", reached_goal=True, heading_calibrated=True)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, tick_dt=0.01)
        p.start()
        assert self._wait_state(p, lambda s: s.reached_goal is True)
        p.stop()


# ==============================================================================
# State / data API
# ==============================================================================

class TestStateAPI:
    def test_data_none_when_not_running(self):
        p = _make_provider()
        assert p.data is None

    def test_data_returns_dict_when_running(self):
        dwa = _make_dwa_mock(mode="DWA", vx_cmd=0.5, vyaw_cmd=0.1)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, tick_dt=0.01)
        p.start()
        time.sleep(0.05)
        d = p.data
        assert d is not None
        assert "vx" in d and "vyaw" in d and "mode" in d
        p.stop()

    def test_get_next_move_returns_tuple(self):
        p = _make_provider()
        move = p.get_next_move()
        assert isinstance(move, tuple)
        assert len(move) == 3


# ==============================================================================
# Control API
# ==============================================================================

class TestControlAPI:
    def test_get_target_speed(self):
        dwa = _make_dwa_mock(vx_fixed=0.8)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa)
        assert p.get_target_speed() == pytest.approx(0.8)

    def test_step_faster_increases_vx_fixed(self):
        dwa = _make_dwa_mock(vx_fixed=0.5)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, speed_step=0.1, speed_max=0.9)
        p.step_faster()
        assert dwa.vx_fixed == pytest.approx(0.6)

    def test_step_slower_decreases_vx_fixed(self):
        dwa = _make_dwa_mock(vx_fixed=0.5)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, speed_step=0.1, speed_min=0.2)
        p.step_slower()
        assert dwa.vx_fixed == pytest.approx(0.4)

    def test_step_faster_capped_at_speed_max(self):
        dwa = _make_dwa_mock(vx_fixed=0.85)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, speed_step=0.1, speed_max=0.9)
        p.step_faster()
        assert dwa.vx_fixed == pytest.approx(0.9)

    def test_step_slower_capped_at_speed_min(self):
        dwa = _make_dwa_mock(vx_fixed=0.25)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, speed_step=0.1, speed_min=0.2)
        p.step_slower()
        assert dwa.vx_fixed == pytest.approx(0.2)

    def test_step_faster_does_not_change_v_max(self):
        """v_max는 speed 조절 시 변경되지 않아야 함."""
        dwa = _make_dwa_mock(vx_fixed=0.5, v_max=0.9)
        p = NavigationProvider(gnss=_make_gnss_mock(), dwa=dwa, speed_step=0.1, speed_max=0.9)
        p.step_faster()
        assert dwa.v_max == 0.9  # 변경 없음

    def test_get_active_path_none_before_set(self):
        p = _make_provider()
        assert p.get_active_path() is None

    def test_get_active_path_after_set(self):
        gnss = _make_gnss_mock()
        gnss.waypoints = []
        p = NavigationProvider(gnss=gnss, dwa=_make_dwa_mock())
        waypoints = [(37.0, 127.0), (37.1, 127.1)]
        p.set_path(waypoints)
        assert p.get_active_path() == waypoints
        p.stop()

    def test_set_path_updates_gnss_waypoints(self):
        gnss = _make_gnss_mock()
        dwa = _make_dwa_mock()
        p = NavigationProvider(gnss=gnss, dwa=dwa)
        new_waypoints = [(37.5, 127.5), (37.6, 127.6)]
        p.set_path(new_waypoints)
        assert gnss.waypoints == new_waypoints
        p.stop()

    def test_set_path_empty_list_ignored(self, caplog):
        p = _make_provider()
        p.set_path([])
        assert p.get_active_path() is None

    def test_get_remaining_distance_no_waypoints(self):
        gnss = _make_gnss_mock()
        gnss.waypoints = []
        p = NavigationProvider(gnss=gnss, dwa=_make_dwa_mock())
        assert p.get_remaining_distance() == 0.0

    def test_get_remaining_distance_with_waypoints(self):
        gnss = _make_gnss_mock()
        gnss.waypoints = [(37.6038667, 127.0453007), (37.6038618, 127.0452945)]
        gnss.reach_tol_m = 5.0

        mock_state = MagicMock()
        mock_state.gnss.lat = 37.6038667
        mock_state.gnss.lon = 127.0453007

        with patch("providers.navigation_provider.LocationProvider") as mock_loc_cls:
            mock_loc_cls.return_value.get_state.return_value = mock_state
            p = NavigationProvider(gnss=gnss, dwa=_make_dwa_mock())
            dist = p.get_remaining_distance()

        assert dist >= 0.0

    def test_get_remaining_distance_returns_zero_on_error(self):
        gnss = _make_gnss_mock()
        gnss.waypoints = [(37.6, 127.0)]
        gnss.reach_tol_m = 5.0

        with patch("providers.navigation_provider.LocationProvider") as mock_loc_cls:
            mock_loc_cls.side_effect = Exception("no location")
            p = NavigationProvider(gnss=gnss, dwa=_make_dwa_mock())
            dist = p.get_remaining_distance()

        assert dist == 0.0
