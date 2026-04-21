from unittest.mock import patch

import pytest

from backgrounds.plugins.GUI_bg import GUIBg, GUIBgConfig
from providers.audio_provider import AudioProvider
from providers.navigation_provider import NavigationProvider


@pytest.fixture(autouse=True)
def reset_singletons():
    AudioProvider.reset()  # type: ignore[attr-defined]
    NavigationProvider.reset()  # type: ignore[attr-defined]
    yield
    AudioProvider.reset()  # type: ignore[attr-defined]
    NavigationProvider.reset()  # type: ignore[attr-defined]


def _singleton_instance(factory):
    singleton_cls = getattr(factory, "_singleton_class", None)
    return getattr(singleton_cls, "_singleton_instance", None)


def test_gui_bg_does_not_create_provider_singletons_on_init():
    config = GUIBgConfig()

    with patch.object(GUIBg, "_start_server_thread", return_value=None):
        GUIBg(config)

    assert _singleton_instance(AudioProvider) is None
    assert _singleton_instance(NavigationProvider) is None


def test_gui_bg_returns_safe_defaults_without_providers():
    config = GUIBgConfig()

    with patch.object(GUIBg, "_start_server_thread", return_value=None):
        bg = GUIBg(config)

    assert bg._build_audio_payload() == {"level": 0.0, "voice_active": False}
    assert bg._build_navigation_payload() == {}


def test_gui_bg_uses_existing_provider_instances():
    class FakeAudioProvider:
        running = True

        @staticmethod
        def get_audio_level():
            return 0.42

        @staticmethod
        def is_voice_active():
            return True

    class FakeNavigationProvider:
        running = True
        data = {"mode": "DWA", "vx": 0.8}

    config = GUIBgConfig()

    with patch.object(GUIBg, "_start_server_thread", return_value=None):
        bg = GUIBg(config)

    bg._get_audio_provider = lambda: FakeAudioProvider()  # type: ignore[method-assign]
    bg._get_navigation_provider = lambda: FakeNavigationProvider()  # type: ignore[method-assign]

    assert bg._build_audio_payload() == {"level": 0.42, "voice_active": True}
    assert bg._build_navigation_payload() == {"mode": "DWA", "vx": 0.8}
