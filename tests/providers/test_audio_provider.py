"""
Tests for AudioProvider.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from providers.audio_provider import AudioProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    AudioProvider.reset()  # type: ignore
    yield
    AudioProvider.reset()  # type: ignore


@pytest.fixture
def default_config():
    """Default configuration for AudioProvider."""
    return {
        "sample_rate": 16000,
        "chunk_size": 1024,
        "channels": 1,
        "device_id": None,
        "device_name": None,
        "vad_enabled": True,
        "vad_threshold": 0.5,
    }


@pytest.fixture
def custom_config():
    """Custom configuration for AudioProvider."""
    return {
        "sample_rate": 48000,
        "chunk_size": 2048,
        "channels": 2,
        "device_id": 1,
        "device_name": "Test Microphone",
        "vad_enabled": False,
        "vad_threshold": 0.7,
    }


class TestAudioProviderInitialization:
    """Test AudioProvider initialization."""

    @patch("providers.audio_provider.SileroVAD")
    def test_default_initialization(self, mock_vad_class, default_config):
        """Test AudioProvider initialization with default values."""
        provider = AudioProvider(**default_config)

        assert provider.sample_rate == 16000
        assert provider.chunk_size == 1024
        assert provider.channels == 1
        assert provider.device_id is None
        assert provider.device_name is None
        assert provider.vad_enabled is True
        assert provider.vad_threshold == 0.5
        assert not provider.running

    def test_custom_initialization(self, custom_config):
        """Test AudioProvider initialization with custom values."""
        provider = AudioProvider(**custom_config)

        assert provider.sample_rate == 48000
        assert provider.chunk_size == 2048
        assert provider.channels == 2
        assert provider.device_id == 1
        assert provider.device_name == "Test Microphone"
        assert provider.vad_enabled is False
        assert provider.vad_threshold == 0.7

    def test_singleton_pattern(self, default_config):
        """Test that AudioProvider follows singleton pattern."""
        provider1 = AudioProvider(**default_config)
        provider2 = AudioProvider(**default_config)

        assert provider1 is provider2


class TestAudioProviderCallbacks:
    """Test AudioProvider callback registration."""

    def test_register_audio_callback(self, default_config):
        """Test registering audio callback."""
        provider = AudioProvider(**default_config)
        callback = MagicMock()

        provider.register_audio_callback(callback)

        assert callback in provider._audio_callbacks

    def test_unregister_audio_callback(self, default_config):
        """Test unregistering audio callback."""
        provider = AudioProvider(**default_config)
        callback = MagicMock()

        provider.register_audio_callback(callback)
        provider.unregister_audio_callback(callback)

        assert callback not in provider._audio_callbacks

    def test_register_vad_callback(self, default_config):
        """Test registering VAD callback."""
        provider = AudioProvider(**default_config)
        callback = MagicMock()

        provider.register_vad_callback(callback)

        assert callback in provider._vad_callbacks

    def test_unregister_vad_callback(self, default_config):
        """Test unregistering VAD callback."""
        provider = AudioProvider(**default_config)
        callback = MagicMock()

        provider.register_vad_callback(callback)
        provider.unregister_vad_callback(callback)

        assert callback not in provider._vad_callbacks

    def test_duplicate_callback_registration(self, default_config):
        """Test that duplicate callbacks are not registered."""
        provider = AudioProvider(**default_config)
        callback = MagicMock()

        provider.register_audio_callback(callback)
        provider.register_audio_callback(callback)

        assert provider._audio_callbacks.count(callback) == 1


class TestAudioProviderState:
    """Test AudioProvider state methods."""

    def test_get_audio_level_initial(self, default_config):
        """Test initial audio level is zero."""
        provider = AudioProvider(**default_config)

        assert provider.get_audio_level() == 0.0

    def test_is_voice_active_initial(self, default_config):
        """Test initial voice active state is False."""
        provider = AudioProvider(**default_config)

        assert provider.is_voice_active() is False


class TestAudioProviderLifecycle:
    """Test AudioProvider start/stop lifecycle."""

    @patch("providers.audio_provider.pyaudio")
    def test_start(self, mock_pa_module, default_config):
        """Test starting the provider."""
        provider = AudioProvider(**default_config)
        provider.start()

        assert provider.running is True

    @patch("providers.audio_provider.pyaudio")
    def test_start_already_running(self, mock_pa_module, default_config, caplog):
        """Test starting already running provider logs warning."""
        provider = AudioProvider(**default_config)
        provider.start()
        provider.start()

        assert "already running" in caplog.text

    @patch("providers.audio_provider.pyaudio")
    def test_stop(self, mock_pa_module, default_config):
        """Test stopping the provider."""
        provider = AudioProvider(**default_config)
        provider.start()
        provider.stop()

        assert provider.running is False

    def test_stop_not_running(self, default_config, caplog):
        """Test stopping not running provider logs warning."""
        provider = AudioProvider(**default_config)
        provider.stop()

        assert "not running" in caplog.text


class TestAudioProviderRemoteInput:
    """Test AudioProvider remote_input mode."""

    def test_remote_input_start_no_pyaudio(self):
        """Test remote_input mode doesn't create PyAudio."""
        provider = AudioProvider(
            sample_rate=16000,
            chunk_size=1024,
            remote_input=True,
            vad_enabled=False,
        )
        provider.start()

        assert provider.running is True
        assert provider._pa is None
        assert provider._stream is None

    def test_fill_buffer_remote(self):
        """Test feeding audio via fill_buffer_remote."""
        provider = AudioProvider(
            sample_rate=16000,
            chunk_size=1024,
            remote_input=True,
            vad_enabled=False,
        )
        provider.start()
        audio_data = b"\x00" * 2048
        provider.fill_buffer_remote(audio_data)

        chunk = provider.get_audio_chunk()
        assert chunk == audio_data

    def test_fill_buffer_remote_ignored_when_not_remote(self):
        """Test fill_buffer_remote is ignored in normal mode."""
        provider = AudioProvider(
            sample_rate=16000,
            chunk_size=1024,
            remote_input=False,
            vad_enabled=False,
        )
        provider.fill_buffer_remote(b"\x00" * 2048)

        chunk = provider.get_audio_chunk()
        assert chunk is None


class TestAudioProviderBuffer:
    """Test AudioProvider buffer operations."""

    def test_get_buffer_stats(self):
        """Test buffer stats reporting."""
        provider = AudioProvider(
            sample_rate=16000,
            chunk_size=1024,
            remote_input=True,
            vad_enabled=False,
        )
        stats = provider.get_buffer_stats()

        assert "queued_chunks" in stats
        assert "max_chunks" in stats
        assert "drops" in stats
        assert "approx_latency_ms" in stats
        assert stats["queued_chunks"] == 0.0

    def test_clear_audio_buffer(self):
        """Test clearing audio buffer."""
        provider = AudioProvider(
            sample_rate=16000,
            chunk_size=1024,
            remote_input=True,
            vad_enabled=False,
        )
        provider.start()
        provider.fill_buffer_remote(b"\x00" * 2048)
        provider.clear_audio_buffer()

        chunk = provider.get_audio_chunk()
        assert chunk is None


class TestAudioProviderConfiguration:
    """Test AudioProvider configuration methods."""

    def test_configure_sample_rate(self, default_config):
        """Test configuring sample rate."""
        provider = AudioProvider(**default_config)

        provider.configure(sample_rate=48000)

        assert provider.sample_rate == 48000

    def test_configure_vad_threshold(self, default_config):
        """Test configuring VAD threshold."""
        provider = AudioProvider(**default_config)

        provider.configure(vad_threshold=0.8)

        assert provider.vad_threshold == 0.8

    def test_configure_vad_enabled(self, default_config):
        """Test configuring VAD enabled."""
        provider = AudioProvider(**default_config)

        provider.configure(vad_enabled=False)

        assert provider.vad_enabled is False

    def test_configure_device_id(self, default_config):
        """Test configuring device ID."""
        provider = AudioProvider(**default_config)

        provider.configure(device_id=2)

        assert provider.device_id == 2
