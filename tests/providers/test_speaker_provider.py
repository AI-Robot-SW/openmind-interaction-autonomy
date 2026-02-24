"""
Tests for SpeakerProvider.
"""

import array
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# PyAudio Mocking (파일 최상단, import 전)
mock_pyaudio = MagicMock()
mock_pyaudio.PyAudio = MagicMock()
mock_pyaudio.paInt16 = 8  # PyAudio constant
sys.modules["pyaudio"] = mock_pyaudio

from providers.speaker_provider import SpeakerProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    SpeakerProvider.reset()  # type: ignore
    yield
    SpeakerProvider.reset()  # type: ignore


@pytest.fixture
def default_config():
    """Default configuration for SpeakerProvider."""
    return {
        "sample_rate": 24000,  # Naver Clova TTS default
        "channels": 1,
        "device_id": None,
        "device_name": None,
        "volume": 1.0,
        "buffer_size": 1024,  # Smaller buffer for lower latency
    }


@pytest.fixture
def custom_config():
    """Custom configuration for SpeakerProvider."""
    return {
        "sample_rate": 48000,
        "channels": 2,
        "device_id": 1,
        "device_name": "Test Speaker",
        "volume": 0.5,
        "buffer_size": 8192,
    }


class TestSpeakerProviderInitialization:
    """Test SpeakerProvider initialization."""

    def test_default_initialization(self, default_config):
        """Test SpeakerProvider initialization with default values."""
        provider = SpeakerProvider(**default_config)

        assert provider.sample_rate == 24000
        assert provider.channels == 1
        assert provider.device_id is None
        assert provider.device_name is None
        assert provider.volume == 1.0
        assert provider.buffer_size == 1024
        assert not provider.running

    def test_custom_initialization(self, custom_config):
        """Test SpeakerProvider initialization with custom values."""
        provider = SpeakerProvider(**custom_config)

        assert provider.sample_rate == 48000
        assert provider.channels == 2
        assert provider.device_id == 1
        assert provider.device_name == "Test Speaker"
        assert provider.volume == 0.5
        assert provider.buffer_size == 8192

    def test_singleton_pattern(self, default_config):
        """Test that SpeakerProvider follows singleton pattern."""
        provider1 = SpeakerProvider(**default_config)
        provider2 = SpeakerProvider(**default_config)

        assert provider1 is provider2

    def test_volume_clamping_high(self):
        """Test volume is clamped to maximum 1.0."""
        provider = SpeakerProvider(volume=1.5)

        assert provider.volume == 1.0

    def test_volume_clamping_low(self):
        """Test volume is clamped to minimum 0.0."""
        provider = SpeakerProvider(volume=-0.5)

        assert provider.volume == 0.0


class TestSpeakerProviderVolume:
    """Test SpeakerProvider volume control."""

    def test_set_volume(self, default_config):
        """Test setting volume."""
        provider = SpeakerProvider(**default_config)

        provider.volume = 0.7

        assert provider.volume == 0.7

    def test_set_volume_clamping_high(self, default_config):
        """Test volume setter clamps high values."""
        provider = SpeakerProvider(**default_config)

        provider.volume = 2.0

        assert provider.volume == 1.0

    def test_set_volume_clamping_low(self, default_config):
        """Test volume setter clamps low values."""
        provider = SpeakerProvider(**default_config)

        provider.volume = -1.0

        assert provider.volume == 0.0


class TestSpeakerProviderCallbacks:
    """Test SpeakerProvider callback registration."""

    def test_register_playback_callback(self, default_config):
        """Test registering playback callback."""
        provider = SpeakerProvider(**default_config)
        callback = MagicMock(__name__="test_callback")

        provider.register_playback_callback(callback)

        assert callback in provider._playback_callbacks

    def test_unregister_playback_callback(self, default_config):
        """Test unregistering playback callback."""
        provider = SpeakerProvider(**default_config)
        callback = MagicMock(__name__="test_callback")

        provider.register_playback_callback(callback)
        provider.unregister_playback_callback(callback)

        assert callback not in provider._playback_callbacks

    def test_duplicate_callback_registration(self, default_config):
        """Test that duplicate callbacks are not registered."""
        provider = SpeakerProvider(**default_config)
        callback = MagicMock(__name__="test_callback")

        provider.register_playback_callback(callback)
        provider.register_playback_callback(callback)

        assert provider._playback_callbacks.count(callback) == 1


class TestSpeakerProviderQueue:
    """Test SpeakerProvider queue management."""

    def test_queue_audio_not_running(self, default_config, caplog):
        """Test queueing audio when not running logs warning."""
        provider = SpeakerProvider(**default_config)

        provider.queue_audio(b"audio_data")

        assert "not running" in caplog.text

    def test_queue_audio_running(self, default_config):
        """Test queueing audio when running."""
        provider = SpeakerProvider(**default_config)
        provider.start()

        provider.queue_audio(b"audio_data")

        assert provider.get_queue_size() == 1

        provider.stop()

    def test_clear_queue(self, default_config):
        """Test clearing the audio queue."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.queue_audio(b"audio_data_1")
        provider.queue_audio(b"audio_data_2")

        provider.clear_queue()

        assert provider.get_queue_size() == 0

        provider.stop()

    def test_get_queue_size_empty(self, default_config):
        """Test queue size when empty."""
        provider = SpeakerProvider(**default_config)

        assert provider.get_queue_size() == 0


class TestSpeakerProviderState:
    """Test SpeakerProvider state methods."""

    def test_is_playing_initial(self, default_config):
        """Test initial playing state is False."""
        provider = SpeakerProvider(**default_config)

        assert provider.is_playing() is False


class TestSpeakerProviderLifecycle:
    """Test SpeakerProvider start/stop lifecycle."""

    def test_start(self, default_config):
        """Test starting the provider."""
        provider = SpeakerProvider(**default_config)
        provider.start()

        assert provider.running is True

        provider.stop()

    def test_start_already_running(self, default_config, caplog):
        """Test starting already running provider logs warning."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.start()

        assert "already running" in caplog.text

        provider.stop()

    def test_stop(self, default_config):
        """Test stopping the provider."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.stop()

        assert provider.running is False

    def test_stop_not_running(self, default_config, caplog):
        """Test stopping not running provider logs warning."""
        provider = SpeakerProvider(**default_config)
        provider.stop()

        assert "not running" in caplog.text

    def test_stop_playback(self, default_config):
        """Test stopping playback."""
        provider = SpeakerProvider(**default_config)
        provider.start()
        provider.queue_audio(b"audio_data")

        provider.stop_playback()

        assert provider.get_queue_size() == 0

        provider.stop()


class TestSpeakerProviderConfiguration:
    """Test SpeakerProvider configuration methods."""

    def test_configure_sample_rate(self, default_config):
        """Test configuring sample rate."""
        provider = SpeakerProvider(**default_config)

        provider.configure(sample_rate=48000)

        assert provider.sample_rate == 48000

    def test_configure_channels(self, default_config):
        """Test configuring channels."""
        provider = SpeakerProvider(**default_config)

        provider.configure(channels=2)

        assert provider.channels == 2

    def test_configure_volume(self, default_config):
        """Test configuring volume."""
        provider = SpeakerProvider(**default_config)

        provider.configure(volume=0.5)

        assert provider.volume == 0.5

    def test_configure_device_id(self, default_config):
        """Test configuring device ID."""
        provider = SpeakerProvider(**default_config)

        provider.configure(device_id=2)

        assert provider.device_id == 2


class TestSpeakerProviderPyAudio:
    """Test SpeakerProvider PyAudio integration."""

    def test_start_initializes_pyaudio(self, default_config):
        """Test start() initializes PyAudio."""
        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_pa.paInt16 = 8
            mock_stream = MagicMock()
            mock_stream.is_active.return_value = True
            mock_instance.open.return_value = mock_stream

            provider = SpeakerProvider(**default_config)
            provider.start()

            mock_pa.PyAudio.assert_called_once()
            mock_instance.open.assert_called_once()

            provider.stop()

    def test_stop_terminates_pyaudio(self, default_config):
        """Test stop() terminates PyAudio properly."""
        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_pa.paInt16 = 8
            mock_stream = MagicMock()
            mock_stream.is_active.return_value = True
            mock_instance.open.return_value = mock_stream

            provider = SpeakerProvider(**default_config)
            provider.start()
            provider.stop()

            mock_stream.stop_stream.assert_called_once()
            mock_stream.close.assert_called_once()
            mock_instance.terminate.assert_called_once()

    def test_start_with_device_name(self, default_config):
        """Test start() with device name lookup."""
        default_config["device_name"] = "Test Speaker"

        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_pa.paInt16 = 8
            mock_instance.get_device_count.return_value = 2
            mock_instance.get_device_info_by_index.side_effect = [
                {"name": "Other Device", "maxOutputChannels": 2},
                {"name": "Test Speaker", "maxOutputChannels": 2},
            ]
            mock_stream = MagicMock()
            mock_stream.is_active.return_value = True
            mock_instance.open.return_value = mock_stream

            provider = SpeakerProvider(**default_config)
            provider.start()

            # device_index=1 (Test Speaker)
            call_kwargs = mock_instance.open.call_args[1]
            assert call_kwargs["output_device_index"] == 1

            provider.stop()


class TestSpeakerProviderVolumeApplication:
    """Test SpeakerProvider volume application."""

    def test_apply_volume_full(self, default_config):
        """Test volume at 1.0 returns unchanged data."""
        provider = SpeakerProvider(**default_config)
        provider._volume = 1.0

        audio_data = b"\x00\x10\x00\x20"  # Sample PCM16 data

        result = provider._apply_volume(audio_data)

        assert result == audio_data

    def test_apply_volume_half(self, default_config):
        """Test volume at 0.5 reduces amplitude."""
        provider = SpeakerProvider(**default_config)
        provider._volume = 0.5

        # Sample: little-endian 16-bit signed integer
        # b'\x00\x10' = 0x1000 = 4096
        audio_data = b"\x00\x10"

        result = provider._apply_volume(audio_data)

        # Expected: 4096 * 0.5 = 2048 = 0x0800
        samples = array.array("h", result)
        assert samples[0] == 2048

    def test_apply_volume_zero(self, default_config):
        """Test volume at 0.0 returns silence."""
        provider = SpeakerProvider(**default_config)
        provider._volume = 0.0

        audio_data = b"\x00\x10\x00\x20"

        result = provider._apply_volume(audio_data)

        assert result == b"\x00\x00\x00\x00"

    def test_apply_volume_clipping_prevention(self, default_config):
        """Test volume application prevents clipping."""
        provider = SpeakerProvider(**default_config)
        provider._volume = 0.9

        # Near max value: 0x7FFF = 32767
        audio_data = b"\xff\x7f"

        result = provider._apply_volume(audio_data)

        # Should not exceed 32767
        samples = array.array("h", result)
        assert samples[0] <= 32767


class TestSpeakerProviderData:
    """Test SpeakerProvider data property."""

    def test_data_property(self, default_config):
        """Test data property returns correct structure."""
        provider = SpeakerProvider(**default_config)

        data = provider.data

        assert "running" in data
        assert "is_playing" in data
        assert "queue_size" in data
        assert "volume" in data
        assert "sample_rate" in data
        assert data["running"] is False
        assert data["queue_size"] == 0

    def test_data_property_running(self, default_config):
        """Test data property when running."""
        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_pa.paInt16 = 8
            mock_stream = MagicMock()
            mock_stream.is_active.return_value = True
            mock_instance.open.return_value = mock_stream

            provider = SpeakerProvider(**default_config)
            provider.start()

            data = provider.data

            assert data["running"] is True

            provider.stop()


class TestSpeakerProviderDeviceDiscovery:
    """Test SpeakerProvider device discovery."""

    def test_get_available_devices(self, default_config):
        """Test getting available output devices."""
        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_instance.get_device_count.return_value = 3
            mock_instance.get_device_info_by_index.side_effect = [
                {"name": "Device 1", "maxOutputChannels": 2, "index": 0},
                {"name": "Input Only", "maxOutputChannels": 0, "index": 1},
                {"name": "Device 2", "maxOutputChannels": 2, "index": 2},
            ]

            provider = SpeakerProvider(**default_config)
            devices = provider.get_available_devices()

            # Input only device should be filtered out
            assert len(devices) == 2
            assert devices[0]["name"] == "Device 1"
            assert devices[1]["name"] == "Device 2"

    def test_find_device_by_name_found(self, default_config):
        """Test finding device by name."""
        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_instance.get_device_count.return_value = 2
            mock_instance.get_device_info_by_index.side_effect = [
                {"name": "Device A", "maxOutputChannels": 2},
                {"name": "Target Device", "maxOutputChannels": 2},
            ]

            provider = SpeakerProvider(**default_config)
            provider._audio_interface = mock_instance

            result = provider._find_device_by_name("Target Device")

            assert result == 1

    def test_find_device_by_name_not_found(self, default_config):
        """Test finding device by name when not found."""
        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_instance.get_device_count.return_value = 1
            mock_instance.get_device_info_by_index.return_value = {
                "name": "Other Device",
                "maxOutputChannels": 2,
            }

            provider = SpeakerProvider(**default_config)
            provider._audio_interface = mock_instance

            result = provider._find_device_by_name("NonExistent")

            assert result is None

    def test_find_device_by_name_partial_match(self, default_config):
        """Test finding device by partial name match."""
        with patch("providers.speaker_provider.pyaudio") as mock_pa:
            mock_instance = MagicMock()
            mock_pa.PyAudio.return_value = mock_instance
            mock_instance.get_device_count.return_value = 2
            mock_instance.get_device_info_by_index.side_effect = [
                {"name": "Device A", "maxOutputChannels": 2},
                {"name": "MacBook Pro Speakers", "maxOutputChannels": 2},
            ]

            provider = SpeakerProvider(**default_config)
            provider._audio_interface = mock_instance

            result = provider._find_device_by_name("MacBook Pro")

            assert result == 1
