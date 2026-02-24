"""
Tests for AudioBg.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.audio_bg import AudioBg, AudioBgConfig


@pytest.fixture
def config():
    """Configuration for AudioBg."""
    return AudioBgConfig(
        sample_rate=16000,
        chunk_size=1024,
        channels=1,
        device_id=None,
        device_name=None,
        vad_enabled=True,
        vad_threshold=0.5,
        health_check_interval_sec=10.0,
    )


@pytest.fixture
def config_custom():
    """Custom configuration for AudioBg."""
    return AudioBgConfig(
        sample_rate=48000,
        chunk_size=2048,
        channels=2,
        device_id=1,
        device_name="Test Microphone",
        vad_enabled=False,
        vad_threshold=0.7,
        health_check_interval_sec=5.0,
    )


@pytest.fixture
def config_default():
    """Default configuration for AudioBg."""
    return AudioBgConfig()


class TestAudioBgConfig:
    """Test AudioBgConfig initialization."""

    def test_config_default_values(self):
        """Test AudioBgConfig default values."""
        config = AudioBgConfig()

        assert config.sample_rate == 16000
        assert config.chunk_size == 1024
        assert config.channels == 1
        assert config.device_id is None
        assert config.device_name is None
        assert config.vad_enabled is True
        assert config.vad_threshold == 0.5
        assert config.health_check_interval_sec == 10.0

    def test_config_custom_values(self):
        """Test AudioBgConfig with custom values."""
        config = AudioBgConfig(
            sample_rate=48000,
            chunk_size=2048,
            channels=2,
            device_id=1,
            vad_enabled=False,
            vad_threshold=0.7,
        )

        assert config.sample_rate == 48000
        assert config.chunk_size == 2048
        assert config.channels == 2
        assert config.device_id == 1
        assert config.vad_enabled is False
        assert config.vad_threshold == 0.7


class TestAudioBgInitialization:
    """Test AudioBg initialization."""

    @patch("backgrounds.plugins.audio_bg.AudioProvider")
    def test_background_initialization(self, mock_provider_class, config):
        """Test AudioBg initialization."""
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        background = AudioBg(config=config)

        # Verify provider was initialized with correct parameters
        mock_provider_class.assert_called_once_with(
            sample_rate=16000,
            chunk_size=1024,
            channels=1,
            device_id=None,
            device_name=None,
            vad_enabled=True,
            vad_threshold=0.5,
        )
        assert background.audio_provider is mock_provider_instance
        mock_provider_instance.start_stream.assert_called_once()

    @patch("backgrounds.plugins.audio_bg.AudioProvider")
    def test_background_initialization_custom(self, mock_provider_class, config_custom):
        """Test AudioBg initialization with custom config."""
        mock_provider_instance = MagicMock()
        mock_provider_class.return_value = mock_provider_instance

        background = AudioBg(config=config_custom)

        mock_provider_class.assert_called_once_with(
            sample_rate=48000,
            chunk_size=2048,
            channels=2,
            device_id=1,
            device_name="Test Microphone",
            vad_enabled=False,
            vad_threshold=0.7,
        )

    def test_background_name(self, config):
        """Test that background has correct name."""
        background = AudioBg(config=config)

        assert background.name == "AudioBg"

    def test_background_config_access(self, config):
        """Test that background has access to config."""
        background = AudioBg(config=config)

        assert background.config == config
        assert background.config.sample_rate == 16000
        assert background.config.vad_enabled is True


class TestAudioBgHealthCheck:
    """Test AudioBg health check functionality."""

    def test_health_check_returns_true(self, config):
        """Test health check returns True when provider is healthy."""
        background = AudioBg(config=config)

        # TODO: Mock provider.running = True
        result = background._health_check()

        assert result is True

    def test_consecutive_failures_tracking(self, config):
        """Test consecutive failures are tracked."""
        background = AudioBg(config=config)

        assert background._consecutive_failures == 0
        assert background._max_failures == 3


class TestAudioBgRun:
    """Test AudioBg run method."""

    def test_run_respects_health_check_interval(self, config):
        """Test run method respects health check interval."""
        background = AudioBg(config=config)
        background._last_health_check = time.time()

        # Should not perform health check immediately
        start_time = time.time()
        background.run()
        elapsed = time.time() - start_time

        # run() should sleep for ~1 second
        assert elapsed >= 0.9

    def test_run_performs_health_check_after_interval(self, config):
        """Test run method performs health check after interval."""
        background = AudioBg(config=config)
        # Set last health check to past
        background._last_health_check = time.time() - 15.0

        background.run()

        # Health check should have been performed
        assert time.time() - background._last_health_check < 2.0
