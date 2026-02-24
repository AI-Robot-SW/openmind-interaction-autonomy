"""
Tests for TTSBg.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from backgrounds.plugins.tts_bg import TTSBg, TTSBgConfig


@pytest.fixture
def config():
    """Configuration for TTSBg."""
    return TTSBgConfig(
        api_key="test_api_key",
        backend_api_key=None,
        backend="elevenlabs",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
        language="ko",
        enable_tts_interrupt=False,
        health_check_interval_sec=10.0,
    )


@pytest.fixture
def config_custom():
    """Custom configuration for TTSBg."""
    return TTSBgConfig(
        api_key="custom_api_key",
        backend_api_key="elevenlabs_key",
        backend="elevenlabs",
        voice_id="custom_voice",
        model_id="custom_model",
        output_format="pcm_44100",
        language="en",
        enable_tts_interrupt=True,
        health_check_interval_sec=5.0,
    )


@pytest.fixture
def config_default():
    """Default configuration for TTSBg."""
    return TTSBgConfig()


class TestTTSBgConfig:
    """Test TTSBgConfig initialization."""

    def test_config_default_values(self):
        """Test TTSBgConfig default values."""
        config = TTSBgConfig()

        assert config.api_key is None
        assert config.backend_api_key is None
        assert config.backend == "elevenlabs"
        assert config.voice_id is None
        assert config.model_id == "eleven_flash_v2_5"
        assert config.output_format == "mp3_44100_128"
        assert config.language == "ko"
        assert config.enable_tts_interrupt is False
        assert config.health_check_interval_sec == 10.0

    def test_config_custom_values(self):
        """Test TTSBgConfig with custom values."""
        config = TTSBgConfig(
            api_key="test_key",
            backend_api_key="elevenlabs_key",
            voice_id="custom_voice",
            model_id="custom_model",
            language="en",
            enable_tts_interrupt=True,
        )

        assert config.api_key == "test_key"
        assert config.backend_api_key == "elevenlabs_key"
        assert config.voice_id == "custom_voice"
        assert config.model_id == "custom_model"
        assert config.language == "en"
        assert config.enable_tts_interrupt is True


class TestTTSBgInitialization:
    """Test TTSBg initialization."""

    @patch("backgrounds.plugins.tts_bg.SpeakerProvider")
    @patch("backgrounds.plugins.tts_bg.TTSProvider")
    def test_background_initialization(
        self, mock_tts_provider_class, mock_speaker_provider_class, config
    ):
        """Test TTSBg initialization."""
        mock_tts_instance = MagicMock()
        mock_tts_provider_class.return_value = mock_tts_instance
        mock_speaker_instance = MagicMock()
        mock_speaker_provider_class.return_value = mock_speaker_instance

        background = TTSBg(config=config)

        # Verify TTS provider was initialized
        assert mock_tts_provider_class.called
        assert background.tts_provider is mock_tts_instance
        mock_tts_instance.start.assert_called_once()

        # Verify audio callback was registered to speaker
        mock_tts_instance.register_audio_callback.assert_called_once_with(
            mock_speaker_instance.queue_audio
        )

    @patch("backgrounds.plugins.tts_bg.SpeakerProvider")
    @patch("backgrounds.plugins.tts_bg.TTSProvider")
    def test_background_initialization_custom(
        self, mock_tts_provider_class, mock_speaker_provider_class, config_custom
    ):
        """Test TTSBg initialization with custom config."""
        mock_tts_instance = MagicMock()
        mock_tts_provider_class.return_value = mock_tts_instance

        background = TTSBg(config=config_custom)

        # Verify custom parameters were passed
        call_kwargs = mock_tts_provider_class.call_args[1]
        assert call_kwargs["voice_id"] == "custom_voice"
        assert call_kwargs["model_id"] == "custom_model"
        assert call_kwargs["language"] == "en"
        assert call_kwargs["enable_tts_interrupt"] is True

    def test_background_name(self, config):
        """Test that background has correct name."""
        background = TTSBg(config=config)

        assert background.name == "TTSBg"

    def test_background_config_access(self, config):
        """Test that background has access to config."""
        background = TTSBg(config=config)

        assert background.config == config
        assert background.config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert background.config.language == "ko"


class TestTTSBgHealthCheck:
    """Test TTSBg health check functionality."""

    def test_health_check_returns_true(self, config):
        """Test health check returns True when provider is healthy."""
        background = TTSBg(config=config)

        result = background._health_check()

        assert result is True

    def test_consecutive_failures_tracking(self, config):
        """Test consecutive failures are tracked."""
        background = TTSBg(config=config)

        assert background._consecutive_failures == 0
        assert background._max_failures == 3


class TestTTSBgRun:
    """Test TTSBg run method."""

    def test_run_respects_health_check_interval(self, config):
        """Test run method respects health check interval."""
        background = TTSBg(config=config)
        background._last_health_check = time.time()

        start_time = time.time()
        background.run()
        elapsed = time.time() - start_time

        assert elapsed >= 0.9

    def test_run_performs_health_check_after_interval(self, config):
        """Test run method performs health check after interval."""
        background = TTSBg(config=config)
        background._last_health_check = time.time() - 15.0

        background.run()

        assert time.time() - background._last_health_check < 2.0

    def test_run_resets_failures_on_success(self, config):
        """Test run resets failure counter on successful health check."""
        background = TTSBg(config=config)
        background._consecutive_failures = 2
        background._last_health_check = time.time() - 15.0

        background.run()

        assert background._consecutive_failures == 0
