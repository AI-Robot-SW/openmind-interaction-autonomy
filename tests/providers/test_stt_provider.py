"""
Tests for STTProvider.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from providers.stt_provider import LANGUAGE_CODE_MAP, STTProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    STTProvider.reset()  # type: ignore
    yield
    STTProvider.reset()  # type: ignore


@pytest.fixture
def default_config():
    """Default configuration for STTProvider."""
    return {
        "ws_url": "wss://api.openmind.org/api/core/google/asr",
        "api_key": "test_api_key",
        "language": "korean",
        "enable_interim_results": True,
        "sample_rate": 16000,
    }


@pytest.fixture
def custom_config():
    """Custom configuration for STTProvider."""
    return {
        "ws_url": "wss://custom.api.com/asr",
        "api_key": "custom_api_key",
        "language": "english",
        "enable_interim_results": False,
        "sample_rate": 48000,
    }


class TestLanguageCodeMap:
    """Test language code mapping."""

    def test_korean_language_code(self):
        """Test Korean language code."""
        assert LANGUAGE_CODE_MAP["korean"] == "ko-KR"

    def test_english_language_code(self):
        """Test English language code."""
        assert LANGUAGE_CODE_MAP["english"] == "en-US"

    # NOW, WE DON'T HAVE TO SUPPORT OTHER LANGUAGES THAN KOREAN AND ENGLISH. SO JUST LEAVE SAMPLE CODE FOR EXTENSION.
    # def test_japanese_language_code(self):
    #     """Test Japanese language code."""
    #     assert LANGUAGE_CODE_MAP["japanese"] == "ja-JP"

    def test_all_required_languages_present(self):
        """Test all required languages are in the map."""
        required = ["korean", "english"]
        # required = ["korean", "english", "japanese", "chinese", "german", "french", "spanish"]
        for lang in required:
            assert lang in LANGUAGE_CODE_MAP

    def test_no_duplicate_codes(self):
        """Test no duplicate language codes."""
        codes = list(LANGUAGE_CODE_MAP.values())
        assert len(codes) == len(set(codes))


class TestSTTProviderInitialization:
    """Test STTProvider initialization."""

    def test_default_initialization(self, default_config):
        """Test STTProvider initialization with default values."""
        provider = STTProvider(**default_config)

        assert provider.ws_url == "wss://api.openmind.org/api/core/google/asr"
        assert provider.api_key == "test_api_key"
        assert provider.language_code == "ko-KR"
        assert provider.enable_interim_results is True
        assert provider.sample_rate == 16000
        assert not provider.running

    def test_custom_initialization(self, custom_config):
        """Test STTProvider initialization with custom values."""
        provider = STTProvider(**custom_config)

        assert provider.ws_url == "wss://custom.api.com/asr"
        assert provider.language_code == "en-US"
        assert provider.enable_interim_results is False
        assert provider.sample_rate == 48000

    def test_singleton_pattern(self, default_config):
        """Test that STTProvider follows singleton pattern."""
        provider1 = STTProvider(**default_config)
        provider2 = STTProvider(**default_config)

        assert provider1 is provider2

    def test_unsupported_language_defaults_to_korean(self, caplog):
        """Test unsupported language defaults to Korean with warning."""
        provider = STTProvider(
            ws_url="wss://test.com",
            language="klingon",
        )

        assert provider.language_code == "ko-KR"
        assert "not supported" in caplog.text

    def test_case_insensitive_language(self):
        """Test language names are case insensitive."""
        provider = STTProvider(
            ws_url="wss://test.com",
            language="KOREAN",
        )

        assert provider.language_code == "ko-KR"

    def test_whitespace_in_language(self):
        """Test language names handle whitespace."""
        provider = STTProvider(
            ws_url="wss://test.com",
            language="  korean  ",
        )

        assert provider.language_code == "ko-KR"


class TestSTTProviderCallbacks:
    """Test STTProvider callback registration."""

    def test_register_result_callback(self, default_config):
        """Test registering result callback."""
        provider = STTProvider(**default_config)
        callback = MagicMock()

        provider.register_result_callback(callback)

        assert callback in provider._result_callbacks

    def test_unregister_result_callback(self, default_config):
        """Test unregistering result callback."""
        provider = STTProvider(**default_config)
        callback = MagicMock()

        provider.register_result_callback(callback)
        provider.unregister_result_callback(callback)

        assert callback not in provider._result_callbacks

    def test_register_interim_callback(self, default_config):
        """Test registering interim callback."""
        provider = STTProvider(**default_config)
        callback = MagicMock()

        provider.register_interim_callback(callback)

        assert callback in provider._interim_callbacks

    def test_unregister_interim_callback(self, default_config):
        """Test unregistering interim callback."""
        provider = STTProvider(**default_config)
        callback = MagicMock()

        provider.register_interim_callback(callback)
        provider.unregister_interim_callback(callback)

        assert callback not in provider._interim_callbacks


class TestSTTProviderMessageHandling:
    """Test STTProvider message handling."""

    def test_on_message_final_result(self, default_config):
        """Test handling final ASR result."""
        provider = STTProvider(**default_config)
        callback = MagicMock()
        provider.register_result_callback(callback)

        message = json.dumps({
            "asr_reply": "안녕하세요",
            "is_final": True,
        })
        provider._on_message(message)

        callback.assert_called_once_with("안녕하세요")
        assert provider.get_current_transcript() == "안녕하세요"

    def test_on_message_interim_result(self, default_config):
        """Test handling interim ASR result."""
        provider = STTProvider(**default_config)
        interim_callback = MagicMock()
        provider.register_interim_callback(interim_callback)

        message = json.dumps({
            "asr_reply": "안녕",
            "is_final": False,
        })
        provider._on_message(message)

        interim_callback.assert_called_once_with("안녕")

    def test_on_message_interim_disabled(self, custom_config):
        """Test interim results not called when disabled."""
        provider = STTProvider(**custom_config)
        interim_callback = MagicMock()
        provider.register_interim_callback(interim_callback)

        message = json.dumps({
            "asr_reply": "hello",
            "is_final": False,
        })
        provider._on_message(message)

        interim_callback.assert_not_called()

    def test_on_message_invalid_json(self, default_config, caplog):
        """Test handling invalid JSON message."""
        provider = STTProvider(**default_config)

        provider._on_message("invalid json")

        assert "Invalid JSON" in caplog.text

    def test_on_message_error(self, default_config, caplog):
        """Test handling error message from service."""
        provider = STTProvider(**default_config)

        message = json.dumps({"error": "Service unavailable"})
        provider._on_message(message)

        assert "Service unavailable" in caplog.text


class TestSTTProviderState:
    """Test STTProvider state methods."""

    def test_get_current_transcript_initial(self, default_config):
        """Test initial transcript is empty."""
        provider = STTProvider(**default_config)

        assert provider.get_current_transcript() == ""

    def test_is_listening_initial(self, default_config):
        """Test initial listening state is False."""
        provider = STTProvider(**default_config)

        assert provider.is_listening() is False


class TestSTTProviderLifecycle:
    """Test STTProvider start/stop lifecycle."""

    def test_start(self, default_config):
        """Test starting the provider."""
        provider = STTProvider(**default_config)
        provider.start()

        assert provider.running is True
        assert provider.is_listening() is True

    def test_start_already_running(self, default_config, caplog):
        """Test starting already running provider logs warning."""
        provider = STTProvider(**default_config)
        provider.start()
        provider.start()

        assert "already running" in caplog.text

    def test_stop(self, default_config):
        """Test stopping the provider."""
        provider = STTProvider(**default_config)
        provider.start()
        provider.stop()

        assert provider.running is False
        assert provider.is_listening() is False

    def test_stop_not_running(self, default_config, caplog):
        """Test stopping not running provider logs warning."""
        provider = STTProvider(**default_config)
        provider.stop()

        assert "not running" in caplog.text


class TestSTTProviderConfiguration:
    """Test STTProvider configuration methods."""

    def test_configure_language(self, default_config):
        """Test configuring language."""
        provider = STTProvider(**default_config)

        provider.configure(language="english")

        assert provider.language_code == "en-US"

    def test_configure_interim_results(self, default_config):
        """Test configuring interim results."""
        provider = STTProvider(**default_config)

        provider.configure(enable_interim_results=False)

        assert provider.enable_interim_results is False

    def test_configure_sample_rate(self, default_config):
        """Test configuring sample rate."""
        provider = STTProvider(**default_config)

        provider.configure(sample_rate=48000)

        assert provider.sample_rate == 48000


class TestSTTProviderPauseResume:
    """Test STTProvider pause/resume for echo prevention."""

    def test_pause(self, default_config):
        """Test pausing STT provider."""
        provider = STTProvider(**default_config)
        provider.start()

        assert provider.is_listening() is True
        provider.pause()
        assert provider.is_listening() is False

        provider.stop()

    def test_resume(self, default_config):
        """Test resuming STT provider."""
        provider = STTProvider(**default_config)
        provider.start()
        provider.pause()

        assert provider.is_listening() is False
        provider.resume()
        assert provider.is_listening() is True

        provider.stop()

    def test_pause_idempotent(self, default_config):
        """Test calling pause twice is safe."""
        provider = STTProvider(**default_config)
        provider.start()

        provider.pause()
        provider.pause()
        assert provider.is_listening() is False

        provider.stop()

    def test_resume_idempotent(self, default_config):
        """Test calling resume twice is safe."""
        provider = STTProvider(**default_config)
        provider.start()

        provider.resume()
        provider.resume()
        assert provider.is_listening() is True

        provider.stop()

    def test_send_audio_dropped_when_paused(self, default_config):
        """Test audio is dropped when paused."""
        provider = STTProvider(**default_config)
        provider.start()
        provider.pause()

        # send_audio should silently return without queuing
        provider.send_audio(b"\x00\x00" * 100)

        if provider._audio_queue is not None:
            assert provider._audio_queue.qsize() == 0

        provider.stop()


class TestSTTProviderAudioSending:
    """Test STTProvider audio sending."""

    def test_send_audio_not_running(self, default_config, caplog):
        """Test sending audio when not running logs warning."""
        provider = STTProvider(**default_config)

        provider.send_audio(b"audio_data")

        assert "not running" in caplog.text

    def test_end_utterance_not_running(self, default_config):
        """Test end utterance when not running does nothing."""
        provider = STTProvider(**default_config)

        # Should not raise
        provider.end_utterance()
