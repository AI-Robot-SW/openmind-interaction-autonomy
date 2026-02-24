"""
Tests for SpeakConnector.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# PyAudio 모킹
mock_pyaudio = MagicMock()
mock_pyaudio.PyAudio = MagicMock()
mock_pyaudio.paInt16 = 8
sys.modules["pyaudio"] = mock_pyaudio

from actions.speak.connector.speak_connector import SpeakConnector, SpeakConnectorConfig
from actions.speak.interface import SpeakInput
from providers.speaker_provider import SpeakerProvider
from providers.tts_provider import TTSBackend, TTSProvider


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    TTSProvider.reset()  # type: ignore
    SpeakerProvider.reset()  # type: ignore
    yield
    TTSProvider.reset()  # type: ignore
    SpeakerProvider.reset()  # type: ignore


@pytest.fixture
def setup_providers():
    """Setup TTS and Speaker providers for testing."""
    with patch("providers.speaker_provider.pyaudio") as mock_pa:
        mock_pa_instance = MagicMock()
        mock_pa.PyAudio.return_value = mock_pa_instance
        mock_pa.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        speaker = SpeakerProvider(sample_rate=24000)
        speaker.start()

        tts = TTSProvider(
            backend=TTSBackend.NAVER_CLOVA,
            naver_client_id="test_id",
            naver_client_secret="test_secret",
        )
        tts.register_audio_callback(speaker.queue_audio)
        tts.start()

        yield tts, speaker

        tts.stop()
        speaker.stop()


class TestSpeakConnectorInitialization:
    """Test SpeakConnector initialization."""

    @patch("providers.speaker_provider.pyaudio")
    def test_default_initialization(self, mock_pa):
        """Test SpeakConnector initializes with default config."""
        mock_pa_instance = MagicMock()
        mock_pa.PyAudio.return_value = mock_pa_instance
        mock_pa.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        SpeakerProvider(sample_rate=24000)
        TTSProvider(
            backend=TTSBackend.NAVER_CLOVA,
            naver_client_id="test_id",
            naver_client_secret="test_secret",
        )

        config = SpeakConnectorConfig()
        connector = SpeakConnector(config)

        assert connector.tts_enabled is True
        assert connector.config.enable_tts_interrupt is True


class TestSpeakConnectorConnect:
    """Test SpeakConnector connect method."""

    @pytest.mark.asyncio
    async def test_connect_queues_text(self, setup_providers):
        """Test connect adds text to TTS queue."""
        tts, speaker = setup_providers

        config = SpeakConnectorConfig()
        connector = SpeakConnector(config)

        speak_input = SpeakInput(action="안녕하세요")
        await connector.connect(speak_input)

        assert tts.get_pending_message_count() >= 1

    @pytest.mark.asyncio
    async def test_connect_empty_text_skipped(self, setup_providers):
        """Test connect skips empty text."""
        tts, speaker = setup_providers

        config = SpeakConnectorConfig()
        connector = SpeakConnector(config)

        speak_input = SpeakInput(action="")
        await connector.connect(speak_input)

        assert tts.get_pending_message_count() == 0

    @pytest.mark.asyncio
    async def test_connect_disabled_skipped(self, setup_providers):
        """Test connect skips when TTS is disabled."""
        tts, speaker = setup_providers

        config = SpeakConnectorConfig()
        connector = SpeakConnector(config)
        connector.tts_enabled = False

        speak_input = SpeakInput(action="테스트")
        await connector.connect(speak_input)

        assert tts.get_pending_message_count() == 0

    @pytest.mark.asyncio
    async def test_connect_with_interrupt(self, setup_providers):
        """Test connect calls stop_playback when interrupt is True."""
        tts, speaker = setup_providers

        config = SpeakConnectorConfig(enable_tts_interrupt=True)
        connector = SpeakConnector(config)

        speak_input = SpeakInput(action="테스트", interrupt=True)
        await connector.connect(speak_input)

        # Should have queued the message
        assert tts.get_pending_message_count() >= 1
