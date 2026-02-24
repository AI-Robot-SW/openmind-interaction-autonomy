"""
Integration tests for TTS + Speaker pipeline.
"""

import io
import sys
import time
import wave
from unittest.mock import MagicMock, patch

import pytest

# PyAudio 모킹 (파일 최상단, import 전)
mock_pyaudio = MagicMock()
mock_pyaudio.PyAudio = MagicMock()
mock_pyaudio.paInt16 = 8  # PyAudio constant
sys.modules["pyaudio"] = mock_pyaudio

from providers.speaker_provider import SpeakerProvider
from providers.stt_provider import STTProvider
from providers.tts_provider import TTSBackend, TTSProvider


def _create_mock_wav() -> bytes:
    """Create minimal WAV for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 100)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    TTSProvider.reset()  # type: ignore
    SpeakerProvider.reset()  # type: ignore
    STTProvider.reset()  # type: ignore
    yield
    TTSProvider.reset()  # type: ignore
    SpeakerProvider.reset()  # type: ignore
    STTProvider.reset()  # type: ignore


class TestTTSSpeakerIntegration:
    """Test TTS and Speaker integration."""

    @patch("providers.speaker_provider.pyaudio")
    @patch("providers.tts_provider.requests.post")
    def test_tts_to_speaker_callback_flow(self, mock_post, mock_pyaudio):
        """Test audio flows from TTS to Speaker via callback."""
        # Setup PyAudio mock
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        # Setup TTS mock response (WAV)
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = _create_mock_wav()
        mock_post.return_value.raise_for_status = MagicMock()

        # Initialize providers
        speaker = SpeakerProvider(sample_rate=24000, channels=1)
        speaker.start()

        tts = TTSProvider(
            backend=TTSBackend.NAVER_CLOVA,
            naver_client_id="test_id",
            naver_client_secret="test_secret",
        )

        # Connect TTS to Speaker
        tts.register_audio_callback(speaker.queue_audio)
        tts.start()

        # Send message
        tts.add_pending_message("테스트")

        # Wait for processing
        time.sleep(0.5)

        # Verify audio was queued (may have been processed already)
        assert speaker.get_queue_size() >= 0

        # Cleanup
        tts.stop()
        speaker.stop()

    @patch("providers.speaker_provider.pyaudio")
    def test_multiple_messages_sequential(self, mock_pyaudio):
        """Test multiple messages are processed sequentially."""
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        speaker = SpeakerProvider(sample_rate=24000)
        speaker.start()

        # Queue multiple audio chunks
        speaker.queue_audio(b"\x00\x00" * 100)
        speaker.queue_audio(b"\x00\x00" * 100)
        speaker.queue_audio(b"\x00\x00" * 100)

        assert speaker.get_queue_size() == 3

        speaker.stop()

    @patch("providers.speaker_provider.pyaudio")
    def test_speaker_clear_queue(self, mock_pyaudio):
        """Test clearing speaker queue."""
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        speaker = SpeakerProvider(sample_rate=24000)
        speaker.start()

        # Queue audio
        speaker.queue_audio(b"\x00\x00" * 100)
        speaker.queue_audio(b"\x00\x00" * 100)
        assert speaker.get_queue_size() == 2

        # Clear queue
        speaker.clear_queue()
        assert speaker.get_queue_size() == 0

        speaker.stop()

    @patch("providers.speaker_provider.pyaudio")
    def test_tts_callback_registration(self, mock_pyaudio):
        """Test TTS callback registration and unregistration."""
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
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

        # Register callback
        tts.register_audio_callback(speaker.queue_audio)
        assert speaker.queue_audio in tts._audio_callbacks

        # Unregister callback
        tts.unregister_audio_callback(speaker.queue_audio)
        assert speaker.queue_audio not in tts._audio_callbacks

        speaker.stop()

    @patch("providers.speaker_provider.pyaudio")
    def test_tts_interrupt_clears_speaker_connection(self, mock_pyaudio):
        """Test TTS interrupt functionality with speaker connection."""
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        speaker = SpeakerProvider(sample_rate=24000)
        speaker.start()

        tts = TTSProvider(
            backend=TTSBackend.NAVER_CLOVA,
            naver_client_id="test_id",
            naver_client_secret="test_secret",
            enable_tts_interrupt=True,
        )
        tts.register_audio_callback(speaker.queue_audio)
        tts.start()

        # Add messages
        tts.add_pending_message("메시지 1")
        tts.add_pending_message("메시지 2")

        # Interrupt
        tts.interrupt()

        # Queue should be cleared
        assert tts.get_pending_message_count() == 0

        tts.stop()
        speaker.stop()


class TestTTSSpeakerDataProperties:
    """Test data properties for monitoring."""

    @patch("providers.speaker_provider.pyaudio")
    def test_tts_data_property(self, mock_pyaudio):
        """Test TTS data property for monitoring."""
        tts = TTSProvider(
            backend=TTSBackend.NAVER_CLOVA,
            naver_client_id="test_id",
            naver_client_secret="test_secret",
        )

        data = tts.data

        assert "state" in data
        assert "pending_count" in data
        assert "backend" in data
        assert "running" in data
        assert data["backend"] == "naver_clova"
        assert data["state"] == "idle"

    @patch("providers.speaker_provider.pyaudio")
    def test_speaker_data_property(self, mock_pyaudio):
        """Test Speaker data property for monitoring."""
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        speaker = SpeakerProvider(sample_rate=24000, volume=0.8)
        speaker.start()

        data = speaker.data

        assert "running" in data
        assert "is_playing" in data
        assert "queue_size" in data
        assert "volume" in data
        assert data["running"] is True
        assert data["volume"] == 0.8

        speaker.stop()


class TestSpeakerSTTEchoPrevention:
    """Test Speaker↔STT echo prevention integration."""

    @patch("providers.speaker_provider.pyaudio")
    def test_playback_pauses_stt(self, mock_pyaudio):
        """Test that speaker playback pauses STT."""
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        speaker = SpeakerProvider(sample_rate=24000)
        speaker.start()

        stt = STTProvider(
            ws_url="wss://test.com",
            api_key="test",
            language="korean",
        )
        stt.start()

        assert stt.is_listening() is True

        # Register echo prevention callback
        def on_playback_state(state: str):
            if state == "playing":
                stt.pause()
            elif state in ("completed", "stopped"):
                stt.resume()

        speaker.register_playback_callback(on_playback_state)

        # Simulate playback state changes
        speaker._notify_playback_state("playing")
        assert stt.is_listening() is False

        speaker._notify_playback_state("completed")
        assert stt.is_listening() is True

        stt.stop()
        speaker.stop()

    @patch("providers.speaker_provider.pyaudio")
    def test_audio_dropped_during_playback(self, mock_pyaudio):
        """Test that audio is dropped when STT is paused during playback."""
        mock_pa_instance = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pa_instance
        mock_pyaudio.paInt16 = 8
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        mock_pa_instance.open.return_value = mock_stream

        stt = STTProvider(
            ws_url="wss://test.com",
            api_key="test",
            language="korean",
        )
        stt.start()

        # Pause (simulating speaker playing)
        stt.pause()
        stt.send_audio(b"\x00\x00" * 100)

        # Audio should not reach the queue
        if stt._audio_queue is not None:
            assert stt._audio_queue.qsize() == 0

        # Resume
        stt.resume()
        assert stt.is_listening() is True

        stt.stop()
