"""
Tests for TTSProvider.
"""

import io
import wave
from unittest.mock import MagicMock, patch

import pytest
import requests

from providers.tts_provider import TTSBackend, TTSProvider, TTSState


def _create_mock_wav_data() -> bytes:
    """Create minimal valid WAV data for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 100)  # 100 samples of silence
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    TTSProvider.reset()  # type: ignore
    yield
    TTSProvider.reset()  # type: ignore


@pytest.fixture
def default_config():
    """Default configuration for TTSProvider."""
    return {
        "url": "https://api.openmind.org/api/core/elevenlabs/tts",
        "api_key": "test_api_key",
        "backend_api_key": None,
        "backend": TTSBackend.ELEVENLABS,
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model_id": "eleven_flash_v2_5",
        "output_format": "mp3_44100_128",
        "language": "ko",
        "enable_tts_interrupt": False,
    }


@pytest.fixture
def custom_config():
    """Custom configuration for TTSProvider."""
    return {
        "url": "https://custom.api.com/tts",
        "api_key": "custom_api_key",
        "backend_api_key": "elevenlabs_key",
        "backend": TTSBackend.ELEVENLABS,
        "voice_id": "custom_voice",
        "model_id": "custom_model",
        "output_format": "pcm_44100",
        "language": "en",
        "enable_tts_interrupt": True,
    }


@pytest.fixture
def naver_clova_config():
    """Naver Clova configuration for TTSProvider."""
    return {
        "backend": TTSBackend.NAVER_CLOVA,
        "naver_client_id": "test_client_id",
        "naver_client_secret": "test_client_secret",
        "speaker": "nara",
        "output_format": "wav",
        "sampling_rate": 24000,
        "volume": 0,
        "speed": 0,
        "pitch": 0,
    }


class TestTTSBackendEnum:
    """Test TTSBackend enum."""

    def test_elevenlabs_value(self):
        """Test ElevenLabs backend value."""
        assert TTSBackend.ELEVENLABS.value == "elevenlabs"

    def test_riva_value(self):
        """Test Riva backend value."""
        assert TTSBackend.RIVA.value == "riva"

    def test_google_value(self):
        """Test Google backend value."""
        assert TTSBackend.GOOGLE.value == "google"

    def test_local_value(self):
        """Test Local backend value."""
        assert TTSBackend.LOCAL.value == "local"

    def test_naver_clova_value(self):
        """Test Naver Clova backend value."""
        assert TTSBackend.NAVER_CLOVA.value == "naver_clova"


class TestTTSStateEnum:
    """Test TTSState enum."""

    def test_idle_value(self):
        """Test idle state value."""
        assert TTSState.IDLE.value == "idle"

    def test_processing_value(self):
        """Test processing state value."""
        assert TTSState.PROCESSING.value == "processing"

    def test_speaking_value(self):
        """Test speaking state value."""
        assert TTSState.SPEAKING.value == "speaking"

    def test_error_value(self):
        """Test error state value."""
        assert TTSState.ERROR.value == "error"


class TestTTSProviderInitialization:
    """Test TTSProvider initialization."""

    def test_default_initialization(self, default_config):
        """Test TTSProvider initialization with default values."""
        provider = TTSProvider(**default_config)

        assert provider.url == "https://api.openmind.org/api/core/elevenlabs/tts"
        assert provider.api_key == "test_api_key"
        assert provider.backend == TTSBackend.ELEVENLABS
        assert provider.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert provider.model_id == "eleven_flash_v2_5"
        assert provider._output_format == "mp3_44100_128"
        assert provider._language == "ko"
        assert not provider.running

    def test_custom_initialization(self, custom_config):
        """Test TTSProvider initialization with custom values."""
        provider = TTSProvider(**custom_config)

        assert provider.url == "https://custom.api.com/tts"
        assert provider.backend_api_key == "elevenlabs_key"
        assert provider.voice_id == "custom_voice"
        assert provider.model_id == "custom_model"
        assert provider._enable_tts_interrupt is True

    def test_singleton_pattern(self, default_config):
        """Test that TTSProvider follows singleton pattern."""
        provider1 = TTSProvider(**default_config)
        provider2 = TTSProvider(**default_config)

        assert provider1 is provider2


class TestTTSProviderProperties:
    """Test TTSProvider properties."""

    def test_voice_id_getter(self, default_config):
        """Test voice_id getter."""
        provider = TTSProvider(**default_config)

        assert provider.voice_id == "JBFqnCBsd6RMkjVDRZzb"

    def test_voice_id_setter(self, default_config):
        """Test voice_id setter."""
        provider = TTSProvider(**default_config)

        provider.voice_id = "new_voice"

        assert provider.voice_id == "new_voice"

    def test_model_id_getter(self, default_config):
        """Test model_id getter."""
        provider = TTSProvider(**default_config)

        assert provider.model_id == "eleven_flash_v2_5"

    def test_model_id_setter(self, default_config):
        """Test model_id setter."""
        provider = TTSProvider(**default_config)

        provider.model_id = "new_model"

        assert provider.model_id == "new_model"


class TestTTSProviderCallbacks:
    """Test TTSProvider callback registration."""

    def test_register_tts_state_callback(self, default_config):
        """Test registering TTS state callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock(__name__="test_callback")

        provider.register_tts_state_callback(callback)

        assert callback in provider._state_callbacks

    def test_unregister_tts_state_callback(self, default_config):
        """Test unregistering TTS state callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock(__name__="test_callback")

        provider.register_tts_state_callback(callback)
        provider.unregister_tts_state_callback(callback)

        assert callback not in provider._state_callbacks

    def test_register_audio_callback(self, default_config):
        """Test registering audio callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock(__name__="test_callback")

        provider.register_audio_callback(callback)

        assert callback in provider._audio_callbacks

    def test_unregister_audio_callback(self, default_config):
        """Test unregistering audio callback."""
        provider = TTSProvider(**default_config)
        callback = MagicMock(__name__="test_callback")

        provider.register_audio_callback(callback)
        provider.unregister_audio_callback(callback)

        assert callback not in provider._audio_callbacks


class TestTTSProviderMessage:
    """Test TTSProvider message creation."""

    def test_create_pending_message(self, default_config):
        """Test creating pending message."""
        provider = TTSProvider(**default_config)

        message = provider.create_pending_message("안녕하세요")

        assert message["text"] == "안녕하세요"
        assert message["voice_id"] == "JBFqnCBsd6RMkjVDRZzb"
        assert message["model_id"] == "eleven_flash_v2_5"
        assert message["output_format"] == "mp3_44100_128"
        assert message["language"] == "ko"

    def test_create_pending_message_with_backend_key(self, custom_config):
        """Test creating pending message includes backend API key."""
        provider = TTSProvider(**custom_config)

        message = provider.create_pending_message("Hello")

        assert message["backend_api_key"] == "elevenlabs_key"

    def test_add_pending_message_not_running(self, default_config, caplog):
        """Test adding message when not running logs warning."""
        provider = TTSProvider(**default_config)

        provider.add_pending_message("test")

        assert "not running" in caplog.text

    def test_add_pending_message_string(self, default_config):
        """Test adding string message."""
        provider = TTSProvider(**default_config)
        provider.start()

        provider.add_pending_message("안녕하세요")

        assert provider.get_pending_message_count() == 1

        provider.stop()

    def test_add_pending_message_dict(self, default_config):
        """Test adding dict message."""
        provider = TTSProvider(**default_config)
        provider.start()

        message = {"text": "test", "voice_id": "custom"}
        provider.add_pending_message(message)

        assert provider.get_pending_message_count() == 1

        provider.stop()


class TestTTSProviderState:
    """Test TTSProvider state methods."""

    def test_get_current_state_initial(self, default_config):
        """Test initial state is IDLE."""
        provider = TTSProvider(**default_config)

        assert provider.get_current_state() == TTSState.IDLE

    def test_get_pending_message_count_empty(self, default_config):
        """Test pending message count when empty."""
        provider = TTSProvider(**default_config)

        assert provider.get_pending_message_count() == 0


class TestTTSProviderLifecycle:
    """Test TTSProvider start/stop lifecycle."""

    def test_start(self, default_config):
        """Test starting the provider."""
        provider = TTSProvider(**default_config)
        provider.start()

        assert provider.running is True

        provider.stop()

    def test_start_already_running(self, default_config, caplog):
        """Test starting already running provider logs warning."""
        provider = TTSProvider(**default_config)
        provider.start()
        provider.start()

        assert "already running" in caplog.text

        provider.stop()

    def test_stop(self, default_config):
        """Test stopping the provider."""
        provider = TTSProvider(**default_config)
        provider.start()
        provider.stop()

        assert provider.running is False

    def test_stop_not_running(self, default_config, caplog):
        """Test stopping not running provider logs warning."""
        provider = TTSProvider(**default_config)
        provider.stop()

        assert "not running" in caplog.text


class TestTTSProviderInterrupt:
    """Test TTSProvider interrupt functionality."""

    def test_interrupt_not_enabled(self, default_config, caplog):
        """Test interrupt when not enabled logs warning."""
        provider = TTSProvider(**default_config)

        provider.interrupt()

        assert "not enabled" in caplog.text

    def test_interrupt_clears_queue(self, custom_config):
        """Test interrupt clears pending queue."""
        provider = TTSProvider(**custom_config)
        provider.start()
        provider.add_pending_message("test1")
        provider.add_pending_message("test2")

        provider.interrupt()

        assert provider.get_pending_message_count() == 0
        assert provider.get_current_state() == TTSState.IDLE

        provider.stop()


class TestTTSProviderConfiguration:
    """Test TTSProvider configuration methods."""

    def test_configure_voice_id(self, default_config):
        """Test configuring voice ID."""
        provider = TTSProvider(**default_config)

        provider.configure(voice_id="new_voice")

        assert provider.voice_id == "new_voice"

    def test_configure_model_id(self, default_config):
        """Test configuring model ID."""
        provider = TTSProvider(**default_config)

        provider.configure(model_id="new_model")

        assert provider.model_id == "new_model"

    def test_configure_language(self, default_config):
        """Test configuring language."""
        provider = TTSProvider(**default_config)

        provider.configure(language="en")

        assert provider._language == "en"

    def test_configure_output_format(self, default_config):
        """Test configuring output format."""
        provider = TTSProvider(**default_config)

        provider.configure(output_format="pcm_44100")

        assert provider._output_format == "pcm_44100"


class TestTTSProviderNaverClova:
    """Test TTSProvider Naver Clova specific functionality."""

    def test_naver_clova_initialization(self, naver_clova_config):
        """Test TTSProvider initialization with Naver Clova config."""
        provider = TTSProvider(**naver_clova_config)

        assert provider.backend == TTSBackend.NAVER_CLOVA
        assert provider._naver_client_id == "test_client_id"
        assert provider._naver_client_secret == "test_client_secret"
        assert provider._speaker == "nara"
        assert provider._output_format == "wav"
        assert provider._sampling_rate == 24000

    def test_create_naver_clova_message(self, naver_clova_config):
        """Test creating Naver Clova message."""
        provider = TTSProvider(**naver_clova_config)

        message = provider.create_pending_message("안녕하세요")

        assert message["text"] == "안녕하세요"
        assert message["speaker"] == "nara"
        assert message["format"] == "wav"
        assert message["sampling-rate"] == 24000

    def test_create_naver_clova_message_with_emotion(self, naver_clova_config):
        """Test creating Naver Clova message with emotion."""
        # Use a Pro speaker that supports emotion (vara, vmikyung, etc.)
        naver_clova_config["speaker"] = "vara"
        provider = TTSProvider(**naver_clova_config)
        provider._emotion = 1
        provider._emotion_strength = 2

        message = provider.create_pending_message("기쁜 소식입니다")

        assert message["emotion"] == 1
        assert message["emotion-strength"] == 2


class TestTTSProviderSynthesis:
    """Test TTSProvider synthesis methods."""

    @patch("providers.tts_provider.requests.post")
    def test_synthesize_naver_clova_success(self, mock_post, naver_clova_config):
        """Test successful Naver Clova synthesis."""
        mock_wav = _create_mock_wav_data()
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = mock_wav
        mock_post.return_value.raise_for_status = MagicMock()

        provider = TTSProvider(**naver_clova_config)

        result = provider._synthesize_naver_clova({"text": "테스트"})

        assert result is not None
        mock_post.assert_called_once()

    @patch("providers.tts_provider.requests.post")
    def test_synthesize_naver_clova_failure(self, mock_post, naver_clova_config):
        """Test Naver Clova synthesis failure."""
        mock_post.side_effect = requests.RequestException("API Error")

        provider = TTSProvider(**naver_clova_config)

        result = provider._synthesize_naver_clova({"text": "테스트"})

        assert result is None

    def test_synthesize_naver_clova_no_credentials(self):
        """Test synthesis without credentials."""
        provider = TTSProvider(backend=TTSBackend.NAVER_CLOVA)

        result = provider._synthesize_naver_clova({"text": "테스트"})

        assert result is None

    @patch("providers.tts_provider.requests.post")
    def test_synthesize_naver_clova_headers(self, mock_post, naver_clova_config):
        """Test Naver Clova synthesis sends correct headers."""
        mock_wav = _create_mock_wav_data()
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = mock_wav
        mock_post.return_value.raise_for_status = MagicMock()

        provider = TTSProvider(**naver_clova_config)
        provider._synthesize_naver_clova({"text": "테스트"})

        call_kwargs = mock_post.call_args
        headers = call_kwargs[1]["headers"]
        assert headers["X-NCP-APIGW-API-KEY-ID"] == "test_client_id"
        assert headers["X-NCP-APIGW-API-KEY"] == "test_client_secret"
        assert headers["Content-Type"] == "application/x-www-form-urlencoded"


class TestTTSProviderData:
    """Test TTSProvider data property."""

    def test_data_property_elevenlabs(self, default_config):
        """Test data property returns correct structure for ElevenLabs."""
        provider = TTSProvider(**default_config)

        data = provider.data

        assert "state" in data
        assert "pending_count" in data
        assert "backend" in data
        assert "running" in data
        assert data["state"] == "idle"
        assert data["backend"] == "elevenlabs"

    def test_data_property_naver_clova(self, naver_clova_config):
        """Test data property returns correct structure for Naver Clova."""
        provider = TTSProvider(**naver_clova_config)

        data = provider.data

        assert "state" in data
        assert "pending_count" in data
        assert "backend" in data
        assert "speaker" in data
        assert "running" in data
        assert data["state"] == "idle"
        assert data["backend"] == "naver_clova"
        assert data["speaker"] == "nara"


class TestTTSProviderPCMExtraction:
    """Test TTSProvider PCM extraction."""

    def test_extract_pcm_from_wav(self, naver_clova_config):
        """Test extracting PCM from WAV data."""
        provider = TTSProvider(**naver_clova_config)
        wav_data = _create_mock_wav_data()

        pcm_data = provider._extract_pcm_from_wav(wav_data)

        assert pcm_data is not None
        assert len(pcm_data) == 200  # 100 samples * 2 bytes

    def test_extract_pcm_from_invalid_wav(self, naver_clova_config):
        """Test extracting PCM from invalid WAV data."""
        provider = TTSProvider(**naver_clova_config)

        result = provider._extract_pcm_from_wav(b"invalid data")

        # 실패 시 원본 반환
        assert result == b"invalid data"
