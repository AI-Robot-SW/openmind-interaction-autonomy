# audio.py - Tests for audio provider and background plugin
import logging
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
PROVIDERS_ROOT = os.path.join(SRC_ROOT, "providers")
if PROVIDERS_ROOT not in sys.path:
    sys.path.insert(0, PROVIDERS_ROOT)

from backgrounds.plugins.audio_bg import AudioBg, AudioBgConfig
from providers.audio_provider import AudioProvider


def test_audio_bg_bootstrap_and_start_stream() -> None:
    """
    Verify this sequence:
    1) AudioBg creates AudioProvider instance
    2) AudioProvider initializes PyAudio interface + audio buffer
    3) start_stream() opens real PyAudio stream and provider enters running state
    """
    AudioProvider.reset()  # type: ignore[attr-defined]
    provider = None
    try:
        cfg = AudioBgConfig(
            sample_rate=16000,
            chunk_size=1024,
            channels=1,
            vad_enabled=False,
        )
        bg = AudioBg(cfg)

        # 1) AudioBg -> AudioProvider instance
        provider = bg.audio_provider
        assert provider is AudioProvider()
        assert provider.sample_rate == cfg.sample_rate
        assert provider.chunk_size == cfg.chunk_size
        assert provider.channels == cfg.channels

        # 2) AudioProvider init: PyAudio interface + audio buffer initialized
        stats = provider.get_buffer_stats()
        chunk_ms = (provider.chunk_size / provider.sample_rate) * 1000.0
        expected_max = max(
            1, math.ceil(provider.buffer_duration_ms / chunk_ms)
        )
        assert int(stats["max_chunks"]) == expected_max
        assert int(stats["queued_chunks"]) == 0
        assert int(stats["drops"]) == 0

        # 3) start_stream() path: real stream opened and provider running
        assert provider.running is True
        assert provider._audio_stream.running is True
        assert provider._audio_stream.rate == cfg.sample_rate
        assert provider._audio_stream.chunk == cfg.chunk_size
        assert provider._audio_stream._stream is not None

        # Explicitly verify start_stream() can reopen after stop.
        provider.stop()
        assert provider.running is False
        assert provider._audio_stream.running is False
        assert provider._audio_stream._stream is None

        provider.start_stream()
        assert provider.running is True
        assert provider._audio_stream.running is True
        assert provider._audio_stream._stream is not None

        provider.stop()
    finally:
        if provider is not None and provider.running:
            provider.stop()

    AudioProvider.reset()  # type: ignore[attr-defined]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    test_audio_bg_bootstrap_and_start_stream()
    logging.info("PASS: audio bg/provider bootstrap and start_stream sequence verified.")


if __name__ == "__main__":
    main()
