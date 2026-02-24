import argparse
import logging
import os
import signal
import sys
import threading
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
PROVIDERS_ROOT = os.path.join(SRC_ROOT, "providers")
if PROVIDERS_ROOT not in sys.path:
    sys.path.insert(0, PROVIDERS_ROOT)

from providers.audio_provider import AudioProvider
from providers.stt_provider import STTBackend, STTProvider
from backgrounds.plugins.stt_bg import STTBg, STTBgConfig
from backgrounds.plugins.audio_bg import AudioBg, AudioBgConfig


def _format_monitor_line(audio_provider: AudioProvider, stt_provider: STTProvider) -> str:
    audio_stats = audio_provider.get_buffer_stats()
    stt_queue_size = stt_provider._audio_queue.qsize() if stt_provider._audio_queue else 0
    stt_stream_alive = (
        stt_provider._streaming_thread.is_alive()
        if stt_provider._streaming_thread is not None
        else False
    )
    stt_consumer_alive = (
        stt_provider._audio_consumer_thread.is_alive()
        if stt_provider._audio_consumer_thread is not None
        else False
    )
    return (
        "MONITOR "
        f"audio_running={audio_provider.running} "
        f"audio_buffer={int(audio_stats['queued_chunks'])}/{int(audio_stats['max_chunks'])} "
        f"audio_drops={int(audio_stats['drops'])} "
        f"audio_latency_ms={audio_stats['approx_latency_ms']:.1f} "
        f"stt_running={stt_provider.running} "
        f"stt_listening={stt_provider.is_listening()} "
        f"stt_queue={stt_queue_size} "
        f"stt_sent={stt_provider._audio_sent} "
        f"stt_consumed={stt_provider._audio_consumed} "
        f"stt_drops={stt_provider._audio_drops} "
        f"stt_stream_alive={stt_stream_alive} "
        f"stt_consumer_alive={stt_consumer_alive}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio/STT integration smoke test for provider and background wiring."
    )
    parser.add_argument(
        "--mode",
        choices=["provider", "bg"],
        default="bg",
        help="provider: 직접 provider 연결 / bg: AudioBg+STTBg 경로 사용",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=2.0,
        help="모니터링 로그 출력 주기(초)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop_event = threading.Event()

    def handle_sigint(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    stt_provider: STTProvider
    audio_provider: AudioProvider

    def on_final(text: str) -> None:
        logging.info("FINAL: %s", text)

    def on_interim(text: str) -> None:
        logging.info("INTERIM: %s", text)

    if args.mode == "provider":
        stt_provider = STTProvider(backend=STTBackend.GOOGLE_CLOUD)
        stt_provider.register_result_callback(on_final)
        stt_provider.register_interim_callback(on_interim)

        audio_provider = AudioProvider()
        stt_provider.attach_audio_provider(audio_provider)

        logging.info("Starting STTProvider (Google Cloud) in provider mode...")
        stt_provider.start()
        logging.info("Starting AudioProvider stream...")
        audio_provider.start_stream()
    else:
        audio_bg = AudioBg(
            AudioBgConfig(
                sample_rate=16000,
                chunk_size=1024,
                channels=1,
            )
        )
        stt_bg = STTBg(
            STTBgConfig(
                backend="google_cloud",
                language="korean",
                sample_rate=16000,
                enable_interim_results=True,
            )
        )
        _ = audio_bg  # keep reference for lifecycle clarity
        _ = stt_bg

        # singleton accessor: same instances used by STTBg
        stt_provider = STTProvider()
        audio_provider = AudioProvider()
        stt_provider.register_result_callback(on_final)
        stt_provider.register_interim_callback(on_interim)

        logging.info("Started STTBg in bg mode.")

    try:
        last_monitor_ts = 0.0
        while not stop_event.is_set():
            time.sleep(0.1)
            now = time.time()
            if now - last_monitor_ts >= args.monitor_interval:
                logging.info(_format_monitor_line(audio_provider, stt_provider))
                last_monitor_ts = now
    finally:
        logging.info("Stopping AudioProvider...")
        audio_provider.stop()
        logging.info("Stopping STTProvider...")
        stt_provider.stop()


if __name__ == "__main__":
    main()
