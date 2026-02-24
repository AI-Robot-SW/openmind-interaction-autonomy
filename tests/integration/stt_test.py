import logging
import os
import signal
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from providers.audio_provider import AudioProvider
from providers.stt_provider import STTBackend, STTProvider

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stop_event = threading.Event()

    def handle_sigint(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    stt_provider = STTProvider(backend=STTBackend.GOOGLE_CLOUD)

    def on_final(text: str) -> None:
        logging.info("FINAL: %s", text)

    def on_interim(text: str) -> None:
        logging.info("INTERIM: %s", text)

    stt_provider.register_result_callback(on_final)
    stt_provider.register_interim_callback(on_interim)

    audio_provider = AudioProvider()

    def on_audio(
        audio_chunk: bytes, frame_count: int, time_info: dict, status_flags: int
    ) -> None:
        stt_provider.send_audio(audio_chunk)

    audio_provider.register_audio_callback(on_audio)

    logging.info("Starting STTProvider (Google Cloud)...")
    stt_provider.start()
    logging.info("Starting AudioProvider...")
    audio_provider.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    finally:
        logging.info("Stopping AudioProvider...")
        audio_provider.stop()
        logging.info("Stopping STTProvider...")
        stt_provider.stop()


if __name__ == "__main__":
    main()
