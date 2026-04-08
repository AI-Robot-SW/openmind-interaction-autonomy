import asyncio
import logging
import threading
import time

from backgrounds.base import Background
from runtime.multi_mode.config import RuntimeConfig


class BackgroundOrchestrator:
    """
    Manages the background tasks for the application.
    """

    _config: RuntimeConfig
    _threads: list[threading.Thread]
    _submitted_backgrounds: set[str]
    _stop_event: threading.Event

    def __init__(self, config: RuntimeConfig):
        """
        Initialize the BackgroundOrchestrator with the provided configuration.

        Parameters
        ----------
        config : RuntimeConfig
            Configuration object for the runtime.
        """
        self._config = config
        self._threads = []
        self._submitted_backgrounds = set()
        self._stop_event = threading.Event()

    def start(self):
        """
        Start background tasks in separate daemon threads.
        Injects _orchestrator_stop_event into each background so run() can exit when stopping.
        """
        for background in self._config.backgrounds:
            if background.name in self._submitted_backgrounds:
                logging.warning(
                    f"Background {background.name} already submitted, skipping."
                )
                continue
            background._orchestrator_stop_event = self._stop_event
            thread = threading.Thread(
                target=self._run_background_loop,
                args=(background,),
                name=f"bg-worker-{background.name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
            self._submitted_backgrounds.add(background.name)

        return asyncio.Future()

    def _run_background_loop(self, background: Background):
        """
        Thread-based background loop.
        Calls run() until stop is requested.

        Parameters
        ----------
        background : Background
            The background task to run.
        """
        while True:
            try:
                background.run()
            except Exception as e:
                logging.error(f"Error in background {background.name}: {e}")
                time.sleep(0.1)
            if self._stop_event.is_set():
                break

    def stop(self, timeout: float = 3.0):
        """
        Stop all background threads.
        Signals threads to stop and waits up to `timeout` seconds for them to finish.
        Threads are daemon threads, so they won't block process exit regardless.
        """
        self._stop_event.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        still_running = [t.name for t in self._threads if t.is_alive()]
        if still_running:
            logging.warning(
                f"BackgroundOrchestrator: threads still running after timeout: {still_running}"
            )
        logging.info("BackgroundOrchestrator stopped")

    def __del__(self):
        """
        Clean up the BackgroundOrchestrator by stopping threads.
        """
        self.stop()
