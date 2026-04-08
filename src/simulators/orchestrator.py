import asyncio
import logging
import threading
import time
import typing as T

from llm.output_model import Action
from runtime.single_mode.config import RuntimeConfig
from simulators.base import Simulator


class SimulatorOrchestrator:
    """
    Manages data flow to one or more simulators.
    Note: It is important that the simulators do not block the event loop.
    """

    promise_queue: T.List[asyncio.Task[T.Any]]
    _config: RuntimeConfig
    _threads: list[threading.Thread]
    _submitted_simulators: T.Set[str]
    _stop_event: threading.Event

    def __init__(self, config: RuntimeConfig):
        self._config = config
        self.promise_queue = []
        self._threads = []
        self._submitted_simulators = set()
        self._stop_event = threading.Event()

    def start(self):
        """
        Start simulators in separate threads
        """
        for simulator in self._config.simulators:
            if simulator.name in self._submitted_simulators:
                logging.warning(
                    f"Simulator {simulator.name} already submitted, skipping."
                )
                continue
            thread = threading.Thread(
                target=self._run_simulator_loop,
                args=(simulator,),
                name=f"sim-worker-{simulator.name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
            self._submitted_simulators.add(simulator.name)

        return asyncio.Future()

    def _run_simulator_loop(self, simulator: Simulator):
        """
        Thread-based simulator loop

        Parameters
        ----------
        simulator : Simulator
            The simulator to run
        """
        while not self._stop_event.is_set():
            try:
                simulator.tick()
            except Exception as e:
                logging.error(f"Error in simulator {simulator.name}: {e}")

    async def flush_promises(self) -> tuple[list[T.Any], list[asyncio.Task[T.Any]]]:
        """
        Flushes the promise queue and returns the completed promises
        and the pending promises.

        Returns
        -------
        tuple[list[Any], list[asyncio.Task[Any]]]
            A tuple containing the completed promises and the pending promises
        """
        done_promises = []
        for promise in self.promise_queue:
            if promise.done():
                await promise
                done_promises.append(promise)
        self.promise_queue = [p for p in self.promise_queue if p not in done_promises]
        return done_promises, self.promise_queue

    async def promise(self, actions: T.List[Action]) -> None:
        """
        Send actions to all simulators

        Parameters
        ----------
        actions : list[Action]
            List of actions to send to the simulators
        """
        for simulator in self._config.simulators:
            simulator_response = asyncio.create_task(
                self._promise_simulator(simulator, actions)
            )
            self.promise_queue.append(simulator_response)

    async def _promise_simulator(
        self, simulator: Simulator, actions: T.List[Action]
    ) -> T.Any:
        """
        Send actions to a single simulator

        Parameters
        ----------
        simulator : Simulator
            The simulator to send actions to
        actions : list[Action]
            List of actions to send to the simulator

        Returns
        -------
        Any
            The result of the simulator's response
        """
        logging.debug(f"Calling simulator {simulator.name} with actions {actions}")
        simulator.sim(actions)
        return None

    def stop(self, timeout: float = 3.0):
        """
        Stop all simulator threads.
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
                f"SimulatorOrchestrator: threads still running after timeout: {still_running}"
            )

    def __del__(self):
        """
        Clean up the SimulatorOrchestrator by stopping threads.
        """
        self.stop()
