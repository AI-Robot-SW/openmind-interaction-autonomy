# gpu_worker.py

"""
GPUWorker — CUDA context를 소유하는 전용 스레드.

모든 CUDA 호출(TRT 추론, 커널 실행 등)은 이 스레드에서만 실행.
provider thread는 submit()으로 작업을 제출하고 Future.result()로 결과를 대기.

각 provider가 독립 인스턴스를 생성해 GPU 경합을 줄인다.

Usage:
    worker = GPUWorker()

    # GPU 작업 제출 (non-blocking)
    future = worker.submit(lambda: engine.infer(data))

    # 결과 대기 (blocking)
    outputs = future.result()
"""

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Callable, TypeVar

import pycuda.driver as cuda


T = TypeVar("T")


class GPUWorker:
    """
    CUDA context를 소유하는 전용 스레드.
    모든 CUDA 호출은 이 스레드를 통해 직렬 실행된다.
    각 provider가 독립 인스턴스를 생성해 pipeline 간 경합을 방지한다.
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._ctx = None
        self._ready = threading.Event()
        self._error: Exception | None = None

        self._thread = threading.Thread(target=self._run, daemon=True, name="GPUWorker")
        self._thread.start()

        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(f"GPUWorker CUDA init failed: {self._error}")

        logging.info("GPUWorker ready")

    def _run(self) -> None:
        try:
            cuda.init()
            self._ctx = cuda.Device(0).make_context()
        except Exception as e:
            self._error = e
            self._ready.set()
            return

        self._ready.set()

        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                fn, future = item
                try:
                    future.set_result(fn())
                except Exception as e:
                    future.set_exception(e)
        finally:
            try:
                self._ctx.pop()
                self._ctx.detach()
            except Exception as e:
                logging.warning(f"GPUWorker CUDA context cleanup failed: {e}")
            self._ctx = None

    def submit(self, fn: Callable[[], T]) -> "Future[T]":
        """
        GPU 작업을 큐에 제출한다.

        Parameters
        ----------
        fn : Callable[[], T]
            인자 없는 callable. GPU thread에서 실행됨.

        Returns
        -------
        Future[T]
            .result()로 결과 대기 (blocking).
        """
        f: Future[T] = Future()
        self._queue.put((fn, f))
        return f

    def stop(self) -> None:
        """GPU thread를 종료하고 CUDA context를 정리한다."""
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logging.info("GPUWorker stopped")
