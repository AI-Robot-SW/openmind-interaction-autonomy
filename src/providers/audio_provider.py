"""
Audio Provider - 오디오 입력 스트림 관리

이 모듈은 마이크로부터 오디오 데이터를 캡처하고,
VAD(Voice Activity Detection)를 통해 음성 구간을 감지합니다.

Architecture:
    AudioBg -> AudioProvider -> Microphone Device
                    ↓
              STTProvider (오디오 데이터 전달)

Dependencies:
    - sounddevice: 오디오 스트림 처리
    - SileroVAD (vad.py): 음성 활동 감지

Note:
    이 Provider는 Singleton 패턴을 사용하여 시스템 전체에서
    하나의 오디오 입력 스트림만 유지합니다.
"""

import logging
import math
import queue
import struct
import threading
from collections import deque
from typing import Callable, Dict, List, Optional

import numpy as np
import sounddevice as sd

from .singleton import singleton
from .vad import SileroVAD


@singleton
class AudioProvider:
    """
    Audio Provider - 오디오 입력 스트림 관리자.

    마이크로부터 오디오 데이터를 캡처하고, VAD를 통해 음성 구간을 감지하며,
    등록된 콜백으로 오디오 데이터를 전달합니다.

    Attributes
    ----------
    running : bool
        Provider 실행 상태
    sample_rate : int
        오디오 샘플링 레이트 (Hz)
    chunk_size : int
        오디오 청크 크기 (samples)
    channels : int
        오디오 채널 수
    device_id : Optional[int]
        마이크 디바이스 ID
    device_name : Optional[str]
        마이크 디바이스 이름
    remote_input : bool
        원격 오디오 입력 모드
    vad_enabled : bool
        VAD 활성화 여부
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_id: Optional[int] = None,
        device_name: Optional[str] = None,
        remote_input: bool = False,
        vad_enabled: bool = True,
        vad_threshold: float = 0.5,
        vad_min_silence_duration_ms: int = 100,
        vad_speech_pad_ms: int = 30,
        vad_min_speech_duration_ms: int = 250,
        vad_device: str = "cpu",
        buffer_duration_ms: int = 200,
    ):
        self.running: bool = False
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_id = device_id
        self.device_name = device_name
        self.remote_input = remote_input
        self.vad_enabled = vad_enabled
        self.vad_threshold = vad_threshold
        self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.vad_min_speech_duration_ms = vad_min_speech_duration_ms
        self.vad_device = vad_device
        self.buffer_duration_ms = buffer_duration_ms

        # 콜백 리스트
        self._audio_callbacks: List[Callable[..., None]] = []
        self._vad_callbacks: List[Callable[[bool], None]] = []

        # 상태 변수
        self._current_audio_level: float = 0.0
        self._is_voice_active: bool = False
        self._lock = threading.Lock()
        self._audio_level_window_buffer = bytearray()
        self._recent_peak_levels = deque([0.0, 0.0, 0.0], maxlen=3)
        self._audio_level_window_bytes = 0
        self._audio_level_tail_samples = 0
        self._init_audio_level_tracker()

        # 오디오 버퍼 (queue 기반)
        self._audio_buffer: "queue.Queue[bytes]"
        self._buffer_drops = 0
        self._chunk_ms = (self.chunk_size / self.sample_rate) * 1000.0
        self.initialize_audio_buffer(buffer_duration_ms=self.buffer_duration_ms)

        # sounddevice (lazy init in start_stream)
        self._stream: Optional[sd.InputStream] = None

        # VAD 엔진
        self._vad_engine: Optional[SileroVAD] = None
        self._init_vad_engine()

        logging.info(
            "AudioProvider initialized: rate=%s, chunk=%s, device=%s, remote_input=%s",
            sample_rate,
            chunk_size,
            device_id or device_name or "default",
            remote_input,
        )

    def initialize_audio_buffer(self, buffer_duration_ms: Optional[int] = None) -> None:
        """
        오디오 버퍼를 초기화합니다.

        Parameters
        ----------
        buffer_duration_ms : Optional[int]
            버퍼 길이(ms). None이면 기존 설정 사용
        """
        if buffer_duration_ms is not None:
            self.buffer_duration_ms = buffer_duration_ms
        self._chunk_ms = (self.chunk_size / self.sample_rate) * 1000.0
        max_chunks = max(1, int(math.ceil(self.buffer_duration_ms / self._chunk_ms)))
        self._audio_buffer = queue.Queue(maxsize=max_chunks)
        self._buffer_drops = 0

    def _init_vad_engine(self) -> None:
        if not self.vad_enabled:
            self._vad_engine = None
            return
        try:
            self._vad_engine = SileroVAD(
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_silence_duration_ms=self.vad_min_silence_duration_ms,
                speech_pad_ms=self.vad_speech_pad_ms,
                min_speech_duration_ms=self.vad_min_speech_duration_ms,
                chunk_size=self.chunk_size,
                device=self.vad_device,
            )
        except Exception as e:
            logging.error("SileroVAD init failed: %s", e)
            self._vad_engine = None
            self.vad_enabled = False

    def _init_audio_level_tracker(self) -> None:
        """
        오디오 레벨 계산기 초기화.

        64ms 단위로 계산하고, 각 64ms 윈도우의 마지막 10ms peak를
        3점 이동평균으로 평활화합니다.
        """
        window_frames = max(1, int(self.sample_rate * 64 / 1000))
        tail_frames = max(1, int(self.sample_rate * 10 / 1000))
        self._audio_level_window_bytes = window_frames * self.channels * 2
        self._audio_level_tail_samples = tail_frames * self.channels
        self._audio_level_window_buffer.clear()
        self._recent_peak_levels = deque([0.0, 0.0, 0.0], maxlen=3)

    def _compute_smoothed_peak_level(self, window_bytes: bytes) -> Optional[float]:
        """
        64ms 윈도우에서 마지막 10ms peak를 구하고 3점 이동평균을 계산합니다.

        Returns
        -------
        Optional[float]
            계산된 레벨(0.0~1.0) 또는 계산 불가 시 None
        """
        sample_count = len(window_bytes) // 2
        if sample_count == 0:
            return None

        samples = struct.unpack("<" + "h" * sample_count, window_bytes)

        # 마지막 10ms 구간 peak amplitude
        tail_samples = max(1, min(self._audio_level_tail_samples, sample_count))
        tail = samples[-tail_samples:]
        peak = max(abs(s) for s in tail)
        peak_norm = min(1.0, peak / 32768.0)

        # volume = (p_{k-2} + p_{k-1} + p_k) / 3
        self._recent_peak_levels.append(peak_norm)
        return sum(self._recent_peak_levels) / 3.0

    # ---- Device resolution ----

    def _resolve_device_index(self) -> Optional[int]:
        """device_id 또는 device_name으로 sounddevice 입력 디바이스 검색."""
        if self.device_id is not None:
            return self.device_id
        if not self.device_name:
            return None
        name_l = self.device_name.lower()
        devices = sd.query_devices()
        for idx, info in enumerate(devices):
            dev_name = str(info.get("name", "")).lower()
            max_in = int(info.get("max_input_channels", 0))
            if name_l in dev_name and max_in > 0:
                logging.info("AudioProvider resolved device_name '%s' -> index %d", self.device_name, idx)
                return idx
        logging.warning("AudioProvider device_name not found: %s", self.device_name)
        return None

    # ---- Callbacks ----

    def register_audio_callback(self, callback: Callable[..., None]) -> None:
        """오디오 데이터 콜백 등록."""
        if callback not in self._audio_callbacks:
            self._audio_callbacks.append(callback)
            logging.debug("Audio callback registered: %s", getattr(callback, '__name__', repr(callback)))

    def unregister_audio_callback(self, callback: Callable[..., None]) -> None:
        """오디오 데이터 콜백 해제."""
        if callback in self._audio_callbacks:
            self._audio_callbacks.remove(callback)
            logging.debug("Audio callback unregistered: %s", getattr(callback, '__name__', repr(callback)))

    def register_vad_callback(self, callback: Callable[[bool], None]) -> None:
        """VAD 이벤트 콜백 등록."""
        if callback not in self._vad_callbacks:
            self._vad_callbacks.append(callback)
            logging.debug("VAD callback registered: %s", getattr(callback, '__name__', repr(callback)))

    def unregister_vad_callback(self, callback: Callable[[bool], None]) -> None:
        """VAD 이벤트 콜백 해제."""
        if callback in self._vad_callbacks:
            self._vad_callbacks.remove(callback)
            logging.debug("VAD callback unregistered: %s", getattr(callback, '__name__', repr(callback)))

    # ---- Audio data handling ----

    def _fill_buffer(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice stream callback — 오디오 저장 및 처리."""
        if status:
            logging.warning("AudioProvider stream status: %s", status)

        # numpy int16 배열 → mono 변환 후 bytes
        # sounddevice 콜백의 indata는 non-contiguous view이므로 .copy() 필요
        mono = indata[:, 0].copy()
        in_data = mono.tobytes()

        # 오디오 버퍼에 저장 (가득 차면 오래된 청크를 버리고 최신 유지)
        try:
            self._audio_buffer.put_nowait(in_data)
        except queue.Full:
            self._buffer_drops += 1
            try:
                _ = self._audio_buffer.get_nowait()
                self._audio_buffer.put_nowait(in_data)
            except queue.Empty:
                pass

        # 오디오 레벨 계산
        self._update_audio_level(in_data)

        # VAD 처리
        if self.vad_enabled:
            self._process_vad(in_data)

        # 등록된 콜백 호출
        for callback in list(self._audio_callbacks):
            try:
                callback(in_data, frames, time_info, status)
            except Exception as e:
                logging.error("Audio callback error: %s", e)

    def _update_audio_level(self, audio_chunk: bytes) -> None:
        if not audio_chunk:
            return
        try:
            self._audio_level_window_buffer.extend(audio_chunk)

            # 충분한 buffer가 모일때마다 계산
            while len(self._audio_level_window_buffer) >= self._audio_level_window_bytes:
                window_bytes = self._audio_level_window_buffer[
                    : self._audio_level_window_bytes
                ]
                del self._audio_level_window_buffer[: self._audio_level_window_bytes]

                level = self._compute_smoothed_peak_level(window_bytes)
                if level is None:
                    continue

                with self._lock:
                    self._current_audio_level = level
        except Exception as e:
            logging.error("Audio level update failed: %s", e)

    def _process_vad(self, audio_chunk: bytes) -> None:
        if not self.vad_enabled or self._vad_engine is None:
            return
        try:
            self._vad_engine.process(audio_chunk)
            is_voice = self._vad_engine.speech_active
        except Exception as e:
            logging.error("VAD processing error: %s", e)
            return

        with self._lock:
            was_active = self._is_voice_active
            self._is_voice_active = is_voice

        if was_active != is_voice:
            for callback in list(self._vad_callbacks):
                try:
                    callback(is_voice)
                except Exception as e:
                    logging.error("VAD callback error: %s", e)

    # ---- Query methods ----

    def get_audio_level(self) -> float:
        """현재 오디오 레벨 반환 (0.0 ~ 1.0)."""
        with self._lock:
            return self._current_audio_level

    def is_voice_active(self) -> bool:
        """현재 음성 활성 상태 반환."""
        with self._lock:
            return self._is_voice_active

    def get_buffer_stats(self) -> Dict[str, float]:
        """오디오 버퍼 상태 반환."""
        queued = self._audio_buffer.qsize()
        maxsize = self._audio_buffer.maxsize
        approx_latency_ms = float(queued) * float(self._chunk_ms)
        return {
            "queued_chunks": float(queued),
            "max_chunks": float(maxsize) if maxsize > 0 else 0.0,
            "drops": float(self._buffer_drops),
            "approx_latency_ms": approx_latency_ms,
        }

    def get_audio_chunk(self, timeout: float = 0.0) -> Optional[bytes]:
        """오디오 버퍼에서 청크를 가져옵니다."""
        try:
            return self._audio_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_audio_buffer(self) -> None:
        """오디오 버퍼 비우기."""
        while True:
            try:
                self._audio_buffer.get_nowait()
            except queue.Empty:
                break

    # ---- Lifecycle ----

    def start(self) -> None:
        """오디오 스트림 시작."""
        self.start_stream()

    def start_stream(self) -> None:
        """오디오 스트림 시작."""
        if self.running:
            logging.warning("AudioProvider is already running")
            return

        if self.remote_input:
            self.running = True
            logging.info("AudioProvider started in remote_input mode")
            return

        device_index = self._resolve_device_index()
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=device_index,
                channels=self.channels,
                dtype="int16",
                callback=self._fill_buffer,
            )
            self._stream.start()
        except Exception:
            self._stream = None
            raise

        self.running = True
        logging.info(
            "AudioProvider started: rate=%s chunk=%s device=%s",
            self.sample_rate,
            self.chunk_size,
            device_index if device_index is not None else "default",
        )

    def stop(self) -> None:
        """오디오 스트림 정지."""
        if not self.running:
            logging.warning("AudioProvider is not running")
            return

        self.running = False

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

        logging.info("AudioProvider stopped")

    def fill_buffer_remote(self, audio_bytes: bytes) -> None:
        """원격 오디오 데이터 입력 (remote_input 모드 전용)."""
        if not self.remote_input:
            return
        self._fill_buffer(audio_bytes, len(audio_bytes), {}, 0)

    def configure(
        self,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
        device_id: Optional[int] = None,
        device_name: Optional[str] = None,
        vad_enabled: Optional[bool] = None,
        vad_threshold: Optional[float] = None,
        vad_min_silence_duration_ms: Optional[int] = None,
        vad_speech_pad_ms: Optional[int] = None,
        vad_min_speech_duration_ms: Optional[int] = None,
        vad_device: Optional[str] = None,
    ) -> None:
        """Provider 설정 변경. 변경된 설정이 있으면 스트림을 재시작합니다."""
        restart_needed = False
        vad_reinit_needed = False

        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            self._init_audio_level_tracker()
            restart_needed = True
            vad_reinit_needed = True

        if chunk_size is not None and chunk_size != self.chunk_size:
            self.chunk_size = chunk_size
            restart_needed = True
            vad_reinit_needed = True

        if device_id is not None and device_id != self.device_id:
            self.device_id = device_id
            restart_needed = True

        if device_name is not None and device_name != self.device_name:
            self.device_name = device_name
            restart_needed = True

        if vad_enabled is not None and vad_enabled != self.vad_enabled:
            self.vad_enabled = vad_enabled
            vad_reinit_needed = True

        if vad_threshold is not None and vad_threshold != self.vad_threshold:
            self.vad_threshold = vad_threshold
            vad_reinit_needed = True

        if (
            vad_min_silence_duration_ms is not None
            and vad_min_silence_duration_ms != self.vad_min_silence_duration_ms
        ):
            self.vad_min_silence_duration_ms = vad_min_silence_duration_ms
            vad_reinit_needed = True

        if vad_speech_pad_ms is not None and vad_speech_pad_ms != self.vad_speech_pad_ms:
            self.vad_speech_pad_ms = vad_speech_pad_ms
            vad_reinit_needed = True

        if (
            vad_min_speech_duration_ms is not None
            and vad_min_speech_duration_ms != self.vad_min_speech_duration_ms
        ):
            self.vad_min_speech_duration_ms = vad_min_speech_duration_ms
            vad_reinit_needed = True

        if vad_device is not None and vad_device != self.vad_device:
            self.vad_device = vad_device
            vad_reinit_needed = True

        if vad_reinit_needed:
            if self.vad_enabled:
                self._init_vad_engine()
            else:
                self._vad_engine = None
                with self._lock:
                    self._is_voice_active = False

        if restart_needed and self.running:
            self.stop()
            self.start()
            logging.info("AudioProvider reconfigured and restarted")
