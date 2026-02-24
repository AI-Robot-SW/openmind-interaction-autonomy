"""
Voice Activity Detection - SileroVAD wrapper.

Streaming VAD using Silero VAD model with VADIterator.
Extracted from om1_speech.py for clean dependency management.
"""

from __future__ import annotations

import logging
from typing import Optional


class SileroVAD:
    """
    Streaming Silero VAD wrapper using VADIterator.

    Usage
    -----
    vad = SileroVAD(sampling_rate=16000, threshold=0.5, chunk_size=1024)
    event = vad.process(audio_chunk)
    if event:
        ...
    if vad.speech_active:
        ...
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        min_speech_duration_ms: int = 250,
        chunk_size: Optional[int] = None,
        device: str = "cpu",
        return_seconds: bool = False,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        if self.sampling_rate == 16000:
            required_samples = 512
        elif self.sampling_rate == 8000:
            required_samples = 256
        else:
            raise ValueError(
                "SileroVAD supports sampling_rate 8000 or 16000 only."
            )
        if chunk_size not in (None, required_samples):
            logging.warning(
                "SileroVAD chunk_size=%s does not match required %s for %s Hz. "
                "Overriding to %s.",
                chunk_size,
                required_samples,
                self.sampling_rate,
                required_samples,
            )
        self.chunk_size = required_samples
        self.device = device
        self.return_seconds = return_seconds

        self._speech_active = False
        self._buffer = bytearray()
        self._prob_buffer = bytearray()
        self._prob_model = None
        self._last_prob: Optional[float] = None

        try:
            from silero_vad import VADIterator, load_silero_vad
        except Exception as exc:
            raise RuntimeError(
                "silero_vad package is required for SileroVAD. "
                "Install silero-vad and its dependencies."
            ) from exc

        try:
            self._model = load_silero_vad(device=device)
        except TypeError:
            self._model = load_silero_vad()
            if hasattr(self._model, "to"):
                self._model.to(device)

        try:
            self._vad_iterator = VADIterator(
                self._model,
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
                min_speech_duration_ms=self.min_speech_duration_ms,
            )
        except TypeError:
            self._vad_iterator = VADIterator(
                self._model,
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
            )

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def last_prob(self) -> Optional[float]:
        return self._last_prob

    def reset(self) -> None:
        if hasattr(self._vad_iterator, "reset_states"):
            self._vad_iterator.reset_states()
        elif hasattr(self._vad_iterator, "reset"):
            self._vad_iterator.reset()
        if self._prob_model is not None and hasattr(self._prob_model, "reset_states"):
            self._prob_model.reset_states()
        self._speech_active = False
        self._buffer.clear()
        self._prob_buffer.clear()
        self._last_prob = None

    def _iter_chunks(self, audio_chunk: bytes, buffer: Optional[bytearray] = None):
        target_buffer = self._buffer if buffer is None else buffer
        if self.chunk_size is None:
            yield audio_chunk
            return

        frame_bytes = self.chunk_size * 2
        target_buffer.extend(audio_chunk)
        while len(target_buffer) >= frame_bytes:
            chunk = bytes(target_buffer[:frame_bytes])
            del target_buffer[:frame_bytes]
            yield chunk

    def _bytes_to_tensor(self, audio_chunk: bytes):
        import numpy as np
        import torch

        if len(audio_chunk) % 2 == 1:
            audio_chunk = audio_chunk[:-1]
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio)
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        return tensor

    def process(self, audio_chunk: bytes):
        if not audio_chunk:
            return None
        import torch

        last_event = None
        for chunk in self._iter_chunks(audio_chunk):
            if not chunk:
                continue
            audio_tensor = self._bytes_to_tensor(chunk)
            with torch.no_grad():
                event = self._vad_iterator(audio_tensor, return_seconds=self.return_seconds)
            if event:
                if "start" in event:
                    self._speech_active = True
                if "end" in event:
                    self._speech_active = False
                last_event = event
        return last_event

    def _get_prob_model(self):
        if self._prob_model is not None:
            return self._prob_model
        try:
            from silero_vad import load_silero_vad
        except Exception as exc:
            raise RuntimeError(
                "silero_vad package is required for SileroVAD. "
                "Install silero-vad and its dependencies."
            ) from exc

        try:
            self._prob_model = load_silero_vad(device=self.device)
        except TypeError:
            self._prob_model = load_silero_vad()
            if hasattr(self._prob_model, "to"):
                self._prob_model.to(self.device)
        return self._prob_model

    def process_prob(self, audio_chunk: bytes) -> Optional[float]:
        """
        Return speech probability for the latest processed chunk.

        Note: Uses a separate model instance to avoid interfering with VADIterator state.
        """
        if not audio_chunk:
            return None

        import torch

        model = self._get_prob_model()
        last_prob = None
        for chunk in self._iter_chunks(audio_chunk, buffer=self._prob_buffer):
            if not chunk:
                continue
            audio_tensor = self._bytes_to_tensor(chunk)
            with torch.no_grad():
                last_prob = model(audio_tensor, self.sampling_rate).item()

        if last_prob is not None:
            self._last_prob = last_prob
        return last_prob
