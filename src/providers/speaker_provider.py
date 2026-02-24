"""
Speaker Provider - 오디오 출력 스트림 관리

이 모듈은 스피커로 오디오 데이터를 출력합니다.

Architecture:
    TTSProvider -> SpeakerProvider -> Speaker Device

Dependencies:
    - TTSProvider: 오디오 데이터 소스
    - PyAudio: 오디오 출력 라이브러리

Note:
    이 Provider는 Singleton 패턴을 사용하여 시스템 전체에서
    하나의 오디오 출력 스트림만 유지합니다.
"""

import array
import logging
import queue
import threading
from typing import Callable, List, Optional

import pyaudio

from .singleton import singleton


@singleton
class SpeakerProvider:
    """
    Speaker Provider - 오디오 출력 스트림 관리자.

    오디오 데이터를 큐에 저장하고 스피커로 재생합니다.
    PyAudio를 사용하여 실제 오디오 출력을 처리합니다.

    Attributes
    ----------
    running : bool
        Provider 실행 상태
    sample_rate : int
        오디오 샘플링 레이트 (Hz)
    channels : int
        오디오 채널 수
    device_id : Optional[int]
        스피커 디바이스 ID
    volume : float
        볼륨 레벨 (0.0 ~ 1.0)

    Methods
    -------
    start()
        오디오 출력 스트림 시작
    stop()
        오디오 출력 스트림 정지
    queue_audio(audio_data)
        오디오 데이터 큐에 추가
    play_audio(audio_data)
        오디오 데이터 즉시 재생
    stop_playback()
        현재 재생 중단
    set_volume(volume)
        볼륨 설정
    register_playback_callback(callback)
        재생 상태 콜백 등록
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        device_id: Optional[int] = None,
        device_name: Optional[str] = None,
        volume: float = 1.0,
        buffer_size: int = 1024,
    ):
        """
        Speaker Provider 초기화.

        Parameters
        ----------
        sample_rate : int
            오디오 샘플링 레이트 (Hz), 기본값 24000 (Naver Clova TTS 기본값)
        channels : int
            오디오 채널 수, 기본값 1 (mono)
        device_id : Optional[int]
            스피커 디바이스 ID, None이면 시스템 기본값
        device_name : Optional[str]
            스피커 디바이스 이름
        volume : float
            초기 볼륨 레벨 (0.0 ~ 1.0), 기본값 1.0
        buffer_size : int
            오디오 버퍼 크기 (frames per buffer), 기본값 1024
        """
        self.running: bool = False
        self.sample_rate = sample_rate
        self.channels = channels
        self.device_id = device_id
        self.device_name = device_name
        self._volume = max(0.0, min(1.0, volume))
        self.buffer_size = buffer_size

        # 오디오 큐 (PCM16 bytes)
        self._audio_queue: queue.Queue[bytes] = queue.Queue()

        # 현재 재생 중인 오디오 버퍼 (청크 단위로 분할하여 재생)
        self._current_audio_buffer: bytes = b""
        self._buffer_lock = threading.Lock()

        # 콜백 리스트
        self._playback_callbacks: List[Callable[[str], None]] = []

        # 상태 변수
        self._is_playing: bool = False
        self._should_stop: bool = False
        self._lock = threading.Lock()
        self._playback_thread: Optional[threading.Thread] = None

        # PyAudio 인스턴스
        self._audio_interface: Optional[pyaudio.PyAudio] = None
        self._audio_stream: Optional[pyaudio.Stream] = None

        logging.info(
            f"SpeakerProvider initialized: rate={sample_rate}, "
            f"channels={channels}, device={device_id or device_name or 'default'}"
        )

    @property
    def volume(self) -> float:
        """현재 볼륨 레벨 반환."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """볼륨 레벨 설정."""
        self._volume = max(0.0, min(1.0, value))
        logging.debug(f"Volume set to {self._volume}")

    @property
    def data(self) -> Optional[dict]:
        """
        Get the current provider data.

        Returns
        -------
        Optional[dict]
            Current speaker provider state information.
        """
        with self._lock:
            return {
                "is_playing": self._is_playing,
                "queue_size": self._audio_queue.qsize(),
                "volume": self._volume,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "running": self.running,
            }

    def register_playback_callback(self, callback: Callable[[str], None]) -> None:
        """
        재생 상태 콜백 등록.

        Parameters
        ----------
        callback : Callable[[str], None]
            재생 상태를 받을 콜백 함수
            상태: "started", "playing", "paused", "stopped", "completed"
        """
        if callback not in self._playback_callbacks:
            self._playback_callbacks.append(callback)
            logging.debug(f"Playback callback registered: {callback.__name__}")

    def unregister_playback_callback(self, callback: Callable[[str], None]) -> None:
        """
        재생 상태 콜백 해제.

        Parameters
        ----------
        callback : Callable[[str], None]
            해제할 콜백 함수
        """
        if callback in self._playback_callbacks:
            self._playback_callbacks.remove(callback)
            logging.debug(f"Playback callback unregistered: {callback.__name__}")

    def _notify_playback_state(self, state: str) -> None:
        """
        재생 상태 변경 알림.

        Parameters
        ----------
        state : str
            새 재생 상태
        """
        for callback in self._playback_callbacks:
            try:
                callback(state)
            except Exception as e:
                logging.error(f"Playback callback error: {e}")

    def get_available_devices(self) -> List[dict]:
        """
        사용 가능한 출력 디바이스 목록 반환.

        Returns
        -------
        List[dict]
            출력 가능한 디바이스 정보 리스트
        """
        devices = []
        audio_interface = None

        try:
            # 임시 PyAudio 인스턴스 생성 (running 상태가 아닐 수 있음)
            if self._audio_interface:
                audio_interface = self._audio_interface
            else:
                audio_interface = pyaudio.PyAudio()

            for i in range(audio_interface.get_device_count()):
                info = audio_interface.get_device_info_by_index(i)
                # 출력 채널이 있는 디바이스만 포함
                if info.get("maxOutputChannels", 0) > 0:
                    devices.append(info)

        except Exception as e:
            logging.error(f"Failed to get available devices: {e}")
        finally:
            # 임시로 생성한 경우에만 종료
            if audio_interface and audio_interface != self._audio_interface:
                audio_interface.terminate()

        return devices

    def _find_device_by_name(self, name: str) -> Optional[int]:
        """
        디바이스 이름으로 출력 디바이스 인덱스 찾기.

        Parameters
        ----------
        name : str
            찾을 디바이스 이름 (부분 일치)

        Returns
        -------
        Optional[int]
            디바이스 인덱스, 찾지 못하면 None
        """
        if not self._audio_interface:
            return None

        for i in range(self._audio_interface.get_device_count()):
            info = self._audio_interface.get_device_info_by_index(i)
            if name.lower() in info["name"].lower() and info["maxOutputChannels"] > 0:
                logging.info(f"Found output device: {info['name']} (index {i})")
                return i

        logging.warning(f"Output device '{name}' not found, using default")
        return None

    def _apply_volume(self, audio_data: bytes) -> bytes:
        """
        PCM16 오디오 데이터에 볼륨 적용.

        Parameters
        ----------
        audio_data : bytes
            PCM16 형식의 오디오 데이터

        Returns
        -------
        bytes
            볼륨이 적용된 오디오 데이터
        """
        if self._volume >= 1.0:
            return audio_data

        if self._volume <= 0.0:
            return bytes(len(audio_data))

        try:
            # PCM16 데이터를 signed short array로 변환
            samples = array.array("h", audio_data)
            # 볼륨 적용 (클리핑 방지)
            scaled = array.array(
                "h", [max(-32768, min(32767, int(s * self._volume))) for s in samples]
            )
            return scaled.tobytes()
        except Exception as e:
            logging.error(f"Failed to apply volume: {e}")
            return audio_data

    def queue_audio(self, audio_data: bytes) -> None:
        """
        오디오 데이터를 재생 큐에 추가.

        Parameters
        ----------
        audio_data : bytes
            재생할 오디오 데이터 (PCM16 형식)
        """
        if not self.running:
            logging.warning("SpeakerProvider is not running. Call start() first.")
            return

        self._audio_queue.put(audio_data)
        logging.debug(f"Audio queued: {len(audio_data)} bytes")

    def play_audio(self, audio_data: bytes, blocking: bool = False) -> None:
        """
        오디오 데이터 즉시 재생.

        Parameters
        ----------
        audio_data : bytes
            재생할 오디오 데이터 (PCM16 형식)
        blocking : bool
            True이면 재생 완료까지 블로킹, 기본값 False
        """
        if not self.running:
            logging.warning("SpeakerProvider is not running. Call start() first.")
            return

        # 큐 비우기
        self.clear_queue()

        # 오디오 큐에 추가
        self._audio_queue.put(audio_data)

        if blocking:
            # 재생 완료 대기
            self._audio_queue.join()

    def clear_queue(self) -> None:
        """
        재생 큐 비우기.
        """
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except queue.Empty:
                break

        # 현재 버퍼도 비우기
        with self._buffer_lock:
            self._current_audio_buffer = b""

        logging.debug("Audio queue cleared")

    def stop_playback(self) -> None:
        """
        현재 재생 중단.
        """
        with self._lock:
            self._should_stop = True
        self.clear_queue()
        self._notify_playback_state("stopped")
        logging.info("Playback stopped")

    def _playback_loop(self) -> None:
        """
        재생 루프 (별도 스레드에서 실행).
        PyAudio 스트림을 통해 오디오 데이터를 출력합니다.
        """
        while self.running:
            try:
                # 큐에서 오디오 데이터 가져오기 (1초 타임아웃)
                audio_data = self._audio_queue.get(timeout=1.0)

                # 새 오디오에 대해 stop 플래그 리셋
                # (_should_stop은 현재 재생 중 청크 중단용이며,
                #  clear_queue()가 이미 이전 오디오를 제거했으므로
                #  여기 도달한 오디오는 stop 이후 새로 큐잉된 것)
                with self._lock:
                    self._should_stop = False

                # 재생 시작 알림
                with self._lock:
                    self._is_playing = True
                self._notify_playback_state("playing")

                # 볼륨 적용
                if self._volume < 1.0:
                    audio_data = self._apply_volume(audio_data)

                # PyAudio 스트림에 쓰기
                if self._audio_stream and self._audio_stream.is_active():
                    try:
                        # 청크 단위로 분할하여 쓰기
                        bytes_per_frame = self.channels * 2  # PCM16 = 2 bytes per sample
                        chunk_size = self.buffer_size * bytes_per_frame

                        offset = 0
                        while offset < len(audio_data) and self.running:
                            if self._should_stop:
                                break

                            chunk = audio_data[offset : offset + chunk_size]
                            # 청크가 부족하면 패딩
                            if len(chunk) < chunk_size:
                                chunk = chunk + bytes(chunk_size - len(chunk))

                            self._audio_stream.write(chunk)
                            offset += chunk_size

                    except Exception as e:
                        logging.error(f"Error writing to audio stream: {e}")

                # 재생 완료
                with self._lock:
                    self._is_playing = False
                self._notify_playback_state("completed")
                self._audio_queue.task_done()

            except queue.Empty:
                # 타임아웃 - 계속 대기
                continue
            except Exception as e:
                logging.error(f"Playback error: {e}")
                with self._lock:
                    self._is_playing = False

        self._notify_playback_state("stopped")

    def is_playing(self) -> bool:
        """
        현재 재생 중인지 확인.

        Returns
        -------
        bool
            재생 중이면 True
        """
        with self._lock:
            return self._is_playing

    def get_queue_size(self) -> int:
        """
        재생 큐 크기 반환.

        Returns
        -------
        int
            큐에 대기 중인 오디오 청크 수
        """
        return self._audio_queue.qsize()

    def start(self) -> None:
        """
        오디오 출력 스트림 시작.
        PyAudio를 초기화하고 출력 스트림을 엽니다.
        """
        if self.running:
            logging.warning("SpeakerProvider is already running")
            return

        self.running = True
        self._should_stop = False

        # PyAudio 초기화
        try:
            self._audio_interface = pyaudio.PyAudio()

            # 디바이스 찾기
            device_index = self.device_id
            if device_index is None and self.device_name:
                device_index = self._find_device_by_name(self.device_name)

            # 출력 스트림 열기
            self._audio_stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=device_index,
                frames_per_buffer=self.buffer_size,
            )

            logging.info(
                f"PyAudio stream opened: rate={self.sample_rate}, "
                f"channels={self.channels}, buffer={self.buffer_size}"
            )

        except Exception as e:
            logging.error(f"Failed to initialize PyAudio: {e}")
            self.running = False
            return

        # 재생 스레드 시작
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

        self._notify_playback_state("started")
        logging.info("SpeakerProvider started")

    def stop(self) -> None:
        """
        오디오 출력 스트림 정지.
        PyAudio 스트림을 닫고 리소스를 해제합니다.
        """
        if not self.running:
            logging.warning("SpeakerProvider is not running")
            return

        self.running = False
        self._should_stop = True
        self.clear_queue()

        # 재생 스레드 종료 대기
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)

        # PyAudio 스트림 닫기
        if self._audio_stream:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception as e:
                logging.error(f"Error closing audio stream: {e}")
            self._audio_stream = None

        # PyAudio 종료
        if self._audio_interface:
            try:
                self._audio_interface.terminate()
            except Exception as e:
                logging.error(f"Error terminating PyAudio: {e}")
            self._audio_interface = None

        self._notify_playback_state("stopped")
        logging.info("SpeakerProvider stopped")

    def configure(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        device_id: Optional[int] = None,
        volume: Optional[float] = None,
    ) -> None:
        """
        Provider 설정 변경.

        Parameters
        ----------
        sample_rate : Optional[int]
            새 샘플링 레이트
        channels : Optional[int]
            새 채널 수
        device_id : Optional[int]
            새 디바이스 ID
        volume : Optional[float]
            새 볼륨 레벨
        """
        restart_needed = False

        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            restart_needed = True

        if channels is not None and channels != self.channels:
            self.channels = channels
            restart_needed = True

        if device_id is not None and device_id != self.device_id:
            self.device_id = device_id
            restart_needed = True

        if volume is not None:
            self.volume = volume

        if restart_needed and self.running:
            self.stop()
            self.start()
            logging.info("SpeakerProvider reconfigured and restarted")
