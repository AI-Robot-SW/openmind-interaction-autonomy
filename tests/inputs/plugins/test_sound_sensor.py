"""
Tests for SoundSensor.
"""

import asyncio
import time
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.sound_sensor import SoundSensor, SoundSensorConfig


@pytest.fixture
def config():
    """Configuration for SoundSensor."""
    return SoundSensorConfig(language="korean")


@pytest.fixture
def config_english():
    """English configuration for SoundSensor."""
    return SoundSensorConfig(language="english")


@pytest.fixture
def mock_stt():
    """Mock STTProvider for SoundSensor tests."""
    with patch("inputs.plugins.sound_sensor.STTProvider") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


class TestSoundSensorConfig:
    """Test SoundSensorConfig initialization."""

    def test_config_default_values(self):
        """Test SoundSensorConfig default values."""
        config = SoundSensorConfig()

        assert config.language == "korean"

    def test_config_custom_values(self):
        """Test SoundSensorConfig with custom values."""
        config = SoundSensorConfig(language="english")

        assert config.language == "english"


class TestSoundSensorInitialization:
    """Test SoundSensor initialization."""

    def test_sensor_initialization(self, config, mock_stt):
        """Test SoundSensor initialization."""
        sensor = SoundSensor(config=config)

        assert sensor.descriptor_for_LLM == "Voice"
        assert isinstance(sensor.message_buffer, Queue)
        assert sensor.messages == []

    def test_sensor_registers_stt_callback(self, config, mock_stt):
        """Test SoundSensor registers callback with STTProvider."""
        sensor = SoundSensor(config=config)

        mock_stt.register_result_callback.assert_called_once_with(
            sensor._handle_stt_result
        )

    def test_sensor_config_access(self, config, mock_stt):
        """Test sensor has access to config."""
        sensor = SoundSensor(config=config)

        assert sensor.config == config
        assert sensor.config.language == "korean"


class TestSoundSensorSTTCallback:
    """Test SoundSensor STT callback handling."""

    def test_handle_stt_result_valid(self, config, mock_stt):
        """Test handling valid STT result."""
        sensor = SoundSensor(config=config)

        sensor._handle_stt_result("안녕하세요")

        assert sensor.message_buffer.qsize() == 1
        assert sensor.message_buffer.get() == "안녕하세요"

    def test_handle_stt_result_empty(self, config, mock_stt):
        """Test handling empty STT result."""
        sensor = SoundSensor(config=config)

        sensor._handle_stt_result("")

        assert sensor.message_buffer.qsize() == 0

    def test_handle_stt_result_whitespace(self, config, mock_stt):
        """Test handling whitespace-only STT result."""
        sensor = SoundSensor(config=config)

        sensor._handle_stt_result("   ")

        assert sensor.message_buffer.qsize() == 0


class TestSoundSensorPoll:
    """Test SoundSensor polling."""

    def test_poll_with_message(self, config, mock_stt):
        """Test polling when message is available."""
        sensor = SoundSensor(config=config)
        sensor.message_buffer.put("테스트 메시지")

        result = asyncio.run(sensor._poll())

        assert result == "테스트 메시지"

    def test_poll_empty_buffer(self, config, mock_stt):
        """Test polling when buffer is empty."""
        sensor = SoundSensor(config=config)

        result = asyncio.run(sensor._poll())

        assert result is None


class TestSoundSensorRawToText:
    """Test SoundSensor raw to text conversion."""

    def test_raw_to_text_valid(self, config, mock_stt):
        """Test converting valid raw input to text."""
        sensor = SoundSensor(config=config)

        message = asyncio.run(sensor._raw_to_text("테스트"))

        assert message is not None
        assert message.message == "테스트"
        assert message.timestamp > 0

    def test_raw_to_text_none(self, config, mock_stt):
        """Test converting None input."""
        sensor = SoundSensor(config=config)

        message = asyncio.run(sensor._raw_to_text(None))

        assert message is None

    def test_raw_to_text_accumulates(self, config, mock_stt):
        """Test that raw_to_text accumulates messages."""
        sensor = SoundSensor(config=config)

        asyncio.run(sensor.raw_to_text("첫번째"))
        asyncio.run(sensor.raw_to_text("두번째"))

        assert len(sensor.messages) == 1
        assert "첫번째" in sensor.messages[0]
        assert "두번째" in sensor.messages[0]


class TestSoundSensorFormattedBuffer:
    """Test SoundSensor formatted buffer output."""

    def test_formatted_latest_buffer_empty(self, config, mock_stt):
        """Test formatted buffer when empty."""
        sensor = SoundSensor(config=config)

        result = sensor.formatted_latest_buffer()

        assert result is None

    def test_formatted_latest_buffer_with_message(self, config, mock_stt):
        """Test formatted buffer with message."""
        sensor = SoundSensor(config=config)
        sensor.messages = ["테스트 메시지"]

        result = sensor.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Voice" in result
        assert "테스트 메시지" in result
        assert sensor.messages == []  # Buffer should be cleared

    def test_formatted_latest_buffer_format(self, config, mock_stt):
        """Test formatted buffer has correct format."""
        sensor = SoundSensor(config=config)
        sensor.messages = ["안녕하세요"]

        result = sensor.formatted_latest_buffer()

        assert "// START" in result
        assert "// END" in result
