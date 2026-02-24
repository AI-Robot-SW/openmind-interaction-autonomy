from unittest.mock import MagicMock, patch
import pytest

from backgrounds.plugins.realsense_camera_bg import (
    RealSenseCameraBg,
    RealSenseCameraBgConfig,
)


@pytest.fixture
def config():
    return RealSenseCameraBgConfig(camera_index=0)


@pytest.fixture
def config_with_custom_index():
    return RealSenseCameraBgConfig(camera_index=1)


def test_config_initialization():
    """Test RealSenseCameraBgConfig initialization."""
    config = RealSenseCameraBgConfig()
    assert config.camera_index == 0

    config_custom = RealSenseCameraBgConfig(camera_index=1)
    assert config_custom.camera_index == 1


@patch("backgrounds.plugins.realsense_camera_bg.RealSenseCameraProvider")
def test_background_initialization_creates_provider(mock_provider_class, config):
    """__init__ should create provider but NOT start it."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = RealSenseCameraBg(config=config)

    mock_provider_class.assert_called_once_with(
        camera_index=0,
        width=640,
        height=480,
        fps=30,
        align_depth_to_color=True,
    )

    assert background.realsense_camera_provider is mock_provider_instance
    # start()는 run()에서 호출되므로 __init__에서는 호출되지 않아야 함
    mock_provider_instance.start.assert_not_called()


@patch("backgrounds.plugins.realsense_camera_bg.RealSenseCameraProvider")
def test_background_initialization_custom_index_creates_provider(
    mock_provider_class, config_with_custom_index
):
    """__init__ should pass custom camera_index to provider."""
    mock_provider_instance = MagicMock()
    mock_provider_class.return_value = mock_provider_instance

    background = RealSenseCameraBg(config=config_with_custom_index)

    mock_provider_class.assert_called_once_with(
        camera_index=1,
        width=640,
        height=480,
        fps=30,
        align_depth_to_color=True,
    )

    assert background.realsense_camera_provider is mock_provider_instance
    mock_provider_instance.start.assert_not_called()


def test_background_name(config):
    """Test that background has correct name."""
    background = RealSenseCameraBg(config=config)
    assert background.name == "RealSenseCameraBg"


def test_background_config_access(config):
    """Test that background has access to config."""
    background = RealSenseCameraBg(config=config)
    assert background.config == config
    assert background.config.camera_index == 0
