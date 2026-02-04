"""
Test device control module
"""
import pytest
from src.device.adb_client import ADBClient
from src.device.actions import DeviceActions
from src.device.app_launcher import AppLauncher


@pytest.fixture
def adb_client():
    """Create ADB client fixture"""
    client = ADBClient()
    if not client.connect():
        pytest.skip("No device connected")
    return client


@pytest.fixture
def device_actions(adb_client):
    """Create device actions fixture"""
    device = adb_client.get_device()
    return DeviceActions(device)


@pytest.fixture
def app_launcher(adb_client):
    """Create app launcher fixture"""
    device = adb_client.get_device()
    return AppLauncher(device)


def test_adb_connection(adb_client):
    """Test ADB connection"""
    assert adb_client.is_connected()
    device_info = adb_client.get_device_info()
    assert "serial" in device_info


def test_screenshot(adb_client):
    """Test screenshot capture"""
    device = adb_client.get_device()
    screenshot = device.screencap()
    assert screenshot is not None
    assert len(screenshot) > 0


def test_tap(device_actions):
    """Test tap action"""
    # Tap at center of screen
    result = device_actions.tap(540, 1200)
    assert result is True


def test_app_launch(app_launcher):
    """Test app launching"""
    # Launch Settings app
    result = app_launcher.launch_app("settings")
    assert result is True


def test_app_mapping(app_launcher):
    """Test app name mapping"""
    package = app_launcher.get_package_name("settings")
    assert package == "com.android.settings"
