"""
ADB Client - Connection and basic device operations
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, List
from ppadb.client import Client as AdbClient
from ppadb.device import Device


class ADBClient:
    """Wrapper for ADB connection and device management"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5037, adb_path: Optional[str] = None, device_serial: Optional[str] = None):
        """
        Initialize ADB client
        
        Args:
            host: ADB server host (default: 127.0.0.1)
            port: ADB server port (default: 5037)
            adb_path: Optional path to adb executable
            device_serial: Optional specific device serial to connect to (e.g., "41311JEHN00968")
        """
        self.host = host
        self.port = port
        self.adb_path = adb_path or self._find_adb_path()
        self.device_serial = device_serial  # Specific device to use
        self.client: Optional[AdbClient] = None
        self.device: Optional[Device] = None
        
    def _find_adb_path(self) -> Optional[str]:
        """Auto-detect ADB path on Windows"""
        # Common Android SDK locations on Windows
        possible_paths = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk", "platform-tools", "adb.exe"),
            os.path.join(os.environ.get("ANDROID_HOME", ""), "platform-tools", "adb.exe"),
            os.path.join(os.environ.get("ANDROID_SDK_ROOT", ""), "platform-tools", "adb.exe"),
        ]
        
        # Check if adb is in PATH
        try:
            subprocess.run(["adb", "version"], capture_output=True, check=True)
            return "adb"  # Use system PATH
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check common locations
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def connect(self) -> bool:
        """
        Connect to ADB server and get device
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = AdbClient(host=self.host, port=self.port)
            devices = self.client.devices()
            
            if len(devices) == 0:
                print("No devices found. Is the device connected?")
                return False
            
            # List all available devices
            print(f"Found {len(devices)} device(s):")
            for i, d in enumerate(devices):
                print(f"  {i+1}. {d.serial}")
            
            # If specific device serial provided, use it
            if self.device_serial:
                for d in devices:
                    if d.serial == self.device_serial:
                        self.device = d
                        print(f"✓ Connected to specified device: {self.device.serial}")
                        return True
                print(f"ERROR: Device '{self.device_serial}' not found!")
                print(f"Available devices: {[d.serial for d in devices]}")
                return False
            
            # No specific device - use first one
            self.device = devices[0]
            print(f"✓ Connected to device: {self.device.serial}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to ADB: {e}")
            return False
    
    def get_device(self) -> Optional[Device]:
        """Get the connected device"""
        if not self.device:
            if not self.connect():
                return None
        return self.device
    
    def is_connected(self) -> bool:
        """Check if device is connected"""
        return self.device is not None
    
    def get_device_info(self) -> dict:
        """Get device information"""
        if not self.device:
            return {}
        
        try:
            return {
                "serial": self.device.serial,
                "model": self.device.shell("getprop ro.product.model").strip(),
                "android_version": self.device.shell("getprop ro.build.version.release").strip(),
                "sdk_version": self.device.shell("getprop ro.build.version.sdk").strip(),
            }
        except Exception as e:
            print(f"Error getting device info: {e}")
            return {}
    
    def disconnect(self):
        """Disconnect from device"""
        self.device = None
        self.client = None
