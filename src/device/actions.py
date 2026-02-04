"""
Device Actions - Primitive device interaction methods
"""
import time
from typing import Optional, Tuple
from ppadb.device import Device


class DeviceActions:
    """Device interaction primitives"""
    
    # Pixel 7A screen dimensions (defaults, will be updated from device)
    SCREEN_WIDTH = 1080
    SCREEN_HEIGHT = 2400
    
    # Safe areas (accounting for status bar and navigation bar)
    STATUS_BAR_HEIGHT = 50
    NAVIGATION_BAR_HEIGHT = 100
    
    def __init__(self, device: Device):
        """
        Initialize device actions
        
        Args:
            device: ADB device instance
        """
        self.device = device
        # Get real device dimensions and status bar height
        self._update_device_dimensions()
    
    def _update_device_dimensions(self):
        """Get real device size and status bar height from device"""
        try:
            # Get physical device size
            result = self.device.shell("wm size")
            if "Physical size:" in result:
                size_str = result.split("Physical size:")[1].strip()
                width, height = size_str.split("x")
                self.SCREEN_WIDTH = int(width.strip())
                self.SCREEN_HEIGHT = int(height.strip())
                print(f"[Device] Real device size: {self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT}")
            elif "Override size:" in result:
                # Some devices show override size
                size_str = result.split("Override size:")[1].strip()
                width, height = size_str.split("x")
                self.SCREEN_WIDTH = int(width.strip())
                self.SCREEN_HEIGHT = int(height.strip())
                print(f"[Device] Device size (override): {self.SCREEN_WIDTH}x{self.SCREEN_HEIGHT}")
        except Exception as e:
            print(f"[Device] Warning: Could not get device size, using defaults: {e}")
        
        try:
            # Get status bar height
            # Note: grep doesn't work on Windows ADB, so we get full output and parse in Python
            result = self.device.shell("dumpsys window displays")
            import re
            
            # Look for mStableInsets or mDisplayInsets in the output
            # Format examples:
            #   mStableInsets=Rect(0, 72, 0, 48)
            #   mStableInsets=0,72,0,48
            #   mDisplayInsets=Rect(0, 72, 0, 48)
            
            # Try mStableInsets first
            match = re.search(r'mStableInsets\s*=\s*Rect\s*\(\s*\d+\s*,\s*(\d+)', result)
            if not match:
                # Try alternative format: mStableInsets=0,72,0,48
                match = re.search(r'mStableInsets\s*=\s*(\d+)\s*,\s*(\d+)', result)
                if match:
                    top_inset = int(match.group(2))  # Second number is top inset
                else:
                    match = None
            else:
                top_inset = int(match.group(1))
            
            if match and top_inset > 0:
                self.STATUS_BAR_HEIGHT = top_inset
                print(f"[Device] Status bar height (from mStableInsets): {self.STATUS_BAR_HEIGHT}px")
            else:
                # Try mDisplayInsets as fallback
                match = re.search(r'mDisplayInsets\s*=\s*Rect\s*\(\s*\d+\s*,\s*(\d+)', result)
                if not match:
                    match = re.search(r'mDisplayInsets\s*=\s*(\d+)\s*,\s*(\d+)', result)
                    if match:
                        top_inset = int(match.group(2))
                    else:
                        match = None
                else:
                    top_inset = int(match.group(1))
                
                if match and top_inset > 0:
                    self.STATUS_BAR_HEIGHT = top_inset
                    print(f"[Device] Status bar height (from mDisplayInsets): {self.STATUS_BAR_HEIGHT}px")
                else:
                    print(f"[Device] Could not parse status bar height from dumpsys, using default {self.STATUS_BAR_HEIGHT}px")
        except Exception as e:
            print(f"[Device] Warning: Could not get status bar height, using default {self.STATUS_BAR_HEIGHT}px: {e}")
    
    def map_vision_to_device_coords(self, x_vision: int, y_vision: int, screenshot_width: int, screenshot_height: int) -> Tuple[int, int]:
        """
        Correctly maps vision coordinates to real device coordinates.
        Accounts for status bar offset since screenshots exclude status bar.
        
        Args:
            x_vision: X coordinate from vision API (screenshot space)
            y_vision: Y coordinate from vision API (screenshot space)
            screenshot_width: Width of screenshot
            screenshot_height: Height of screenshot
            
        Returns:
            Tuple of (device_x, device_y) in real device coordinate space
        """
        # Calculate scale factors
        scale_x = self.SCREEN_WIDTH / screenshot_width if screenshot_width > 0 else 1.0
        scale_y = self.SCREEN_HEIGHT / screenshot_height if screenshot_height > 0 else 1.0
        
        # Scale coordinates
        device_x = int(x_vision * scale_x)
        # CRITICAL: Add status bar offset to Y coordinate
        # Screenshots start below status bar, but ADB taps use full screen coordinates
        device_y = int(y_vision * scale_y + self.STATUS_BAR_HEIGHT)
        
        return device_x, device_y
    
    def tap(self, x: int, y: int, delay: float = 0.1) -> bool:
        """
        Tap at coordinates
        
        Args:
            x: X coordinate (0-1080)
            y: Y coordinate (0-2400)
            delay: Delay after tap in seconds
            
        Returns:
            True if successful
        """
        try:
            # Validate coordinates
            x = max(0, min(x, self.SCREEN_WIDTH))
            y = max(0, min(y, self.SCREEN_HEIGHT))
            
            # Validate coordinates are not zero (invalid)
            if x == 0 and y == 0:
                print(f"[TAP ERROR] Invalid coordinates: ({x}, {y}) - cannot tap at origin")
                return False
            
            if x <= 0 or y <= 0:
                print(f"[TAP ERROR] Invalid coordinates: ({x}, {y}) - coordinates must be positive")
                return False
            
            print(f"[TAP] Executing: adb shell input tap {x} {y}")
            result = self.device.shell(f"input tap {x} {y}")
            
            # Check if command succeeded (empty result usually means success)
            if result and "error" in result.lower():
                print(f"[TAP ERROR] ADB command failed: {result}")
                return False
            
            time.sleep(delay)
            # Note: "Successfully tapped" means ADB command succeeded, not that the tap had the intended effect
            print(f"[TAP] ADB tap command executed at ({x}, {y}) - verify screen changed to confirm success")
            return True
        except Exception as e:
            print(f"[TAP ERROR] Exception tapping at ({x}, {y}): {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """
        Swipe gesture
        
        Args:
            x1: Start X coordinate
            y1: Start Y coordinate
            x2: End X coordinate
            y2: End Y coordinate
            duration: Swipe duration in milliseconds
            
        Returns:
            True if successful
        """
        try:
            self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"Error swiping from ({x1}, {y1}) to ({x2}, {y2}): {e}")
            return False
    
    def swipe_up(self, distance: int = 500) -> bool:
        """Swipe up (scroll down)"""
        center_x = self.SCREEN_WIDTH // 2
        start_y = self.SCREEN_HEIGHT - self.NAVIGATION_BAR_HEIGHT - 200
        end_y = start_y - distance
        return self.swipe(center_x, start_y, center_x, end_y)
    
    def swipe_down(self, distance: int = 500) -> bool:
        """Swipe down (scroll up)"""
        center_x = self.SCREEN_WIDTH // 2
        start_y = self.STATUS_BAR_HEIGHT + 200
        end_y = start_y + distance
        return self.swipe(center_x, start_y, center_x, end_y)
    
    def type_text(self, text: str, clear_first: bool = False) -> bool:
        """
        Type text into current input field
        
        Args:
            text: Text to type
            clear_first: Clear field before typing
            
        Returns:
            True if successful
        """
        try:
            if clear_first:
                self.clear_text_field()
            
            # For ADB 'input text' command:
            # - Spaces must be replaced with %s
            # - Other special characters work fine without escaping
            # - DO NOT escape @ for emails - it causes literal backslash to appear!
            escaped_text = text.replace(' ', '%s')
            
            print(f"[Type] Typing text via ADB: {text[:20]}..." if len(text) > 20 else f"[Type] Typing text via ADB: {text}")
            
            # Use the shell command directly without extra escaping
            # ADB input text handles most characters correctly
            result = self.device.shell(f'input text {escaped_text}')
            time.sleep(0.3)
            
            # Verify by checking if any error in result
            if result and "error" in result.lower():
                print(f"[Type] Warning: ADB returned: {result}")
                return False
            
            return True
        except Exception as e:
            print(f"[Type] Error typing text: {e}")
    
    def clear_text_field(self) -> bool:
        """
        Clear all text in current focused input field.
        Uses multiple methods for reliability on Android.
        
        Returns:
            True if successful
        """
        try:
            print("[Type] Clearing existing text in field...")
            
            # Method 1: Try CTRL+A (works on emulators and some devices)
            self.device.shell("input keyevent KEYCODE_CTRL_A")
            time.sleep(0.1)
            self.device.shell("input keyevent KEYCODE_DEL")
            time.sleep(0.1)
            
            # Method 2: Move to end and delete backwards (more reliable on real devices)
            # Move cursor to end of text
            self.device.shell("input keyevent KEYCODE_MOVE_END")
            time.sleep(0.05)
            
            # Delete multiple times to clear any remaining text
            # Most email fields are <50 chars
            for _ in range(60):
                self.device.shell("input keyevent KEYCODE_DEL")
            time.sleep(0.1)
            
            return True
        except Exception as e:
            print(f"[Type] Error clearing text: {e}")
            return False
    
    def type_digits_keyevent(self, digits_str: str) -> bool:
        """
        Type digits using keyevents (KEYCODE_0 to KEYCODE_9).
        Use this when 'input text' does not work (e.g. Paytm amount field).
        
        Args:
            digits_str: String of digits only, e.g. "10" or "250"
        Returns:
            True if successful
        """
        keycode_map = {
            "0": "KEYCODE_0", "1": "KEYCODE_1", "2": "KEYCODE_2", "3": "KEYCODE_3",
            "4": "KEYCODE_4", "5": "KEYCODE_5", "6": "KEYCODE_6", "7": "KEYCODE_7",
            "8": "KEYCODE_8", "9": "KEYCODE_9",
        }
        try:
            for char in digits_str:
                if char in keycode_map:
                    self.device.shell(f"input keyevent {keycode_map[char]}")
                    time.sleep(0.15)
            return True
        except Exception as e:
            print(f"[Type] Error typing digits via keyevent: {e}")
            return False
    
    def press_key(self, keycode: str) -> bool:
        """
        Press hardware/system key
        
        Args:
            keycode: Android keycode (e.g., "KEYCODE_BACK", "KEYCODE_HOME")
            
        Returns:
            True if successful
        """
        try:
            self.device.shell(f"input keyevent {keycode}")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"Error pressing key {keycode}: {e}")
            return False
    
    def back(self) -> bool:
        """Press back button"""
        return self.press_key("KEYCODE_BACK")
    
    def home(self) -> bool:
        """Press home button"""
        return self.press_key("KEYCODE_HOME")
    
    def recent_apps(self) -> bool:
        """Open recent apps"""
        return self.press_key("KEYCODE_APP_SWITCH")
    
    def long_press(self, x: int, y: int, duration: int = 500) -> bool:
        """
        Long press at coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Press duration in milliseconds
            
        Returns:
            True if successful
        """
        try:
            # Long press is implemented as swipe with same start/end points
            self.device.shell(f"input swipe {x} {y} {x} {y} {duration}")
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"Error long pressing at ({x}, {y}): {e}")
            return False
    
    def wait(self, seconds: float):
        """Wait for specified time"""
        time.sleep(seconds)
