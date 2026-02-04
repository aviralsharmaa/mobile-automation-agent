"""
App Launcher - App management and package name mapping
"""
from typing import Dict, Optional, List
from ppadb.device import Device


class AppLauncher:
    """App management and launching"""
    
    # Common app package mappings
    DEFAULT_APP_MAPPINGS = {
        "settings": "com.android.settings",
        "gmail": "com.google.android.gm",
        "whatsapp": "com.whatsapp",
        "chatgpt": "com.openai.chatgpt",
        "chrome": "com.android.chrome",
        "youtube": "com.google.android.youtube",
        "maps": "com.google.android.apps.maps",
        "camera": "com.android.camera2",
        "phone": "com.android.dialer",
        "contacts": "com.android.contacts",
        "messages": "com.android.mms",
        "calendar": "com.google.android.calendar",
        "play store": "com.android.vending",
        "files": "com.android.documentsui",
    }
    
    def __init__(self, device: Device, app_mappings: Optional[Dict[str, str]] = None):
        """
        Initialize app launcher
        
        Args:
            device: ADB device instance
            app_mappings: Custom app mappings (friendly name -> package name)
        """
        self.device = device
        self.app_mappings = {**self.DEFAULT_APP_MAPPINGS}
        if app_mappings:
            self.app_mappings.update(app_mappings)
    
    def add_mapping(self, friendly_name: str, package_name: str):
        """Add or update app mapping"""
        self.app_mappings[friendly_name.lower()] = package_name
    
    def get_package_name(self, app_name: str) -> Optional[str]:
        """
        Get package name from friendly name with fuzzy matching
        
        Args:
            app_name: Friendly app name (e.g., "settings", "gmail", "chatgpt")
            
        Returns:
            Package name or None if not found
        """
        app_name_lower = app_name.lower().strip()
        
        # Direct lookup
        if app_name_lower in self.app_mappings:
            return self.app_mappings[app_name_lower]
        
        # Remove common words that might interfere
        words_to_remove = ["the", "a", "an", "open", "launch", "start"]
        words = [w for w in app_name_lower.split() if w not in words_to_remove]
        app_name_clean = " ".join(words)
        
        if app_name_clean in self.app_mappings:
            return self.app_mappings[app_name_clean]
        
        # Fuzzy matching for common variations
        # Remove spaces and special characters for matching
        normalized = app_name_clean.replace(" ", "").replace("-", "").replace("_", "")
        
        # Try exact normalized match first
        for key, value in self.app_mappings.items():
            key_normalized = key.replace(" ", "").replace("-", "").replace("_", "")
            if normalized == key_normalized:
                return value
        
        # Try substring matching (e.g., "chargerbt" contains "chatgpt" characters)
        for key, value in self.app_mappings.items():
            key_normalized = key.replace(" ", "").replace("-", "").replace("_", "")
            # Check if normalized strings match or one contains the other
            if normalized in key_normalized or key_normalized in normalized:
                return value
        
        # Try character-based similarity (for misrecognitions like "chargerbt" -> "chatgpt")
        best_match = None
        best_score = 0.0
        
        for key, value in self.app_mappings.items():
            score = self._similarity_score(normalized, key.replace(" ", "").replace("-", "").replace("_", ""))
            if score > best_score and score > 0.5:  # 50% similarity threshold
                best_score = score
                best_match = value
        
        if best_match:
            return best_match
        
        # Check for partial matches (e.g., "charge pt" -> "chatgpt")
        for key, value in self.app_mappings.items():
            if key in app_name_lower or app_name_lower in key:
                return value
        
        return None
    
    def _similarity_score(self, word1: str, word2: str) -> float:
        """
        Calculate similarity score between two words
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not word1 or not word2:
            return 0.0
        
        if word1 == word2:
            return 1.0
        
        # Check if one contains the other (strong match)
        if word1 in word2 or word2 in word1:
            return 0.8
        
        # Character overlap
        set1 = set(word1)
        set2 = set(word2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union)
        
        # Length similarity
        len_sim = 1.0 - abs(len(word1) - len(word2)) / max(len(word1), len(word2))
        
        # Combined score
        return (jaccard * 0.7 + len_sim * 0.3)
    
    def launch_app(self, app_name: str) -> bool:
        """
        Launch app by friendly name or package name
        
        Args:
            app_name: Friendly name or package name
            
        Returns:
            True if successful
        """
        package_name = self.get_package_name(app_name) or app_name
        
        try:
            # Use monkey command to launch app
            result = self.device.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
            
            # Check if app launched successfully
            if "No activities found" in result or "Error" in result:
                print(f"Failed to launch {app_name} ({package_name})")
                return False
            
            return True
        except Exception as e:
            print(f"Error launching app {app_name}: {e}")
            return False
    
    def is_app_running(self, app_name: str) -> bool:
        """
        Check if app is currently running
        
        Args:
            app_name: Friendly name or package name
            
        Returns:
            True if app is running
        """
        package_name = self.get_package_name(app_name) or app_name
        
        try:
            # Get list of running apps
            result = self.device.shell("dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp'")
            return package_name in result
        except Exception:
            # Fallback method
            try:
                result = self.device.shell(f"pidof {package_name}")
                return bool(result.strip())
            except Exception:
                return False
    
    def get_installed_apps(self) -> List[str]:
        """
        Get list of installed app packages
        
        Returns:
            List of package names
        """
        try:
            result = self.device.shell("pm list packages")
            packages = []
            for line in result.strip().split('\n'):
                if line.startswith('package:'):
                    packages.append(line.replace('package:', ''))
            return packages
        except Exception as e:
            print(f"Error getting installed apps: {e}")
            return []
    
    def close_app(self, app_name: str) -> bool:
        """
        Close/force stop app
        
        Args:
            app_name: Friendly name or package name
            
        Returns:
            True if successful
        """
        package_name = self.get_package_name(app_name) or app_name
        
        try:
            self.device.shell(f"am force-stop {package_name}")
            return True
        except Exception as e:
            print(f"Error closing app {app_name}: {e}")
            return False
    
    def get_current_app(self) -> Optional[str]:
        """
        Get currently focused app package name
        
        Returns:
            Package name or None
        """
        try:
            result = self.device.shell("dumpsys window windows | grep -E 'mCurrentFocus'")
            # Parse the output to extract package name
            if "mCurrentFocus" in result:
                # Format: mCurrentFocus=Window{... package.name/...}
                parts = result.split()
                for part in parts:
                    if '/' in part:
                        package = part.split('/')[0].split('}')[-1]
                        if '.' in package:
                            return package
            return None
        except Exception as e:
            print(f"Error getting current app: {e}")
            return None
