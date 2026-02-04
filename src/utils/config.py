"""
Configuration Management - Load and manage configuration files
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml (default: config/config.yaml)
        """
        # Load environment variables
        load_dotenv()
        
        # Determine config file path
        if config_path is None:
            # Get project root (assuming this file is in src/utils/)
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
            else:
                print(f"Warning: Config file not found at {self.config_path}")
                self.config = {}
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (supports dot notation, e.g., "agent.wake_word")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_app_mapping(self, app_name: str) -> Optional[str]:
        """
        Get app package name from friendly name
        
        Args:
            app_name: Friendly app name
            
        Returns:
            Package name or None
        """
        apps = self.get("apps", {})
        return apps.get(app_name.lower())
    
    def get_prompt(self, prompt_name: str) -> str:
        """
        Get prompt template
        
        Args:
            prompt_name: Name of prompt (e.g., "screen_description")
            
        Returns:
            Prompt text or empty string
        """
        prompts = self.get("prompts", {})
        return prompts.get(prompt_name, "")
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable
        
        Args:
            key: Environment variable name
            default: Default value
            
        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment"""
        return self.get_env("OPENAI_API_KEY")
    
    
    def get_adb_config(self) -> Dict[str, Any]:
        """Get ADB configuration"""
        # Device serial can come from config.yaml or environment variable
        device_serial = self.get("device.serial") or self.get_env("ADB_DEVICE_SERIAL")
        
        return {
            "host": self.get_env("ADB_HOST", "127.0.0.1"),
            "port": int(self.get_env("ADB_PORT", "5037")),
            "path": self.get_env("ADB_PATH"),
            "device_serial": device_serial
        }
