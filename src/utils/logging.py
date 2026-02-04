"""
Secure Logging - Logging utilities that never log credentials
"""
import logging
import re
from typing import Any


class SecureLogger:
    """Logger that redacts sensitive information"""
    
    # Patterns to redact
    SENSITIVE_PATTERNS = [
        r'password["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'token["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'secret["\']?\s*[:=]\s*["\']?([^"\']+)',
        r'credential["\']?\s*[:=]\s*["\']?([^"\']+)',
    ]
    
    def __init__(self, name: str = "mobile_automation_agent", level: int = logging.INFO):
        """
        Initialize secure logger
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _redact(self, message: str) -> str:
        """
        Redact sensitive information from log message
        
        Args:
            message: Log message
            
        Returns:
            Redacted message
        """
        redacted = message
        for pattern in self.SENSITIVE_PATTERNS:
            redacted = re.sub(pattern, r'\1[REDACTED]', redacted, flags=re.IGNORECASE)
        return redacted
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(self._redact(str(message)), *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(self._redact(str(message)), *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(self._redact(str(message)), *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(self._redact(str(message)), *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(self._redact(str(message)), *args, **kwargs)
    
    def log_credential_request(self, credential_type: str):
        """
        Log that a credential was requested (but not the actual credential)
        
        Args:
            credential_type: Type of credential (e.g., "email", "password", "OTP")
        """
        self.info(f"Credential requested: {credential_type} (value not logged)")


# Global logger instance
logger = SecureLogger()
