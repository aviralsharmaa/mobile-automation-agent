"""
Error Recovery - Handle errors and unexpected UI changes
"""
import time
from typing import Dict, Optional, Callable
from src.vision.screen_analyzer import ScreenAnalyzer
from src.device.actions import DeviceActions
from src.utils.logging import logger


class ErrorRecovery:
    """Error recovery and retry logic"""
    
    def __init__(
        self,
        screen_analyzer: ScreenAnalyzer,
        device_actions: DeviceActions,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize error recovery
        
        Args:
            screen_analyzer: Screen analyzer instance
            device_actions: Device actions instance
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.screen_analyzer = screen_analyzer
        self.device_actions = device_actions
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def handle_popup(
        self,
        device,
        action: Optional[Callable] = None
    ) -> bool:
        """
        Detect and handle popups/alerts
        
        Args:
            device: ADB device instance
            action: Optional action to take (default: dismiss/back)
            
        Returns:
            True if popup handled
        """
        try:
            if self.screen_analyzer.detect_popup(device):
                logger.info("Popup detected, attempting to dismiss")
                
                # Try to find dismiss/close button
                result = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                elements = result.get("elements", [])
                
                # Look for close/dismiss buttons
                for element in elements:
                    desc = element.get("description", "").lower()
                    if any(word in desc for word in ["close", "dismiss", "ok", "cancel", "x"]):
                        x = element.get("x", 0)
                        y = element.get("y", 0)
                        self.device_actions.tap(x, y)
                        time.sleep(1.0)
                        return True
                
                # Fallback: press back button
                if action:
                    action()
                else:
                    self.device_actions.back()
                
                time.sleep(1.0)
                return True
        except Exception as e:
            logger.error(f"Error handling popup: {e}")
        
        return False
    
    def retry_with_backoff(
        self,
        action: Callable,
        *args,
        **kwargs
    ) -> Optional[any]:
        """
        Retry action with exponential backoff
        
        Args:
            action: Function to retry
            *args: Arguments for action
            **kwargs: Keyword arguments for action
            
        Returns:
            Result of action or None if all retries failed
        """
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                result = action(*args, **kwargs)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < self.max_retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
        
        logger.error(f"Action failed after {self.max_retries} attempts")
        return None
    
    def wait_for_screen_change(
        self,
        device,
        timeout: float = 10.0,
        check_interval: float = 0.5
    ) -> bool:
        """
        Wait for screen to change
        
        Args:
            device: ADB device instance
            timeout: Maximum time to wait
            check_interval: How often to check
            
        Returns:
            True if screen changed
        """
        # Capture initial screen
        initial_screenshot = self.screen_analyzer.capture_screenshot(device)
        if not initial_screenshot:
            return False
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            time.sleep(check_interval)
            current_screenshot = self.screen_analyzer.capture_screenshot(device)
            
            if current_screenshot and initial_screenshot != current_screenshot:
                return True
        
        return False
    
    def handle_element_not_found(
        self,
        device,
        element_description: str,
        alternative_strategies: Optional[list] = None
    ) -> bool:
        """
        Handle case when element is not found
        
        Args:
            device: ADB device instance
            element_description: Description of missing element
            alternative_strategies: List of alternative strategies to try
            
        Returns:
            True if alternative strategy succeeded
        """
        logger.warning(f"Element not found: {element_description}")
        
        # Check for popups first
        if self.handle_popup(device):
            logger.info("Popup dismissed, retrying")
            return True
        
        # Try alternative strategies
        if alternative_strategies:
            for strategy in alternative_strategies:
                try:
                    if strategy(device):
                        logger.info("Alternative strategy succeeded")
                        return True
                except Exception as e:
                    logger.error(f"Alternative strategy failed: {e}")
        
        # Try scrolling to reveal more content
        logger.info("Attempting to scroll to find element")
        self.device_actions.swipe_up()
        time.sleep(1.0)
        
        return False
    
    def recover_from_error(
        self,
        device,
        error_type: str,
        context: Optional[Dict] = None
    ) -> bool:
        """
        General error recovery
        
        Args:
            device: ADB device instance
            error_type: Type of error
            context: Additional context
            
        Returns:
            True if recovery successful
        """
        logger.info(f"Attempting to recover from error: {error_type}")
        
        recovery_strategies = {
            "popup": lambda: self.handle_popup(device),
            "timeout": lambda: self.device_actions.back(),
            "element_not_found": lambda: self.handle_element_not_found(device, ""),
            "network": lambda: self.device_actions.wait(2.0),
        }
        
        strategy = recovery_strategies.get(error_type)
        if strategy:
            return strategy()
        
        # Default: try back button
        self.device_actions.back()
        time.sleep(1.0)
        return True
