"""
Authentication Handler - Secure credential handling for login flows
"""
from typing import Dict, Optional, Tuple, Any
from src.vision.screen_analyzer import ScreenAnalyzer
from src.vision.element_detector import ElementDetector
from src.device.actions import DeviceActions
from src.voice.tts import TextToSpeech
from src.voice.stt import SpeechToText
from src.utils.logging import logger
from ppadb.device import Device


class AuthHandler:
    """Handle authentication flows securely"""
    
    def __init__(
        self,
        screen_analyzer: ScreenAnalyzer,
        element_detector: ElementDetector,
        device_actions: DeviceActions,
        tts: TextToSpeech,
        stt: SpeechToText
    ):
        """
        Initialize auth handler
        
        Args:
            screen_analyzer: Screen analyzer instance
            element_detector: Element detector instance
            device_actions: Device actions instance
            tts: Text-to-speech instance
            stt: Speech-to-text instance
        """
        self.screen_analyzer = screen_analyzer
        self.element_detector = element_detector
        self.device_actions = device_actions
        self.tts = tts
        self.stt = stt
    
    def detect_login_screen(self, device: Device) -> bool:
        """
        Detect if current screen is a login screen
        
        Args:
            device: ADB device instance
            
        Returns:
            True if login screen detected
        """
        return self.screen_analyzer.detect_login_screen(device)
    
    def get_login_stage(self, device: Device) -> Dict[str, Any]:
        """
        Get detailed login stage information
        
        Args:
            device: ADB device instance
            
        Returns:
            Dictionary with login stage information
        """
        screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
        return {
            "is_login": screen_analysis.get("is_login_screen", False),
            "stage": screen_analysis.get("login_stage", "none"),
            "has_email": screen_analysis.get("has_email_field", False),
            "has_password": screen_analysis.get("has_password_field", False),
            "description": screen_analysis.get("description", "")
        }
    
    def handle_login(
        self,
        device: Device,
        email: Optional[str] = None,
        password: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Handle login flow
        
        Args:
            device: ADB device instance
            email: Email (will prompt if None)
            password: Password (will prompt if None)
            
        Returns:
            Tuple of (success, message)
        """
        # Analyze screen to determine login stage
        screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
        
        description = screen_analysis.get("description", "").lower()
        is_login = screen_analysis.get("is_login_screen", False)
        login_stage = screen_analysis.get("login_stage", "initial")
        has_email_field = screen_analysis.get("has_email_field", False)
        has_password_field = screen_analysis.get("has_password_field", False)
        
        # Enhanced detection: Check description text for password-only indicators
        # If screen says "enter password" or "password" and no email field, it's password-only stage
        password_keywords = ["password", "enter password", "enter your password", "type password"]
        email_keywords = ["email", "username", "enter email", "enter username"]
        
        has_password_text = any(keyword in description for keyword in password_keywords)
        has_email_text = any(keyword in description for keyword in email_keywords)
        
        if has_password_text and not has_email_text and not has_email_field:
            login_stage = "password_only"
            has_password_field = True
            is_login = True
            logger.info("Detected password-only screen from description text")
        
        logger.info(f"Screen analysis - Login: {is_login}, Stage: {login_stage}, Has email: {has_email_field}, Has password: {has_password_field}")
        logger.info(f"Screen description: {description[:150]}...")
        
        if not is_login and login_stage == "none":
            return False, "Not a login screen"
        
        try:
            # Handle email entry (only if we're at initial or email_only stage)
            if login_stage in ["initial", "email_only"] and not email:
                # Check if email field exists
                if has_email_field or login_stage == "email_only":
                    self.tts.speak("Login required. Please provide your email address.")
                    logger.log_credential_request("email")
                    email_text = self.stt.listen(timeout=10.0, phrase_time_limit=10.0)
                    if not email_text:
                        return False, "No email provided"
                    email = email_text.strip()
            
            # Enter email if we have it and we're at initial or email_only stage
            if email and login_stage in ["initial", "email_only"]:
                email_field = self.element_detector.find_text_field(device, "email")
                if not email_field:
                    # Try to find any text field
                    email_field = self.element_detector.find_text_field(device)
                
                if email_field:
                    self.device_actions.tap(email_field[0], email_field[1])
                    self.device_actions.wait(0.5)
                    self.device_actions.type_text(email, clear_first=True)
                    logger.info(f"Email entered: {email[:3]}*** (redacted)")
                    
                    # Press enter or next to proceed to password
                    self.device_actions.press_key("KEYCODE_ENTER")
                    self.device_actions.wait(2.0)  # Wait for screen transition
                    
                    # Re-analyze screen after email entry
                    screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                    login_stage = screen_analysis.get("login_stage", "initial")
                    has_password_field = screen_analysis.get("has_password_field", False)
                    logger.info(f"After email entry, login stage: {login_stage}")
                else:
                    return False, "Could not find email field"
            
            # Handle password entry (if we're at password_only stage or initial stage)
            # Skip email if we're already at password stage
            if login_stage == "password_only" or (login_stage in ["initial"] and has_password_field):
                if not password:
                    self.tts.speak("Enter your password.")
                    logger.log_credential_request("password")
                    password_text = self.stt.listen(timeout=10.0, phrase_time_limit=10.0)
                    if not password_text:
                        return False, "No password provided"
                    password = password_text.strip()
                
                # Find password field
                password_field = self.element_detector.find_text_field(device, "password")
                if not password_field:
                    # Try to find any text field (might be the only one visible on password-only screen)
                    elements = self.element_detector.find_all_elements(device, element_type="text_field")
                    if elements:
                        # Use the first text field (likely password if we're at password_only stage)
                        password_field = (elements[0].get("x", 0), elements[0].get("y", 0))
                        logger.info(f"Using first text field as password field: {password_field}")
                
                if password_field:
                    self.device_actions.tap(password_field[0], password_field[1])
                    self.device_actions.wait(0.5)
                    self.device_actions.type_text(password, clear_first=True)
                    logger.log_credential_request("password")
                    logger.info("Password entered successfully")
                    # Clear password from memory immediately
                    password = None
                else:
                    logger.error("Could not find password field")
                    return False, "Could not find password field"
            
            # Find and click login button
            login_button = self.element_detector.find_button(device, "login")
            if not login_button:
                login_button = self.element_detector.find_button(device, "sign in")
            if not login_button:
                login_button = self.element_detector.find_button(device, "submit")
            
            if login_button:
                self.device_actions.tap(login_button[0], login_button[1])
                self.device_actions.wait(3.0)  # Wait for login to process
                
                # Verify login success
                if not self.detect_login_screen(device):
                    self.tts.speak("Login successful.")
                    return True, "Login successful"
                else:
                    return False, "Login may have failed - still on login screen"
            else:
                return False, "Could not find login button"
            
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return False, f"Login error: {str(e)}"
    
    def handle_otp(self, device: Device, otp: Optional[str] = None) -> Tuple[bool, str]:
        """
        Handle OTP/2FA flow
        
        Args:
            device: ADB device instance
            otp: OTP code (will prompt if None)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not otp:
                self.tts.speak("Please provide the verification code.")
                logger.log_credential_request("OTP")
                otp_text = self.stt.listen(timeout=15.0, phrase_time_limit=15.0)
                if not otp_text:
                    return False, "No OTP provided"
                otp = otp_text.strip().replace(" ", "").replace("-", "")
            
            # Find OTP input field
            otp_field = self.element_detector.find_text_field(device, "code")
            if not otp_field:
                otp_field = self.element_detector.find_text_field(device, "otp")
            if not otp_field:
                otp_field = self.element_detector.find_text_field(device, "verification")
            
            if otp_field:
                self.device_actions.tap(otp_field[0], otp_field[1])
                self.device_actions.wait(0.5)
                self.device_actions.type_text(otp, clear_first=True)
                logger.log_credential_request("OTP")
                
                # Find submit/verify button
                submit_button = self.element_detector.find_button(device, "verify")
                if not submit_button:
                    submit_button = self.element_detector.find_button(device, "submit")
                if not submit_button:
                    submit_button = self.element_detector.find_button(device, "continue")
                
                if submit_button:
                    self.device_actions.tap(submit_button[0], submit_button[1])
                    self.device_actions.wait(2.0)
                    return True, "OTP submitted"
                else:
                    return False, "Could not find submit button"
            else:
                return False, "Could not find OTP field"
                
        except Exception as e:
            logger.error(f"Error during OTP flow: {e}")
            return False, f"OTP error: {str(e)}"
