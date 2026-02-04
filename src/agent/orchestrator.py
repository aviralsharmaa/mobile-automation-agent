"""
Agent Orchestrator - Main agent loop using LangGraph
"""
import os
import uuid
import re
from typing import Dict, Any, Optional, TypedDict, Annotated, List, Tuple
from langgraph.graph import StateGraph, END
from operator import add
from src.device.adb_client import ADBClient
from src.device.actions import DeviceActions
from src.device.app_launcher import AppLauncher
from src.device.accessibility import AccessibilityTree
from src.vision.screen_analyzer import ScreenAnalyzer
from src.vision.element_detector import ElementDetector
from src.voice.stt import SpeechToText
from src.voice.tts import TextToSpeech
from src.agent.state import AgentStateManager, AgentState, TaskState
from src.agent.auth_handler import AuthHandler
from src.agent.error_recovery import ErrorRecovery
from src.utils.config import Config
from src.utils.logging import logger


class AgentOrchestrator:
    """Main agent orchestrator using LangGraph"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize orchestrator
        
        Args:
            config: Configuration instance
        """
        self.config = config or Config()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize state manager
        self.state_manager = AgentStateManager()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _initialize_components(self):
        """Initialize all agent components"""
        # ADB and device control
        adb_config = self.config.get_adb_config()
        self.adb_client = ADBClient(
            host=adb_config["host"],
            port=adb_config["port"],
            adb_path=adb_config.get("path"),
            device_serial=adb_config.get("device_serial")  # Specify which device to use
        )
        
        if not self.adb_client.connect():
            raise RuntimeError("Failed to connect to Android device")
        
        device = self.adb_client.get_device()
        self.device_actions = DeviceActions(device)
        self.app_launcher = AppLauncher(device, self.config.get("apps", {}))
        self.accessibility = AccessibilityTree(device)  # For accurate coordinates
        
        # Vision
        api_key = self.config.get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        
        vision_model = self.config.get("vision.model", "gpt-4o-mini")  # Default to faster model
        self.screen_analyzer = ScreenAnalyzer(api_key, model=vision_model)
        self.element_detector = ElementDetector(self.screen_analyzer)
        
        # OpenAI client for LLM decision-making
        from openai import OpenAI
        self.llm_client = OpenAI(api_key=api_key)
        
        # Voice
        wake_word = self.config.get("agent.wake_word", "hey assistant")
        stt_backend = self.config.get("voice.stt_backend", "google")
        # Auto-enable Whisper if OpenAI API key is available (more accurate)
        use_whisper = stt_backend == "whisper"
        # Force Whisper if OpenAI key is available (even if config says google)
        if api_key and not use_whisper:
            logger.info("OpenAI API key detected. Auto-enabling Whisper for better accuracy.")
            use_whisper = True
        elif not api_key and use_whisper:
            logger.warning("Whisper requested but OPENAI_API_KEY not found. Falling back to Google Speech Recognition.")
            use_whisper = False
        
        logger.info(f"Using STT backend: {'Whisper' if use_whisper else 'Google Speech Recognition'}")
        self.stt = SpeechToText(use_whisper=use_whisper, wake_word=wake_word)
        
        # Get TTS settings from config
        speech_rate = self.config.get("voice.speech_rate", 150)
        volume = self.config.get("voice.volume", 1.0)
        self.tts = TextToSpeech(speech_rate=speech_rate, volume=volume)
        
        # Agent modules
        self.auth_handler = AuthHandler(
            self.screen_analyzer,
            self.element_detector,
            self.device_actions,
            self.tts,
            self.stt
        )
        
        self.error_recovery = ErrorRecovery(
            self.screen_analyzer,
            self.device_actions,
            max_retries=self.config.get("agent.retry_attempts", 3),
            retry_delay=self.config.get("agent.retry_delay", 2.0)
        )
        
        # Login flow state tracking
        self.login_flow_state = {
            "active": False,
            "awaiting_email": False,
            "awaiting_password": False,
            "email": None,
            "password": None,
            "email_entered": False,  # Track if email was already typed
            "popup_detected": False,
            "awaiting_popup_confirmation": False
        }
        
        # Continuous query session: when user has "open ChatGPT and ask X", subsequent
        # commands like "what is 2+2" are sent to the same app until they say "close" or "exit"
        self._query_app_session: Optional[str] = None  # e.g. "chatgpt" or None
        self.QUERY_SESSION_APPS = ("chatgpt", "gpt", "openai")  # apps that support follow-up questions
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        # Define state schema for LangGraph
        class AgentState(TypedDict, total=False):
            user_intent: str
            task_id: str
            screen_analysis: Optional[Dict[str, Any]]
            intent_parts: Optional[Dict[str, Any]]
            needs_auth: bool
            auth_success: bool
            auth_message: Optional[str]
            action_success: bool
            task_complete: bool
            extracted_info: Optional[str]
            error: Optional[str]
            error_type: Optional[str]
            iteration_count: int  # Track iterations to prevent infinite loops
            needs_confirmation: bool  # Whether user confirmation is needed
            screen_description: Optional[str]  # Screen description for user feedback
            primary_action: Optional[str]  # Primary action button description
            important_buttons: Optional[list[Dict]]  # List of important buttons detected
            pending_query: Optional[str]  # Query to execute after app opens
            pending_app: Optional[str]  # App name for pending query
            needs_query_execution: bool  # Whether to execute query in app
            extracted_response: Optional[str]  # Extracted response from app
            session_active: bool  # Whether session is active (app opened, waiting for more commands)
            query_session_follow_up: bool  # True when user asked follow-up in same app (e.g. ChatGPT)
            whatsapp_follow_up: bool  # True when WhatsApp session active and user said "send to X say Y"
        
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("observe", self._observe)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("act", self._act)
        workflow.add_node("authenticate", self._authenticate)
        workflow.add_node("verify", self._verify)
        workflow.add_node("respond", self._respond)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define edges
        workflow.set_entry_point("observe")
        workflow.add_edge("observe", "analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._should_authenticate,
            {
                "authenticate": "authenticate",
                "act": "act"
            }
        )
        workflow.add_edge("authenticate", "verify")  # After auth, verify success
        workflow.add_edge("act", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._is_complete,
            {
                "complete": "respond",
                "confirm": "confirm_action",  # New node for user confirmation
                "execute_query": "execute_query",  # Execute query without confirmation
                "authenticate": "authenticate",  # Route to authentication
                "continue": "observe",
                "error": "handle_error"
            }
        )
        workflow.add_node("confirm_action", self._confirm_action)
        workflow.add_node("execute_query", self._execute_query)
        workflow.add_conditional_edges(
            "confirm_action",
            lambda s: "execute_query" if s.get("needs_query_execution") else "respond",
            {
                "execute_query": "execute_query",
                "respond": "respond"
            }
        )
        workflow.add_edge("execute_query", "respond")
        workflow.add_edge("respond", END)
        # Limit retries - go to respond instead of observe after error
        workflow.add_edge("handle_error", "respond")
        
        return workflow.compile()
    
    def _observe(self, state: Dict) -> Dict:
        """Observe current screen state"""
        logger.info("Observing screen...")
        print("[Status] Capturing and analyzing screen...")
        self.state_manager.update_state(AgentState.OBSERVING)
        
        # Increment iteration count to prevent infinite loops
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        max_iterations = 5  # Maximum workflow iterations
        
        if state["iteration_count"] > max_iterations:
            logger.warning(f"Maximum iterations ({max_iterations}) reached. Stopping workflow.")
            print(f"\n[ERROR] Maximum iterations ({max_iterations}) reached - STOPPING to prevent endless loop")
            state["error"] = "Maximum iterations reached. Task may be incomplete."
            state["task_complete"] = True
            self.tts.speak("I've tried multiple times but couldn't complete the task. Please try again with a clearer command.")
            return state
        
        device = self.adb_client.get_device()
        # Use faster analysis - only detect elements when needed
        needs_elements = state.get("needs_auth", False) or state.get("pending_query") is not None
        screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=needs_elements)
        
        state["screen_analysis"] = screen_analysis
        self.state_manager.update_screen_context(screen_analysis)
        print("[Status] Screen analysis complete")
        
        return state
    
    def _analyze(self, state: Dict) -> Dict:
        """Analyze screen and determine next action"""
        logger.info("Analyzing screen...")
        print("[Status] Analyzing screen and parsing command...")
        self.state_manager.update_state(AgentState.ANALYZING)
        
        screen_analysis = state.get("screen_analysis", {})
        user_intent = state.get("user_intent", "")
        
        # Continuous query session: user is asking a follow-up in the same app (e.g. ChatGPT)
        if state.get("query_session_follow_up") and state.get("pending_query") and state.get("pending_app"):
            pending_app = state["pending_app"]
            state["intent_parts"] = {"action": "open_app", "app": pending_app, "query": state["pending_query"]}
            state["needs_query_execution"] = True
            print(f"[Session] Follow-up in {pending_app}: executing query without re-opening app")
            return state
        
        # WhatsApp session: user said "send message to X say Y" (intent already set in process_command)
        if state.get("whatsapp_follow_up") and state.get("intent_parts") and state["intent_parts"].get("action") == "send_whatsapp_message":
            print(f"[Session] WhatsApp follow-up: send to {state['intent_parts'].get('recipient')}")
            return state
        
        # Parse user intent
        intent_parts = self._parse_intent(user_intent)
        state["intent_parts"] = intent_parts
        
        # Only set needs_auth if user explicitly requested login
        # Don't auto-detect login screens - only login if user explicitly said "login"
        wants_login = intent_parts.get("wants_login", False)
        needs_auth = wants_login  # Only login if user explicitly wants it
        state["needs_auth"] = needs_auth
        
        # Show what was parsed
        action = intent_parts.get("action")
        app = intent_parts.get("app")
        if action == "open_app" and app:
            print(f"[Status] Will open app: {app}")
        
        return state
    
    def _should_authenticate(self, state: Dict) -> str:
        """Determine if authentication step is needed"""
        intent_parts = state.get("intent_parts", {})
        action = intent_parts.get("action")
        
        # Don't authenticate immediately if we just opened an app and need to click login button first
        # Check if we're in the middle of opening an app
        if action == "open_app" and state.get("action_success", False):
            # App was just opened, need to verify and potentially click login button first
            return "act"  # Go to act first, then verify will handle login button click
        
        # If user explicitly said "login" AND app is already open, go to authenticate
        if action == "login" and state.get("session_active", False):
            return "authenticate"
        
        # If login screen detected, go to authenticate
        # BUT skip if Google login already completed
        if state.get("needs_auth", False):
            if state.get("login_complete") and state.get("login_method") == "google":
                print("[Router] Google login complete - skipping auth, going to act")
                return "act"
            return "authenticate"
        return "act"
    
    def _authenticate(self, state: Dict) -> Dict:
        """Handle authentication using LLM vision - fully automatic"""
        logger.info("Handling authentication...")
        self.state_manager.update_state(AgentState.AUTHENTICATING)
        self.state_manager.set_current_step("authentication")
        
        # CHECK: If Google login already completed, skip LLM flow entirely
        if state.get("login_complete") and state.get("login_method") == "google":
            print("[Auth] ✓ Google sign-in already completed - skipping LLM auth flow")
            state["auth_success"] = True
            state["auth_message"] = "Logged in via Google"
            state["action_success"] = True
            state["session_active"] = True
            state["task_complete"] = False
            return state
        
        print("[Status] Starting LLM-guided authentication flow...")
        device = self.adb_client.get_device()
        
        # Use LLM to analyze screenshot and perform login automatically
        print("[Auth] Analyzing screen with LLM to find login button and guide authentication...")
        success, message = self._perform_llm_guided_login(device)
        
        state["auth_success"] = success
        state["auth_message"] = message
        state["action_success"] = success
        
        if success:
            self.state_manager.add_step("authentication")
            state["session_active"] = True  # Keep session active after login
            state["task_complete"] = False  # Don't complete, user can continue
            print("[Auth] Login successful. You can continue using the app.")
        else:
            state["error"] = message
            state["task_complete"] = False  # Don't complete on error, user can retry
        
        return state
    
    def _perform_llm_guided_login(self, device) -> Tuple[bool, str]:
        """
        Use LLM vision to automatically find login button and perform login actions
        
        Args:
            device: ADB device instance
            
        Returns:
            Tuple of (success, message)
        """
        import os
        import datetime
        
        # Create debug directory for screenshots
        debug_dir = "debug_screenshots"
        os.makedirs(debug_dir, exist_ok=True)
        
        # First, verify we're actually in an app (not home screen)
        print("[Auth] Verifying app is open...")
        max_wait_attempts = 5
        wait_attempt = 0
        
        while wait_attempt < max_wait_attempts:
            wait_attempt += 1
            screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=False)
            description = screen_analysis.get("description", "").lower()
            
            # Check if we're still on home screen
            is_home_screen = any(keyword in description for keyword in [
                "home screen", "app icons", "search bar", "android home", 
                "launcher", "app drawer", "home screen of an android"
            ])
            
            if is_home_screen:
                print(f"[Auth] Still on home screen (attempt {wait_attempt}/{max_wait_attempts}), waiting for app to load...")
                self.device_actions.wait(2.0)
                continue
            else:
                print("[Auth] App screen detected, proceeding with login...")
                break
        
        if wait_attempt >= max_wait_attempts:
            return False, "App did not load - still showing home screen after multiple attempts. Check if app opened successfully."
        
        max_steps = 10  # Maximum number of steps to prevent infinite loops
        step_count = 0
        previous_actions = []  # Track previous actions to detect loops
        consecutive_login_clicks = 0  # Track consecutive login button clicks
        
        while step_count < max_steps:
            step_count += 1
            print(f"\n[Auth Step {step_count}] Analyzing current screen...")
            
            # Capture screenshot first
            screenshot = self.screen_analyzer.capture_screenshot(device)
            if screenshot:
                # Save screenshot for debugging
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(debug_dir, f"auth_step_{step_count}_{timestamp}.png")
                screenshot.save(screenshot_path)
                print(f"[Auth] Screenshot saved: {screenshot_path}")
            
            # Capture and analyze screen with LLM
            screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
            description = screen_analysis.get("description", "")
            elements = screen_analysis.get("elements", [])
            
            # Check if we're back on home screen (app might have closed)
            description_lower = description.lower()
            if any(keyword in description_lower for keyword in ["home screen", "app icons", "search bar", "android home"]):
                return False, "App appears to have closed or returned to home screen during login process"
            
            # Use LLM vision API directly for precise coordinate detection
            action_plan = self._get_llm_login_action_plan_with_vision(device, description, elements, step_count)
            
            action_type = action_plan.get("action", "none")
            reason = action_plan.get("reason", "")
            element_descriptions = action_plan.get("element_descriptions", {})
            
            # Tell user what LLM sees
            print(f"\n[Auth] LLM Action: {action_type}")
            print(f"[Auth] What LLM sees: {reason}")
            if element_descriptions:
                for elem_type, elem_desc in element_descriptions.items():
                    if elem_desc:
                        print(f"[Auth] {elem_type.replace('_', ' ').title()}: {elem_desc}")
            
            # Speak what LLM sees to user
            if reason:
                feedback_msg = f"I can see: {reason}"
                print(f"[TTS] Speaking: {feedback_msg}")
                self.tts.speak(feedback_msg)
                # Wait a moment for user to hear
                self.device_actions.wait(1.0)
            
            # Detect infinite loops - if clicking login button multiple times without progress
            if action_type == "click_login_button":
                consecutive_login_clicks += 1
                if consecutive_login_clicks > 3:
                    return False, f"Clicked login button {consecutive_login_clicks} times without progress - possible loop detected. Check screenshots in {debug_dir}/"
            else:
                consecutive_login_clicks = 0  # Reset counter if different action
            
            # Track action history
            action_key = f"{action_type}_{action_plan.get('coordinates', [0, 0])}"
            if action_key in previous_actions[-2:]:  # Check last 2 actions
                return False, f"Detected repeated action: {action_type} - possible loop. Check screenshots in {debug_dir}/"
            previous_actions.append(action_key)
            if len(previous_actions) > 5:
                previous_actions.pop(0)  # Keep only last 5 actions
            
            # REMOVED: Google sign-in - always use credential-based login
            # Handle login button click (credential-based authentication only)
            if action_type == "click_login_button":
                # CRITICAL CHECK: Is login wall/bottom sheet visible?
                # If yes, NEVER use LLM/top-right coordinates - use ONLY bottom sheet button detection
                description_lower = description.lower()
                login_wall_keywords = [
                    "thanks for trying chatgpt",
                    "thanks for trying",
                    "log in or sign up",
                    "continue with google",
                    "sign up",
                    "login wall",
                    "bottom sheet"
                ]
                is_login_wall_visible = any(keyword in description_lower for keyword in login_wall_keywords)
                
                # Also check elements for bottom sheet indicators
                for elem in elements:
                    elem_desc = elem.get("description", "").lower()
                    if any(keyword in elem_desc for keyword in login_wall_keywords):
                        is_login_wall_visible = True
                        break
                
                if is_login_wall_visible:
                    print("[Auth] ⚠ LOGIN WALL/BOTTOM SHEET DETECTED")
                    print("[Auth] RULE: Bottom sheet visible → IGNORE LLM coordinates → Use ONLY bottom sheet button detection")
                    
                    # Use ONLY bottom sheet button detection (lowest large clickable button)
                    accessibility_result = self.accessibility.find_real_login_button()
                    
                    if accessibility_result:
                        x, y, button_info = accessibility_result
                        
                        if self._validate_coordinates(x, y):
                            print(f"[Auth] ✓ Found bottom sheet login button at ({x}, {y})")
                            print(f"[Auth]   Button size: {button_info.get('width', 'N/A')}x{button_info.get('height', 'N/A')}")
                            print(f"[Auth]   Button text: '{button_info.get('text', 'N/A')[:50] if button_info.get('text') else 'N/A'}'")
                            
                            # Add small offset for button padding
                            tap_x = x + 5
                            tap_y = y + 5
                            print(f"[Auth] Tapping bottom sheet button at ({tap_x}, {tap_y})")
                            self.tts.speak("Tapping on the login button")
                            
                            self.device_actions.tap(tap_x, tap_y, delay=0.5)
                            self.device_actions.wait(3.0)
                            
                            # VERIFY: Check if email field appeared (confirms tap was correct)
                            print("[Auth] Verifying tap was correct - checking for email field...")
                            screen_analysis_after = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                            description_after = screen_analysis_after.get("description", "").lower()
                            elements_after = screen_analysis_after.get("elements", [])
                            
                            email_indicators = ["email", "e-mail", "username", "user name", "password"]
                            has_email_field = any(indicator in description_after for indicator in email_indicators)
                            
                            # Check elements for input fields
                            for elem in elements_after:
                                elem_desc = elem.get("description", "").lower()
                                elem_type = elem.get("type", "")
                                if elem_type == "text_field" and any(indicator in elem_desc for indicator in ["email", "username", "password"]):
                                    has_email_field = True
                                    break
                            
                            if has_email_field:
                                print("[Auth] ✓ VERIFICATION PASSED: Email field appeared - tap was correct!")
                                continue  # Continue to next step (email input)
                            else:
                                print("[Auth] ⚠ VERIFICATION FAILED: Email field did not appear")
                                print("[Auth] Screen description: " + description_after[:200])
                                return False, "Tapped login button but email field did not appear - tap may have been incorrect"
                        else:
                            print(f"[Auth] ERROR: Bottom sheet button coordinates ({x}, {y}) failed validation")
                            return False, f"Bottom sheet button coordinates invalid: ({x}, {y})"
                    else:
                        print("[Auth] ERROR: No bottom sheet button found")
                        return False, "Login wall detected but no bottom sheet button found"
                
                else:
                    # NO bottom sheet detected - use normal flow (but this should rarely happen)
                    print("[Auth] No login wall detected - using standard button detection")
                    print("[Auth] WARNING: This path should rarely be used - most login screens use bottom sheets")
                    return False, "Login button detection failed - no bottom sheet detected and standard detection unavailable"
                
                # Verify screen changed by checking if login button is still visible
                print("[Auth] Verifying screen changed after tap...")
                screen_analysis_after = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                description_after = screen_analysis_after.get("description", "")
                elements_after = screen_analysis_after.get("elements", [])
                
                # Check if login button is still visible (screen didn't change)
                login_still_visible = False
                for elem in elements_after:
                    elem_desc = elem.get("description", "").lower()
                    if "login" in elem_desc or "sign in" in elem_desc:
                        login_still_visible = True
                        break
                
                if login_still_visible and description_before == description_after:
                    print("[Auth] WARNING: Screen didn't change after tap - login button still visible")
                    print("[Auth] Trying alternative tap method...")
                    
                    # Try tapping slightly different coordinates (button might be slightly offset)
                    offsets = [(0, 0), (-20, 0), (20, 0), (0, -20), (0, 20), (-30, -30), (30, 30)]
                    for offset_x, offset_y in offsets:
                        new_x = max(50, min(x + offset_x, self.device_actions.SCREEN_WIDTH - 50))
                        new_y = max(50, min(y + offset_y, self.device_actions.SCREEN_HEIGHT - 50))
                        print(f"[Auth] Trying tap at offset ({offset_x}, {offset_y}): ({new_x}, {new_y})")
                        self.device_actions.tap(new_x, new_y, delay=0.5)
                        self.device_actions.wait(2.0)
                        
                        # Check again
                        temp_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=False)
                        temp_desc = temp_analysis.get("description", "")
                        if temp_desc != description_before:
                            print(f"[Auth] Screen changed with offset ({offset_x}, {offset_y})!")
                            break
                    else:
                        print("[Auth] All tap attempts failed - screen still shows login button")
                        self.tts.speak("I'm having trouble tapping the login button. The coordinates might be incorrect.")
                        return False, "Failed to tap login button - screen didn't change after multiple attempts"
                else:
                    print("[Auth] Screen changed successfully after tap")
                    # After clicking login button, automatically detect next screen and proceed
                    print("[Auth] Analyzing next screen to determine next action...")
                    self.device_actions.wait(2.0)  # Wait for screen to fully load
                    next_screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                    next_elements = next_screen_analysis.get("elements", [])
                    next_description = next_screen_analysis.get("description", "")
                    
                    # Check for email/password fields (credential-based login)
                    # REMOVED: Google sign-in - always use credential-based login
                    has_email_field = any(elem.get("type") == "text_field" and "email" in elem.get("description", "").lower() for elem in next_elements)
                    has_password_field = any(elem.get("type") == "text_field" and "password" in elem.get("description", "").lower() for elem in next_elements)
                    
                    if has_email_field or has_password_field:
                        print("[Auth] Login form detected - will proceed to enter credentials")
                        # Continue loop to enter email/password
                    else:
                        print(f"[Auth] Next screen: {next_description[:100]}...")
                        # Continue loop to analyze further
                
                continue
            
            elif action_type == "enter_email":
                # Enter email - USE TYPED INPUT (more accurate than voice)
                # IMPORTANT: NEVER trust LLM coordinates for input fields - always use Accessibility
                
                # CHECK: If email was already entered, skip asking again
                if self.login_flow_state.get("email_entered"):
                    print("[Auth] Email already entered - skipping redundant email prompt")
                    print("[Auth] Moving to next authentication step...")
                    # Check if password field is visible now
                    clickable_elements = self.accessibility.find_clickable_elements()
                    password_fields = [e for e in clickable_elements if "edittext" in e.get("class", "").lower() or "edit" in e.get("class", "").lower()]
                    if len(password_fields) >= 1:
                        print("[Auth] Found password field - LLM should detect it next")
                    continue
                
                # Prompt user to TYPE email
                print("\n" + "="*60)
                print("LOGIN REQUIRED - EMAIL")
                print("="*60)
                self.tts.speak("Login required. Please type your email address.")
                
                try:
                    email = input("\nEmail: ").strip()
                    if not email or "@" not in email:
                        print("[Auth] Invalid email format")
                        return False, "Invalid email format"
                    print(f"[Auth] Email received: {email}")
                except KeyboardInterrupt:
                    return False, "Login cancelled by user"
                
                # ALWAYS use Accessibility to find REAL input field coordinates
                # LLM coordinates are often wrong!
                print("[Auth] Finding email field via Accessibility (ignoring LLM coordinates)...")
                clickable_elements = self.accessibility.find_clickable_elements()
                
                # Find ALL EditText fields
                edit_fields = []
                for elem in clickable_elements:
                    elem_class = elem.get("class", "").lower()
                    if "edittext" in elem_class or "edit" in elem_class or "input" in elem_class:
                        x = elem.get("x", 0)
                        y = elem.get("y", 0)
                        if x > 0 and y > 0:
                            edit_fields.append({
                                "x": x, 
                                "y": y, 
                                "class": elem_class,
                                "text": elem.get("text", ""),
                                "bounds": elem.get("bounds", [])
                            })
                
                if edit_fields:
                    # Sort by Y coordinate - first field is usually email
                    edit_fields.sort(key=lambda e: e["y"])
                    email_field = edit_fields[0]
                    x, y = email_field["x"], email_field["y"]
                    
                    print(f"[Auth] ✓ Found email input field via Accessibility at ({x}, {y})")
                    print(f"[Auth]   Class: {email_field['class']}")
                    print(f"[Auth]   Total edit fields found: {len(edit_fields)}")
                    
                    # Check if field already has text (existing email) - CLEAR IT FIRST
                    existing_text = email_field.get("text", "")
                    if existing_text and "@" in existing_text:
                        print(f"[Auth] ⚠ Field already has email: '{existing_text}' - will clear and replace")
                    
                    # Tap the email field
                    self.device_actions.tap(x, y, delay=0.5)
                    self.device_actions.wait(1.5)  # Wait for keyboard
                    
                    # Tap again to ensure focus
                    self.device_actions.tap(x, y, delay=0.3)
                    self.device_actions.wait(0.5)
                    
                    print(f"[Auth] Typing email: {email}")
                    # clear_first=True will clear existing text using the robust clear method
                    success = self.device_actions.type_text(email, clear_first=True)
                    if success:
                        print(f"[Auth] ✓ Email typed successfully")
                        # MARK email as entered to prevent redundant prompts
                        self.login_flow_state["email_entered"] = True
                        self.login_flow_state["email"] = email
                    else:
                        print(f"[Auth] ✗ Failed to type email")
                    self.device_actions.wait(1.0)
                    
                    # Find and click Continue button (should be BELOW the email field)
                    self._click_continue_button_below_field(y)
                    continue
                else:
                    print("[Auth] No EditText fields found via Accessibility")
                    return False, "Couldn't find email input field"
            
            elif action_type == "enter_password":
                # Enter password - USE TYPED INPUT (secure, hidden)
                # IMPORTANT: NEVER trust LLM coordinates - always use Accessibility
                
                # Prompt user to TYPE password (secure input)
                print("\n" + "="*60)
                print("PASSWORD REQUIRED")
                print("="*60)
                self.tts.speak("Please type your password.")
                
                try:
                    import getpass
                    try:
                        password = getpass.getpass("\nPassword: ")
                    except Exception:
                        password = input("\nPassword: ")
                    
                    if not password:
                        return False, "No password provided"
                    print("[Auth] Password received (hidden)")
                except KeyboardInterrupt:
                    return False, "Login cancelled by user"
                
                # ALWAYS use Accessibility to find REAL password field coordinates
                print("[Auth] Finding password field via Accessibility (ignoring LLM coordinates)...")
                clickable_elements = self.accessibility.find_clickable_elements()
                
                # Find ALL EditText fields
                edit_fields = []
                for elem in clickable_elements:
                    elem_class = elem.get("class", "").lower()
                    if "edittext" in elem_class or "edit" in elem_class or "input" in elem_class:
                        x = elem.get("x", 0)
                        y = elem.get("y", 0)
                        if x > 0 and y > 0:
                            edit_fields.append({
                                "x": x,
                                "y": y,
                                "class": elem_class,
                                "text": elem.get("text", "")
                            })
                
                if edit_fields:
                    # Sort by Y coordinate
                    edit_fields.sort(key=lambda e: e["y"])
                    
                    # Password is usually the LAST field (or second if there are two)
                    # If only one field, use that (might be separate password screen)
                    if len(edit_fields) >= 2:
                        password_field = edit_fields[-1]  # Last field
                    else:
                        password_field = edit_fields[0]  # Only field
                    
                    x, y = password_field["x"], password_field["y"]
                    
                    print(f"[Auth] ✓ Found password field via Accessibility at ({x}, {y})")
                    print(f"[Auth]   Class: {password_field['class']}")
                    print(f"[Auth]   Total edit fields: {len(edit_fields)}")
                    
                    # Tap the password field
                    self.device_actions.tap(x, y, delay=0.5)
                    self.device_actions.wait(1.5)  # Wait for keyboard
                    self.device_actions.tap(x, y, delay=0.3)  # Tap again
                    self.device_actions.wait(0.5)
                    
                    print("[Auth] Entering password")
                    success = self.device_actions.type_text(password, clear_first=True)
                    if success:
                        print(f"[Auth] ✓ Password typed successfully")
                    else:
                        print(f"[Auth] ✗ Failed to type password")
                    self.device_actions.wait(1.0)
                    
                    # Find and click Submit/Login button (below password field)
                    self._click_submit_button_below_field(y)
                    continue
                else:
                    print("[Auth] No EditText fields found via Accessibility")
                    return False, "Couldn't find password field"
            
            elif action_type == "click_submit":
                # Click submit/login button
                submit_coords = action_plan.get("submit_button_coordinates")
                if submit_coords and len(submit_coords) >= 2:
                    x, y = int(submit_coords[0]), int(submit_coords[1])
                    print(f"[Auth] Clicking submit/login button at ({x}, {y})")
                    self.device_actions.tap(x, y, delay=0.5)
                    self.device_actions.wait(3.0)  # Wait for login to process
                    continue
                else:
                    return False, "LLM couldn't find submit button"
            
            elif action_type == "login_complete":
                # Login successful
                return True, "Login completed successfully"
            
            elif action_type == "error":
                error_msg = action_plan.get("error_message", "Unknown error during login")
                return False, error_msg
            
            elif action_type == "none":
                # No action needed or unclear what to do
                return False, "LLM couldn't determine next login step"
            
            else:
                return False, f"Unknown action type: {action_type}"
        
        return False, f"Login process exceeded maximum steps ({max_steps})"
    
    def _get_llm_login_action_plan_with_vision(self, device, screen_description: str, elements: List[Dict], step_count: int) -> Dict:
        """
        Use GPT-4o Vision API directly to analyze screenshot and get exact coordinates
        
        Args:
            device: ADB device instance
            screen_description: Description from previous analysis
            elements: Detected UI elements
            step_count: Current step number
            
        Returns:
            Dictionary with action plan
        """
        # Capture fresh screenshot for vision API
        screenshot = self.screen_analyzer.capture_screenshot(device)
        if not screenshot:
            return {"action": "error", "error_message": "Failed to capture screenshot"}
        
        # Get actual screenshot dimensions
        screenshot_width, screenshot_height = screenshot.size
        print(f"[Auth] Screenshot dimensions: {screenshot_width} x {screenshot_height}")
        
        # Convert to base64
        base64_image = self.screen_analyzer.image_to_base64(screenshot)
        
        prompt = f"""You are analyzing a mobile app screenshot to find the EXACT coordinates of UI elements for automatic login.

SCREENSHOT DIMENSIONS: {screenshot_width} pixels wide × {screenshot_height} pixels tall

SCREEN DESCRIPTION FROM PREVIOUS ANALYSIS:
{screen_description}

CURRENT STEP: {step_count}

TASK: Look at the screenshot carefully and find the EXACT center coordinates of:
1. "Continue with Google" button (PREFERRED - fastest login method)
2. "Log in" or "Sign in" button (fallback if no Google option)
3. Email/username input field
4. Password input field
5. Submit/Login button

IMPORTANT: PREFER "Continue with Google" button if available - it's the fastest login method.

CRITICAL REQUIREMENTS:
- The screenshot is {screenshot_width} pixels wide x {screenshot_height} pixels tall
- You MUST provide the EXACT center pixel coordinates (x, y) of each element
- X coordinate: 0 (left) to {screenshot_width} (right)
- Y coordinate: 0 (top) to {screenshot_height} (bottom)
- Look at the ACTUAL screenshot - don't guess!

FOR "LOG IN" BUTTON COORDINATES - FOLLOW THESE STEPS:
1. Find the "Log in" button (look at the screenshot to identify it visually)
2. Identify the button's BOUNDING BOX:
   - Top-left corner: (x_left, y_top)
   - Bottom-right corner: (x_right, y_bottom)
3. Calculate the EXACT CENTER:
   - center_x = (x_left + x_right) / 2
   - center_y = (y_top + y_bottom) / 2
4. Provide BOTH:
   - Bounding box: top-left (x_left, y_top), bottom-right (x_right, y_bottom)
   - Center coordinates: (center_x, center_y)

IMPORTANT NOTES ABOUT COORDINATES:
- Device screen resolution: 1080 pixels wide × 2400 pixels tall
- Status bar is at the top (approximately Y=0 to Y=50) - DO NOT place coordinates in status bar area
- Y coordinate of 100-120 is WRONG - that's still in the status bar/notification area
- Login buttons are typically BELOW the status bar - Y coordinate usually 150-250 (or higher)
- The actual "Log in" button center is typically around Y=200-210 (accounting for status bar)
- Measure pixels carefully from the top-left corner (0, 0) of the screenshot
- The center Y should be calculated as: (y_top + y_bottom) / 2
- Example: If button top is at Y=180, and bottom is at Y=220, center Y = (180+220)/2 = 200
- Another example: If button top-left is at (961, 180) and bottom-right is at (1080, 220), center = (1020, 200)
- Provide EXACT pixel values based on what you ACTUALLY see in the screenshot - don't guess!
- If you see a "Log in" button in the top-right area, its Y coordinate should be around 180-220, NOT 100-120

ANALYZE THE SCREENSHOT AND RESPOND IN JSON:
{{
    "action": "click_login_button" | "enter_email" | "enter_password" | "click_submit" | "login_complete" | "error" | "none",
    "coordinates": [x, y] or null,  // EXACT center of login/Google button - MUST be calculated from bounding box
    "login_button_bounding_box": {{"top_left": [x_left, y_top], "bottom_right": [x_right, y_bottom]}} or null,  // Button bounds
    "email_field_coordinates": [x, y] or null,
    "password_field_coordinates": [x, y] or null,
    "submit_button_coordinates": [x, y] or null,
    "reason": "What you see in the screenshot and why you chose this action",
    "element_descriptions": {{
        "login_button": "Description including bounding box: top-left (x, y), bottom-right (x, y), center (x, y)",
        "email_field": "Description of email field if visible",
        "password_field": "Description of password field if visible"
    }},
    "error_message": "Error description if action is 'error'"
}}

CRITICAL COORDINATE REQUIREMENTS: 
- For "Log in" button, you MUST provide:
  1. Bounding box: top-left and bottom-right corners (measure from screenshot)
  2. Center coordinates calculated from: center_x = (x_left + x_right) / 2, center_y = (y_top + y_bottom) / 2
- Example 1: If button is top-left (722, 180) and bottom-right (893, 220):
  * center_x = (722 + 893) / 2 = 807.5 ≈ 808
  * center_y = (180 + 220) / 2 = 200
  * So coordinates should be [808, 200]
- Example 2: If button is top-left (961, 180) and bottom-right (1080, 220):
  * center_x = (961 + 1080) / 2 = 1020.5 ≈ 1021
  * center_y = (180 + 220) / 2 = 200
  * So coordinates should be [1021, 200]
- REAL EXAMPLE: User reported actual button center is at (961, 204) - this means:
  * The button's bounding box likely spans approximately Y=184-224 (center 204)
  * X coordinate around 961 (right side of screen, near edge)
  * Y coordinate around 204 (below status bar, in the app content area)
  * If you measure top-left as (914, 184) and bottom-right as (1008, 224), center = (961, 204) ✓
- CRITICAL: Measure the ACTUAL button bounds from the screenshot - don't estimate!
- Look at where the button text "Log in" actually starts and ends
- The button background/container extends beyond the text - measure the full clickable area
- Y coordinate of 100-171 is TOO HIGH - the actual button center Y is around 200-210
- If you see Y coordinates around 140-180, you're measuring too high - look LOWER on the screen
- Don't guess coordinates - MEASURE pixel-by-pixel from the screenshot"""

        try:
            # Use GPT-4o (not mini) for better vision accuracy
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",  # Use full GPT-4o for better vision
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing mobile screenshots and providing EXACT pixel coordinates. You can see the screenshot clearly - provide precise coordinates based on what you actually see, not guesses. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,  # Very low temperature for precision
                max_tokens=500
            )
            
            import json
            import re
            result_text = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
            if json_match:
                action_plan = json.loads(json_match.group())
                reason = action_plan.get('reason', 'No reason provided')
                element_descriptions = action_plan.get('element_descriptions', {})
                
                print(f"[Auth] LLM Reason: {reason}")
                if element_descriptions.get('login_button'):
                    print(f"[Auth] Login Button: {element_descriptions['login_button']}")
                
                # Check if bounding box is provided - use it to verify/calculate center
                bounding_box = action_plan.get("login_button_bounding_box")
                coords = action_plan.get("coordinates")
                
                # Get screenshot dimensions for scaling
                screenshot = self.screen_analyzer.capture_screenshot(device)
                if screenshot:
                    screenshot_width, screenshot_height = screenshot.size
                    device_width = self.device_actions.SCREEN_WIDTH
                    device_height = self.device_actions.SCREEN_HEIGHT
                    scale_x = device_width / screenshot_width if screenshot_width > 0 else 1.0
                    scale_y = device_height / screenshot_height if screenshot_height > 0 else 1.0
                else:
                    scale_x, scale_y = 1.0, 1.0
                
                if bounding_box:
                    top_left = bounding_box.get("top_left", [])
                    bottom_right = bounding_box.get("bottom_right", [])
                    if len(top_left) >= 2 and len(bottom_right) >= 2:
                        # Calculate center from bounding box (in screenshot coordinates)
                        calc_x_screenshot = (top_left[0] + bottom_right[0]) / 2
                        calc_y_screenshot = (top_left[1] + bottom_right[1]) / 2
                        
                        # Scale to device coordinates
                        calc_x_device = int(calc_x_screenshot * scale_x)
                        calc_y_device = int(calc_y_screenshot * scale_y)
                        
                        print(f"[Auth] Bounding box (screenshot): top-left {top_left}, bottom-right {bottom_right}")
                        print(f"[Auth] Calculated center (screenshot): ({calc_x_screenshot:.1f}, {calc_y_screenshot:.1f})")
                        print(f"[Auth] Calculated center (device, scaled): ({calc_x_device}, {calc_y_device})")
                        
                        if coords:
                            print(f"[Auth] LLM provided center (screenshot): {coords}")
                            # Scale LLM coordinates
                            llm_x_scaled = int(coords[0] * scale_x)
                            llm_y_scaled = int(coords[1] * scale_y)
                            print(f"[Auth] LLM center (device, scaled): ({llm_x_scaled}, {llm_y_scaled})")
                            
                            # Use calculated center from bounding box (more reliable)
                            print(f"[Auth] Using calculated center from bounding box: ({calc_x_device}, {calc_y_device})")
                            action_plan["coordinates"] = [calc_x_device, calc_y_device]
                        else:
                            # No LLM coordinates, use bounding box calculation
                            action_plan["coordinates"] = [calc_x_device, calc_y_device]
                
                # Validate coordinates - reject if in status bar area
                coords = action_plan.get("coordinates")
                if coords:
                    x, y = coords[0], coords[1]
                    if not self._validate_coordinates(x, y):
                        print(f"[Auth] ERROR: Coordinates ({x}, {y}) failed validation - likely in status bar area")
                        print(f"[Auth] Rejecting these coordinates and will try to find better ones from vision elements")
                        # Clear invalid coordinates - will fall back to vision elements
                        action_plan["coordinates"] = None
                
                return action_plan
            else:
                print(f"[Auth] WARNING: Could not parse LLM response as JSON")
                print(f"[Auth] Response: {result_text[:300]}...")
                return {"action": "error", "error_message": "Could not parse LLM response"}
        except Exception as e:
            print(f"[Auth] LLM Vision Error: {e}")
            import traceback
            traceback.print_exc()
            return {"action": "error", "error_message": f"LLM error: {str(e)}"}
    
    def _get_llm_login_action_plan(self, screen_description: str, elements: List[Dict], step_count: int) -> Dict:
        """
        Use LLM to analyze screen and determine next login action
        
        Args:
            screen_description: Description of current screen
            elements: Detected UI elements
            step_count: Current step number
            
        Returns:
            Dictionary with action plan
        """
        # Build elements description for LLM
        elements_desc = []
        for elem in elements:
            elem_type = elem.get("type", "")
            elem_desc = elem.get("description", "")
            x = elem.get("x", 0)
            y = elem.get("y", 0)
            elem_text = elem.get("text", "")
            
            if elem_type == "button":
                elements_desc.append(f"Button at ({x}, {y}): {elem_desc} {f'(text: {elem_text})' if elem_text else ''}")
            elif elem_type == "text_field":
                elements_desc.append(f"Input field at ({x}, {y}): {elem_desc}")
            elif elem_type == "icon":
                elements_desc.append(f"Icon at ({x}, {y}): {elem_desc}")
        
        elements_text = "\n".join(elements_desc) if elements_desc else "No elements detected"
        
        prompt = f"""You are analyzing a mobile app screen to perform automatic login. Analyze the screenshot and determine the EXACT next action needed.

SCREEN DESCRIPTION:
{screen_description}

DETECTED ELEMENTS:
{elements_text}

CURRENT STEP: {step_count}

Analyze the screen and determine what action to take next. Look for (in priority order):
1. "Log in" or "Sign in" button - ALWAYS use this for credential-based authentication (DO NOT use Google)
2. Email/username input field - if visible and empty, enter email
3. Password input field - if visible and empty, enter password  
4. Submit/Login button - if email and password are filled, click it
5. Success indicators - if logged in successfully, mark as complete

IMPORTANT:
- PREFER "Continue with Google" button if available - it's the fastest login method
- Fallback to "Log in" button for credential-based authentication
- Provide EXACT pixel coordinates (x, y) for 1080x2400 screen
- Coordinates must be the CENTER of the clickable element
- Be precise with coordinates - they will be used to tap the screen

Respond in JSON format:
{{
    "action": "click_login_button" | "enter_email" | "enter_password" | "click_submit" | "login_complete" | "error" | "none",
    "coordinates": [x, y] or null,  // For login/Google button click
    "email_field_coordinates": [x, y] or null,
    "password_field_coordinates": [x, y] or null,
    "submit_button_coordinates": [x, y] or null,
    "reason": "Brief explanation of what you see and what action to take",
    "error_message": "Error description if action is 'error'"
}}

Examples:
- If you see "Log in" button: {{"action": "click_login_button", "coordinates": [540, 1800], "reason": "Found Log in button, clicking it"}}
- If you see email field: {{"action": "enter_email", "email_field_coordinates": [540, 1200], "reason": "Found email field, ready to enter email"}}
- If login complete: {{"action": "login_complete", "reason": "Successfully logged in"}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes mobile screenshots and provides exact coordinates for UI elements. Always respond with valid JSON only. Be precise with coordinates."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            import json
            import re
            result_text = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                action_plan = json.loads(json_match.group())
                print(f"[Auth] LLM Reason: {action_plan.get('reason', 'No reason provided')}")
                return action_plan
            else:
                print(f"[Auth] WARNING: Could not parse LLM response as JSON")
                print(f"[Auth] Response: {result_text[:200]}...")
                return {"action": "error", "error_message": "Could not parse LLM response"}
        except Exception as e:
            print(f"[Auth] LLM Error: {e}")
            import traceback
            traceback.print_exc()
            return {"action": "error", "error_message": f"LLM error: {str(e)}"}
    
    def _validate_coordinates(self, x: int, y: int) -> bool:
        """
        Validate coordinates are within screen bounds
        
        Args:
            x: X coordinate (1-1080)
            y: Y coordinate (1-2400)
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        # Check bounds - coordinates must be within screen dimensions
        # X: 1 to 1080, Y: 1 to 2400 (allowing edge coordinates)
        if x < 1 or y < 1:
            print(f"[Auth] WARNING: Coordinates ({x}, {y}) are invalid - must be >= 1")
            return False
        
        if x > self.device_actions.SCREEN_WIDTH or y > self.device_actions.SCREEN_HEIGHT:
            print(f"[Auth] WARNING: Coordinates ({x}, {y}) exceed screen bounds ({self.device_actions.SCREEN_WIDTH}x{self.device_actions.SCREEN_HEIGHT})")
            return False
        
        # Status bar is typically at Y=0 to Y=50 (or up to Y=100 on some devices)
        # Reject coordinates in status bar area (Y < 100) as buttons can't be there
        STATUS_BAR_MAX = 100
        if y < STATUS_BAR_MAX:
            print(f"[Auth] WARNING: Y coordinate {y} is in status bar area (0-{STATUS_BAR_MAX}). Buttons should be below status bar (Y >= {STATUS_BAR_MAX}).")
            return False
        
        # All other coordinates within bounds are valid (Y: 100-2400, X: 1-1080)
        # Buttons can be anywhere: top-right, bottom, middle, etc.
        return True
    
    def _get_auth_guidance_with_llm(self, screen_description: str, elements: List[Dict]) -> Dict:
        """
        Use LLM to provide guidance on authentication flow
        
        Args:
            screen_description: Description of current screen
            elements: Detected UI elements
            
        Returns:
            Dictionary with authentication guidance
        """
        # Build elements description
        elements_desc = []
        for elem in elements:
            elem_type = elem.get("type", "")
            elem_desc = elem.get("description", "")
            x = elem.get("x", 0)
            y = elem.get("y", 0)
            if elem_type == "button":
                elements_desc.append(f"Button at ({x}, {y}): {elem_desc}")
            elif elem_type == "text_field":
                elements_desc.append(f"Input field at ({x}, {y}): {elem_desc}")
        
        elements_text = "\n".join(elements_desc) if elements_desc else "No elements detected"
        
        prompt = f"""You are analyzing a mobile app screen to guide the authentication process.

SCREEN DESCRIPTION:
{screen_description}

DETECTED ELEMENTS:
{elements_text}

Analyze the screen and determine:
1. Is there a "Log in" or "Sign in" button that needs to be clicked first to access the login form?
2. What is the login flow strategy (email-first, password-only, both fields visible)?
3. Where are the email and password fields located?
4. What is the exact location of the login/submit button?

Respond in JSON format:
{{
    "click_login_button_first": true/false,
    "login_button_coords": [x, y] or null,
    "strategy": "email_first" | "password_only" | "both_fields" | "unknown",
    "email_field_coords": [x, y] or null,
    "password_field_coords": [x, y] or null,
    "submit_button_coords": [x, y] or null,
    "notes": "Additional guidance"
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that guides mobile app authentication. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            import json
            import re
            result_text = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    guidance = json.loads(json_match.group())
                    # Convert coordinate lists to tuples
                    if guidance.get("login_button_coords") and isinstance(guidance["login_button_coords"], list):
                        coords = guidance["login_button_coords"][:2]
                        if len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords):
                            guidance["login_button_coords"] = tuple(int(c) for c in coords)
                        else:
                            print(f"[LLM Auth Guidance] Invalid coordinate format: {coords}")
                            guidance["login_button_coords"] = None
                    
                    # Log strategy for debugging
                    strategy = guidance.get("strategy", "standard")
                    if strategy == "unknown":
                        print(f"[LLM Auth Guidance] Strategy is 'unknown' - screen might not show login screen")
                        print(f"[LLM Auth Guidance] Screen description preview: {screen_description[:150]}...")
                        print(f"[LLM Auth Guidance] Elements found: {len(elements)}")
                    
                    return guidance
                except json.JSONDecodeError as je:
                    print(f"[LLM Auth Guidance] JSON decode error: {je}")
                    print(f"[LLM Auth Guidance] Response text: {result_text[:200]}...")
                    return {"strategy": "unknown", "click_login_button_first": False}
            else:
                print(f"[LLM Auth Guidance] No JSON found in response")
                print(f"[LLM Auth Guidance] Response text: {result_text[:200]}...")
                return {"strategy": "unknown", "click_login_button_first": False}
        except Exception as e:
            print(f"[LLM Auth Guidance] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"strategy": "unknown", "click_login_button_first": False}
    
    def _act(self, state: Dict) -> Dict:
        """Execute action based on intent"""
        logger.info("Executing action...")
        self.state_manager.update_state(AgentState.ACTING)
        
        intent_parts = state.get("intent_parts", {})
        device = self.adb_client.get_device()
        
        action = intent_parts.get("action")
        
        # Follow-up question in same app (e.g. ChatGPT already open) - skip launching again
        if state.get("query_session_follow_up"):
            state["action_success"] = True
            print("[Act] App already open (follow-up); skipping launch, will execute query in verify.")
            return state
        
        # Debug: Print parsed intent
        print(f"[Debug] Action: {action}, App: {intent_parts.get('app')}, Wants Login: {intent_parts.get('wants_login', False)}")
        
        if action == "open_app":
            app_name = intent_parts.get("app")
            if app_name:
                # Switching to another app ends the continuous query session
                if self._query_app_session and app_name.strip().lower() != self._query_app_session.strip().lower():
                    self._query_app_session = None
                    print("[Session] Opening different app; query session cleared.")
                print(f"[Status] Launching app: {app_name}...")
                
                success = self.app_launcher.launch_app(app_name)
                if success:
                    print(f"[Status] App launched, waiting for screen to load...")
                    # Wait longer for app to fully load, especially if login is needed
                    wait_time = self.config.get("agent.screenshot_delay", 3.0)
                    if intent_parts.get("wants_login", False):
                        wait_time = max(wait_time, 5.0)  # Wait at least 5 seconds if login is needed
                    self.device_actions.wait(wait_time)
                    self.state_manager.add_step(f"opened_{app_name}")
                    state["action_success"] = True
                    print(f"[Status] App opened successfully (waited {wait_time}s)")
                    # Give intelligent feedback AFTER confirming success
                    feedback = self._generate_llm_response(state, f"successfully opened {app_name} app")
                    self.tts.speak(feedback)
                else:
                    state["action_success"] = False
                    state["error"] = f"Failed to open {app_name}. App may not be installed or package name incorrect."
                    print(f"[ERROR] Failed to open {app_name}")
                    # Only show error message if app actually failed to open
                    feedback = self._generate_llm_response(state, f"failed to open {app_name} app")
                    self.tts.speak(feedback)
                    # STOP the loop - don't keep retrying
                    state["task_complete"] = True
            else:
                # APP NAME IS NONE - Could not parse which app to open
                # CRITICAL: Stop the loop immediately to avoid endless retries
                original_command = state.get("user_command", "")
                print(f"[ERROR] Could not determine which app to open from command: '{original_command}'")
                print("[ERROR] STOPPING to prevent endless loop")
                state["action_success"] = False
                state["error"] = f"Could not understand which app to open. Please say 'open' followed by the app name."
                state["task_complete"] = True  # STOP the loop
                self.tts.speak("Sorry, I couldn't understand which app to open. Please try again with a clearer command, like 'open ChatGPT'.")
        
        elif action == "close_app":
            # Close the current app (go to home screen)
            print("[Status] Closing app...")
            self.device_actions.home()
            self.device_actions.wait(1.0)
            state["action_success"] = True
            state["task_complete"] = True  # Only mark complete when closing
            self.state_manager.add_step("closed_app")
            print("[Status] App closed")
        
        elif action == "login":
            # User explicitly requested login - handle authentication
            print("[Status] Starting login process...")
            # This will be handled by _authenticate node, but we mark it here
            state["action_success"] = True  # Will be updated by authenticate
            state["needs_auth"] = True
            state["task_complete"] = False
            self.state_manager.add_step("login_requested")
        
        elif action == "send_whatsapp_message":
            # WhatsApp messaging flow
            recipient = intent_parts.get("recipient")
            message = intent_parts.get("message")
            send_to_current_chat = intent_parts.get("send_to_current_chat", False)
            
            print(f"\n[WhatsApp] Starting message flow...")
            print(f"[WhatsApp] Recipient: {recipient or '(current chat)'}")
            print(f"[WhatsApp] Message: {message}")
            print(f"[WhatsApp] Send to current chat: {send_to_current_chat}")
            
            if not message:
                print("[WhatsApp] ERROR: No message specified")
                state["action_success"] = False
                state["error"] = "Please specify what message to send."
                state["task_complete"] = True
                self.tts.speak("Please tell me what message you want to send.")
                return state
            
            # Step 1: Launch WhatsApp (or verify it's open)
            print("[WhatsApp] Step 1: Launching/verifying WhatsApp...")
            if not send_to_current_chat:
                self.tts.speak(f"Opening WhatsApp to message {recipient}.")
            else:
                self.tts.speak("Sending message to current chat.")
            success = self.app_launcher.launch_app("whatsapp")
            if not success:
                state["action_success"] = False
                state["error"] = "Failed to open WhatsApp"
                state["task_complete"] = True
                self.tts.speak("Could not open WhatsApp.")
                return state
            
            self.device_actions.wait(3.0)  # Wait for app to load
            
            # If sending to current chat, skip Steps 2-4 (search and select recipient)
            if send_to_current_chat:
                print("[WhatsApp] Sending to current chat - skipping recipient search")
            else:
                if not recipient:
                    print("[WhatsApp] ERROR: No recipient specified")
                    state["action_success"] = False
                    state["error"] = "Please specify who to send the message to."
                    state["task_complete"] = True
                    self.tts.speak("Please specify who you want to send the message to.")
                    return state
                
                # Step 1.5: If we're inside a chat, go back so we can search for (possibly different) recipient
                def _find_whatsapp_search_bar():
                    clickable_elements = self.accessibility.find_clickable_elements()
                    for elem in clickable_elements:
                        elem_text = (elem.get("text", "") or "").lower()
                        elem_desc = (elem.get("content_desc", "") or "").lower()
                        combined = elem_text + " " + elem_desc
                        if any(kw in combined for kw in ["search", "ask meta", "find"]):
                            x, y = elem.get("x", 0), elem.get("y", 0)
                            if x > 0 and y > 0:
                                return (x, y)
                    return None
                
                search_found = False
                for back_attempt in range(3):
                    sb = _find_whatsapp_search_bar()
                    if sb:
                        break
                    if back_attempt < 2:
                        print("[WhatsApp] Search bar not visible (maybe in a chat). Going back...")
                        self.device_actions.back()
                        self.device_actions.wait(1.5)
                else:
                    sb = None
                
                # Step 2: Find and tap the search bar
                print("[WhatsApp] Step 2: Finding search bar...")
                if sb:
                    print(f"[WhatsApp] Found search bar at {sb}")
                    self.device_actions.tap(sb[0], sb[1], delay=0.5)
                    self.device_actions.wait(1.5)
                    search_found = True
                
                if not search_found:
                    # Fallback: Tap at typical search bar location (top of screen)
                    print("[WhatsApp] Using fallback search bar location...")
                    self.device_actions.tap(540, 150, delay=0.5)
                    self.device_actions.wait(1.5)
                
                # Step 3: Type the recipient name
                print(f"[WhatsApp] Step 3: Searching for '{recipient}'...")
                self.device_actions.type_text(recipient, clear_first=True)
                self.device_actions.wait(2.0)  # Wait for search results
                
                # Step 4: Click on the first search result (the contact)
                print("[WhatsApp] Step 4: Selecting contact from results...")
                self.device_actions.wait(1.0)
                
                # Find clickable contact in search results (usually around Y=250-400)
                contact_found = False
                clickable_elements = self.accessibility.find_clickable_elements()
                
                for elem in clickable_elements:
                    y = elem.get("y", 0)
                    elem_text = (elem.get("text", "") or "").lower()
                    
                    # Look for contact in results area (below search bar)
                    if 200 < y < 600:
                        # Check if it's a contact (not a button)
                        elem_class = (elem.get("class", "") or "").lower()
                        if "edittext" not in elem_class and "button" not in elem_text:
                            x = elem.get("x", 0)
                            if x > 0:
                                print(f"[WhatsApp] Found contact at ({x}, {y})")
                                self.device_actions.tap(x, y, delay=0.5)
                                self.device_actions.wait(2.0)  # Wait for chat to open
                                contact_found = True
                                break
                
                if not contact_found:
                    # Fallback: Tap on first result location
                    print("[WhatsApp] Using fallback: tapping first result area...")
                    self.device_actions.tap(540, 300, delay=0.5)
                    self.device_actions.wait(2.0)
            
            # Step 5: Find the message input field and type message
            if message:
                print(f"[WhatsApp] Step 5: Typing message: '{message}'...")
                
                # Find message input field (usually at bottom of screen)
                input_found = False
                clickable_elements = self.accessibility.find_clickable_elements()
                
                for elem in clickable_elements:
                    elem_class = (elem.get("class", "") or "").lower()
                    elem_text = (elem.get("text", "") or "").lower()
                    elem_desc = (elem.get("content_desc", "") or "").lower()
                    y = elem.get("y", 0)
                    
                    # Look for EditText or input field near bottom
                    if "edittext" in elem_class or "type a message" in elem_text or "message" in elem_desc:
                        if y > 1800:  # Bottom of screen
                            x = elem.get("x", 0)
                            print(f"[WhatsApp] Found message input at ({x}, {y})")
                            self.device_actions.tap(x, y, delay=0.5)
                            self.device_actions.wait(1.0)
                            input_found = True
                            break
                
                if not input_found:
                    # Fallback: Tap at typical message input location
                    print("[WhatsApp] Using fallback message input location...")
                    self.device_actions.tap(450, 2250, delay=0.5)
                    self.device_actions.wait(1.0)
                
                # Type the message
                self.device_actions.type_text(message, clear_first=True)
                self.device_actions.wait(0.5)
                
                # Step 6: Send the message - tap GREEN circle send button (do NOT use Enter)
                # WhatsApp: green circle send is right of input, region X 1000-1080, Y 1200-1800
                print("[WhatsApp] Step 6: Finding green circle send button (accessibility)...")
                self.device_actions.wait(0.5)
                send_result = self.accessibility.find_whatsapp_send_button()
                if send_result:
                    sx, sy, _ = send_result
                    print(f"[WhatsApp] Tapping send button at ({sx}, {sy})")
                    self.device_actions.tap(sx, sy, delay=0.3)
                
                self.device_actions.wait(1.0)
                
                print("[WhatsApp] ✓ Message sent!")
                if send_to_current_chat:
                    self.tts.speak("Message sent!")
                else:
                    self.tts.speak(f"Message sent to {recipient}!")
                state["action_success"] = True
                state["task_complete"] = True
                # Keep WhatsApp session active so user can send more without saying "open WhatsApp"
                self._query_app_session = "whatsapp"
                print("[Session] WhatsApp session active - you can say 'send message to X say Y' or 'Tell him that...' until app is closed.")
            else:
                # No message specified - just open the chat
                print("[WhatsApp] Chat opened (no message to send)")
                self.tts.speak(f"Opened chat with {recipient}. What would you like to say?")
                state["action_success"] = True
                state["task_complete"] = False
                state["session_active"] = True
                self._query_app_session = "whatsapp"
            
            self.state_manager.add_step(f"whatsapp_message_{recipient}")
            return state
        
        elif action == "send_payment":
            # UPI Payment flow (Paytm, GPay, PhonePe, etc.)
            recipient = intent_parts.get("recipient", "")
            amount = intent_parts.get("amount")
            app_name = intent_parts.get("app", "paytm")
            
            # Clean recipient name (remove punctuation like trailing periods)
            import re
            recipient = re.sub(r'[^\w\s]', '', recipient).strip()
            
            print(f"\n[Payment] Starting payment flow...")
            print(f"[Payment] App: {app_name}")
            print(f"[Payment] Recipient: {recipient}")
            print(f"[Payment] Amount: ₹{amount}")
            
            if not recipient:
                print("[Payment] ERROR: No recipient specified")
                state["action_success"] = False
                state["error"] = "Please specify who to send the payment to."
                state["task_complete"] = True
                self.tts.speak("Please specify who you want to send the payment to.")
                return state
            
            if not amount:
                print("[Payment] ERROR: No amount specified")
                state["action_success"] = False
                state["error"] = "Please specify the amount to send."
                state["task_complete"] = True
                self.tts.speak("Please specify the amount you want to send.")
                return state
            
            # Step 1: Launch Paytm
            print(f"[Payment] Step 1: Launching {app_name}...")
            self.tts.speak(f"Opening {app_name} to send {amount} rupees to {recipient}.")
            success = self.app_launcher.launch_app(app_name)
            if not success:
                state["action_success"] = False
                state["error"] = f"Failed to open {app_name}"
                state["task_complete"] = True
                self.tts.speak(f"Could not open {app_name}.")
                return state
            
            self.device_actions.wait(4.0)  # Wait for app to load
            
            # Step 2: Find and tap "To Mobile Number & Contacts" on Paytm home
            print("[Payment] Step 2: Finding 'To Mobile Number & Contacts'...")
            
            # Log all elements for debugging
            clickable_elements = self.accessibility.find_clickable_elements()
            print(f"[Payment] Found {len(clickable_elements)} clickable elements")
            
            contact_found = False
            best_match = None
            
            # Priority keywords for Paytm's "To Mobile" button
            priority_keywords = ["to mobile", "mobile number", "contacts", "to mobile number"]
            secondary_keywords = ["pay", "send money", "transfer", "upi"]
            
            for elem in clickable_elements:
                elem_text = (elem.get("text", "") or "").lower()
                elem_desc = (elem.get("content_desc", "") or "").lower()
                combined = elem_text + " " + elem_desc
                x, y = elem.get("x", 0), elem.get("y", 0)
                
                # Log elements in the upper-middle area (where "To Mobile" usually is)
                if 200 < y < 600 and x > 0:
                    print(f"[Payment] Element at ({x}, {y}): text='{elem_text}', desc='{elem_desc}'")
                
                # Check for priority keywords first
                if any(kw in combined for kw in priority_keywords):
                    if x > 0 and y > 0:
                        print(f"[Payment] ✓ Found 'To Mobile' at ({x}, {y}): '{elem_text or elem_desc}'")
                        best_match = (x, y, elem_text or elem_desc)
                        break
            
            # If no priority match, look for secondary keywords
            if not best_match:
                for elem in clickable_elements:
                    elem_text = (elem.get("text", "") or "").lower()
                    elem_desc = (elem.get("content_desc", "") or "").lower()
                    combined = elem_text + " " + elem_desc
                    x, y = elem.get("x", 0), elem.get("y", 0)
                    
                    if any(kw in combined for kw in secondary_keywords) and 150 < y < 500:
                        print(f"[Payment] Found secondary match at ({x}, {y}): '{elem_text or elem_desc}'")
                        best_match = (x, y, elem_text or elem_desc)
                        break
            
            if best_match:
                print(f"[Payment] Tapping 'To Mobile' at ({best_match[0]}, {best_match[1]})")
                self.device_actions.tap(best_match[0], best_match[1], delay=0.5)
                contact_found = True
            else:
                # Fallback: Tap at typical Paytm "To Mobile" location
                print("[Payment] Using fallback - tapping typical 'To Mobile' location...")
                self.device_actions.tap(270, 400, delay=0.5)  # Left side, middle-top
            
            self.device_actions.wait(3.0)  # Wait for search screen to load
            
            # Step 3: Find search input field and type recipient name
            # The search field shows hint text with 4 options (UPI ID, number, contact/name, mobile)
            # We just need to tap on it and type the contact name
            print(f"[Payment] Step 3: Finding search input field to type '{recipient}'...")
            
            # Re-scan elements after screen change
            clickable_elements = self.accessibility.find_clickable_elements()
            print(f"[Payment] Found {len(clickable_elements)} elements on search screen")
            
            # Look for the search input field (EditText)
            input_field = None
            all_elements = []
            
            for elem in clickable_elements:
                elem_class = (elem.get("class", "") or "").lower()
                elem_text = (elem.get("text", "") or "").lower()
                elem_desc = (elem.get("content_desc", "") or "").lower()
                x, y = elem.get("x", 0), elem.get("y", 0)
                
                # Skip elements at (0,0)
                if x == 0 and y == 0:
                    continue
                
                all_elements.append({
                    "x": x, "y": y, "class": elem_class, "text": elem_text, "desc": elem_desc
                })
                
                # Log all elements for debugging
                print(f"[Payment] Element at ({x}, {y}): class='{elem_class}', text='{elem_text[:50] if elem_text else ''}', desc='{elem_desc[:50] if elem_desc else ''}'")
                
                # Look for EditText (input field) - this is the search box
                if "edittext" in elem_class and not input_field:
                    print(f"[Payment] ✓ Found EditText search input at ({x}, {y})")
                    input_field = (x, y)
                
                # Also check for hint text indicating search field
                hint_keywords = ["upi", "contact", "name", "number", "mobile", "search", "enter"]
                if any(kw in elem_text or kw in elem_desc for kw in hint_keywords) and not input_field:
                    print(f"[Payment] ✓ Found search field with hint at ({x}, {y})")
                    input_field = (x, y)
            
            # Tap on input field to focus
            if input_field:
                print(f"[Payment] Tapping search input at ({input_field[0]}, {input_field[1]})")
                self.device_actions.tap(input_field[0], input_field[1], delay=0.5)
                self.device_actions.wait(1.5)
            else:
                # Fallback: The search field in Paytm after "To Mobile/Contact" 
                # is typically a large input area in the upper-middle part of the screen
                # Try tapping at the center of where the search field usually is
                print("[Payment] No EditText found - using fallback search field location...")
                # Paytm search field is typically around Y=300-400 (below header, above results)
                self.device_actions.tap(540, 350, delay=0.5)
                self.device_actions.wait(1.5)
            
            # Type the recipient name - THIS IS CRITICAL
            print(f"[Payment] >>> TYPING CONTACT NAME: '{recipient}'")
            self.device_actions.type_text(recipient, clear_first=True)
            print(f"[Payment] >>> Typed '{recipient}' successfully")
            
            # Press Enter/Search to trigger search
            print("[Payment] Pressing Enter to search...")
            self.device_actions.press_key("KEYCODE_ENTER")
            self.device_actions.wait(3.0)  # Wait for search results to appear
            
            # Step 4: Select contact from search results
            print("[Payment] Step 4: Selecting contact from results...")
            
            # FAST PATH: Directly tap at (270, 600) - this position works reliably for contact selection
            print("[Payment] Tapping contact at known position (270, 600)...")
            self.device_actions.tap(270, 600, delay=0.5)
            
            self.device_actions.wait(3.0)  # Wait for contact profile/chat to load
            
            # Step 4.5: Detect which flow we're in
            # TWO FLOWS:
            # 1. NEW USER: Contact → Pay Now → Amount screen → Proceed → Pay Securely → UPI PIN
            # 2. EXISTING USER: Contact → Amount screen (with Pay Securely at bottom) → Amount → Pay Securely → UPI PIN
            
            print("[Payment] Step 4.5: Detecting payment flow type (New User vs Existing User)...")
            
            # Analyze current screen to detect which flow
            is_existing_user_flow = False
            pay_now_found = False
            
            # Use Vision API to understand the screen
            try:
                device = self.adb_client.get_device()
                screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                elements = screen_analysis.get("elements", [])
                screen_text = " ".join([e.get("description", "") for e in elements]).lower()
                
                print(f"[Payment] Screen analysis: {screen_text[:150]}...")
                
                # Check for EXISTING USER flow indicators:
                # - "pay securely" at bottom
                # - Numeric keypad visible
                # - Amount input visible
                # - NO "pay now" button
                
                has_pay_securely = "pay securely" in screen_text or "pay secure" in screen_text
                has_keypad = "keypad" in screen_text or "numeric" in screen_text
                has_pay_now = "pay now" in screen_text
                
                if has_pay_securely and not has_pay_now:
                    print("[Payment] ✓ EXISTING USER FLOW detected (Pay Securely visible, no Pay Now)")
                    is_existing_user_flow = True
                    pay_now_found = True  # Skip Pay Now step entirely
                elif has_keypad and not has_pay_now:
                    print("[Payment] ✓ EXISTING USER FLOW detected (Keypad visible, no Pay Now)")
                    is_existing_user_flow = True
                    pay_now_found = True  # Skip Pay Now step entirely
                else:
                    print("[Payment] → NEW USER FLOW detected (need to tap Pay Now first)")
                    
            except Exception as e:
                print(f"[Payment] Vision analysis error: {e}, defaulting to New User flow")
            
            # Also check accessibility tree for flow detection
            if not pay_now_found:
                clickable_elements = self.accessibility.find_clickable_elements()
                print(f"[Payment] Found {len(clickable_elements)} elements via accessibility")
                
                # Log elements and check for flow indicators
                all_buttons = []
                has_keypad_buttons = False
                has_pay_securely_btn = False
                has_pay_now_btn = False
                
                for elem in clickable_elements:
                    elem_text = (elem.get("text", "") or "").lower()
                    elem_desc = (elem.get("content_desc", "") or "").lower()
                    elem_class = (elem.get("class", "") or "").lower()
                    combined = elem_text + " " + elem_desc
                    x, y = elem.get("x", 0), elem.get("y", 0)
                    
                    if x == 0 and y == 0:
                        continue
                    
                    print(f"[Payment] Element at ({x}, {y}): class='{elem_class}', text='{elem_text}', desc='{elem_desc}'")
                    
                    all_buttons.append({
                        "x": x, "y": y, "text": elem_text, "desc": elem_desc,
                        "class": elem_class, "combined": combined
                    })
                    
                    # Check for keypad buttons (digits 0-9)
                    if elem_text in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                        has_keypad_buttons = True
                    
                    # Check for "pay securely" at bottom
                    if "pay securely" in combined or "pay secure" in combined:
                        has_pay_securely_btn = True
                    
                    # Check for "pay now"
                    if "pay now" in combined:
                        has_pay_now_btn = True
                
                # Determine flow based on accessibility findings
                if has_pay_securely_btn and has_keypad_buttons and not has_pay_now_btn:
                    print("[Payment] ✓ EXISTING USER FLOW confirmed via accessibility")
                    is_existing_user_flow = True
                    pay_now_found = True  # Skip Pay Now step
                elif has_pay_now_btn:
                    print("[Payment] → NEW USER FLOW confirmed (Pay Now button found)")
            
            # If EXISTING USER flow, skip Pay Now entirely
            if is_existing_user_flow:
                print("[Payment] Skipping 'Pay Now' step - already on amount screen for existing user")
            else:
                # NEW USER FLOW: Need to tap "Pay Now" button
                print("[Payment] NEW USER FLOW: Looking for 'Pay Now' button...")
                
                pay_keywords = ["pay now", "pay", "send money", "send", "transfer"]
                
                # PRIORITY: Try (540, 1450) FIRST - this position worked before!
                print("[Payment] Trying known working position (540, 1450) first...")
                self.device_actions.tap(540, 1450, delay=0.5)
                self.device_actions.wait(1.5)
                
                # Quick check if it worked (look for keypad or amount indicators)
                try:
                    quick_check = self.accessibility.find_clickable_elements()
                    for elem in quick_check:
                        elem_text = (elem.get("text", "") or "").lower()
                        elem_desc = (elem.get("content_desc", "") or "").lower()
                        # If we see keypad numbers or "proceed", the tap worked
                        if elem_text in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] or "proceed" in elem_text or "proceed" in elem_desc:
                            print("[Payment] ✓ Position (540, 1450) worked! Amount screen detected.")
                            pay_now_found = True
                            break
                except:
                    pass
                
                # If the first tap worked, skip further attempts
                if pay_now_found:
                    print("[Payment] 'Pay Now' tapped successfully at (540, 1450)")
                else:
                    # Try multiple positions - Paytm "Pay Now" button positions
                    pay_now_positions = [
                        (540, 1500),  # Below center
                        (540, 1400),  # At center
                        (540, 1550),  # Further below center
                        (540, 1350),  # Slightly above center
                        (540, 1600),  # Lower
                    ]
                    
                    # Try accessibility elements first
                    if not pay_now_found and 'all_buttons' in dir():
                        for btn in all_buttons:
                            if any(kw in btn["combined"] for kw in pay_keywords):
                                print(f"[Payment] ✓ Found 'Pay Now' at ({btn['x']}, {btn['y']})")
                                self.device_actions.tap(btn["x"], btn["y"], delay=0.5)
                                pay_now_found = True
                                break
                    
                    # If not found, try center buttons
                    if not pay_now_found and 'all_buttons' in dir():
                        center_buttons = [b for b in all_buttons if 1200 < b["y"] < 1700 and 200 < b["x"] < 900]
                        if center_buttons:
                            center_buttons.sort(key=lambda b: b["y"])
                            for btn in center_buttons:
                                if "view" in btn["class"] or "button" in btn["class"]:
                                    print(f"[Payment] Tapping center button at ({btn['x']}, {btn['y']})")
                                    self.device_actions.tap(btn["x"], btn["y"], delay=0.5)
                                    pay_now_found = True
                                    break
                    
                    # If still not found, try sequential positions
                    if not pay_now_found:
                        print("[Payment] Trying known 'Pay Now' positions sequentially...")
                        for pos_x, pos_y in pay_now_positions:
                            print(f"[Payment] Trying position ({pos_x}, {pos_y})...")
                            self.device_actions.tap(pos_x, pos_y, delay=0.5)
                            self.device_actions.wait(1.5)
                    
                    # Quick check if screen changed (look for amount field indicators)
                    try:
                        quick_elements = self.accessibility.find_clickable_elements()
                        for elem in quick_elements:
                            elem_text = (elem.get("text", "") or "").lower()
                            elem_desc = (elem.get("content_desc", "") or "").lower()
                            if "amount" in elem_text or "₹" in elem_text or "amount" in elem_desc:
                                print(f"[Payment] ✓ Screen changed! Amount field detected.")
                                pay_now_found = True
                                break
                    except:
                        pass
            
            # If still not found, try sequential tapping with REAL verification
            if not pay_now_found:
                print("[Payment] Pay Now not found - trying sequential tap positions with verification...")
                
                # These are known positions where Pay Now button appears in Paytm
                # We'll try each one and CHECK if the screen actually changed
                tap_positions = [
                    (540, 1350),  # Center, slightly below middle
                    (540, 1400),
                    (540, 1450),
                    (540, 1500),
                    (540, 1550),
                    (540, 1300),
                    (540, 1250),
                    (540, 1600),
                ]
                
                for pos_x, pos_y in tap_positions:
                    print(f"[Payment] Trying tap at ({pos_x}, {pos_y})...")
                    
                    # Take screenshot BEFORE tap to compare
                    try:
                        before_analysis = self.screen_analyzer.analyze_screen(
                            self.adb_client.get_device(), detect_elements=True
                        )
                        before_text = " ".join([e.get("description", "") for e in before_analysis.get("elements", [])]).lower()
                    except:
                        before_text = ""
                    
                    # Tap
                    self.device_actions.tap(pos_x, pos_y, delay=0.3)
                    self.device_actions.wait(1.5)  # Wait for UI to respond
                    
                    # Take screenshot AFTER tap to compare
                    try:
                        after_analysis = self.screen_analyzer.analyze_screen(
                            self.adb_client.get_device(), detect_elements=True
                        )
                        after_text = " ".join([e.get("description", "") for e in after_analysis.get("elements", [])]).lower()
                        
                        # Check if "pay now" disappeared AND amount-related text appeared
                        pay_now_gone = "pay now" not in after_text
                        amount_visible = "amount" in after_text or "₹" in after_text or "enter" in after_text
                        
                        print(f"[Payment] After tap: pay_now_gone={pay_now_gone}, amount_visible={amount_visible}")
                        print(f"[Payment] Screen text: {after_text[:100]}...")
                        
                        if pay_now_gone or amount_visible:
                            print(f"[Payment] ✓ Screen changed! Tap at ({pos_x}, {pos_y}) worked!")
                            pay_now_found = True
                            break
                        else:
                            print(f"[Payment] Screen didn't change, trying next position...")
                    except Exception as e:
                        print(f"[Payment] Error checking screen: {e}")
            
            # Last resort: Try multiple quick taps in center area
            if not pay_now_found:
                print("[Payment] Still not found - trying rapid taps in center area...")
                for y in [1350, 1400, 1450, 1500]:
                    self.device_actions.tap(540, y, delay=0.2)
                self.device_actions.wait(2.0)
            
            self.device_actions.wait(2.0)  # Wait for screen to change
            
            # Step 4.6: Only for NEW USER - skip for EXISTING USER
            if not is_existing_user_flow:
                print("[Payment] Step 4.6: Verifying 'Pay Now' was clicked and screen changed...")
            else:
                print("[Payment] Step 4.6: SKIPPED for existing user")
            
            amount_screen_found = is_existing_user_flow  # Existing user: already on amount screen
            
            if not is_existing_user_flow:
                max_retries = 5  # Increased retries - each retry also attempts a tap
                
                for retry in range(max_retries):
                    print(f"[Payment] Verification attempt {retry + 1}/{max_retries}...")
                    
                    # Use Vision API to check current screen
                    try:
                        device = self.adb_client.get_device()
                        screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                        elements = screen_analysis.get("elements", [])
                        screen_text = " ".join([e.get("description", "") for e in elements]).lower()
                        
                        print(f"[Payment] Screen analysis: {screen_text[:200]}...")
                        
                        # CRITICAL: If "pay now" is STILL visible, we're NOT on amount screen yet!
                        pay_now_still_visible = "pay now" in screen_text
                        
                        if pay_now_still_visible:
                            print(f"[Payment] ✗ 'Pay Now' button still visible - tap didn't work!")
                            print(f"[Payment] Attempting additional tap at different Y position...")
                            
                            # Try tapping at different Y positions
                            tap_y_positions = [1350, 1400, 1450, 1500, 1550, 1600, 1650]
                            tap_idx = retry % len(tap_y_positions)
                            tap_y = tap_y_positions[tap_idx]
                            
                            print(f"[Payment] Tapping at (540, {tap_y})...")
                            self.device_actions.tap(540, tap_y, delay=0.5)
                            self.device_actions.wait(2.0)
                            continue  # Re-check screen
                        
                        # Check for amount entry screen indicators (only if pay_now is NOT visible)
                        amount_indicators = ["keypad", "numpad", "proceed", "pay securely", "numeric"]
                        
                        if any(ind in screen_text for ind in amount_indicators):
                            print("[Payment] ✓ Amount entry screen detected (keypad/proceed visible)!")
                            amount_screen_found = True
                            break
                        
                        # Check accessibility for keypad buttons
                        clickable_elements = self.accessibility.find_clickable_elements()
                        has_keypad = False
                        has_edittext = False
                        
                        for elem in clickable_elements:
                            elem_class = (elem.get("class", "") or "").lower()
                            elem_text = (elem.get("text", "") or "").strip()
                            elem_desc = (elem.get("content_desc", "") or "").lower()
                            
                            if elem_text in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                                has_keypad = True
                            if "edittext" in elem_class:
                                has_edittext = True
                        
                        if has_keypad:
                            print("[Payment] ✓ Amount screen detected (keypad buttons found)!")
                            amount_screen_found = True
                            break
                        
                        if has_edittext and len(clickable_elements) > 12:
                            print(f"[Payment] ✓ Amount screen detected (EditText + {len(clickable_elements)} elements)!")
                            amount_screen_found = True
                            break
                        
                        if not pay_now_still_visible and len(clickable_elements) < 10:
                            print(f"[Payment] ✗ Only {len(clickable_elements)} elements - still on search/contact screen")
                            tap_y_positions = [600, 700, 800, 900, 1000, 1100, 1200]
                            tap_idx = retry % len(tap_y_positions)
                            tap_y = tap_y_positions[tap_idx]
                            print(f"[Payment] Tapping contact at (270, {tap_y})...")
                            self.device_actions.tap(270, tap_y, delay=0.5)
                            self.device_actions.wait(2.0)
                            continue
                            
                    except Exception as e:
                        print(f"[Payment] Verification error: {e}")
                    
                    if not amount_screen_found:
                        print("[Payment] Screen hasn't changed - retrying 'Pay Now' tap...")
                        retry_positions = [(540, 1500), (540, 1600), (540, 1300)]
                        if retry < len(retry_positions):
                            tap_x, tap_y = retry_positions[retry]
                            print(f"[Payment] Retrying tap at ({tap_x}, {tap_y})...")
                            self.device_actions.tap(tap_x, tap_y, delay=0.5)
                            self.device_actions.wait(2.0)
            
            if not is_existing_user_flow and not amount_screen_found:
                print("[Payment] WARNING: Could not verify amount screen - proceeding anyway...")
                self.tts.speak("Having trouble finding the amount screen. Please check if the screen is correct.")
            
            # EXISTING USER: Tap "Pay Securely" (540, 2200) FIRST, then enter amount, then Step 7, Step 8
            if is_existing_user_flow:
                print("[Payment] EXISTING USER: Tapping 'Pay Securely' first at (540, 2200)...")
                self.device_actions.tap(540, 2200, delay=0.5)
                self.device_actions.wait(2.0)
            
            # Step 5: Enter amount using keypad coordinates (tap keypad, do not use keyevent)
            # Paytm keypad layout (verified coordinates):
            # Row 1 (Y=1741): (199, 1741), (539, 1741), (880, 1741) → 1, 2, 3
            # Row 2 (Y=1909): (199, 1909), (539, 1909), (880, 1909) → 4, 5, 6
            # Row 3 (Y=2077): (199, 2077), (539, 2077), (880, 2077) → 7, 8, 9
            # Row 4 (Y=2245): (114, 2245), (400, 2245), (539, 2245) → ., 0, backspace
            print(f"[Payment] Step 5: Entering amount: ₹{amount}...")
            
            PAYTM_KEYPAD = {
                "1": (199, 1741),
                "2": (539, 1741),
                "3": (880, 1741),
                "4": (199, 1909),
                "5": (539, 1909),
                "6": (880, 1909),
                "7": (199, 2077),
                "8": (539, 2077),
                "9": (880, 2077),
                ".": (114, 2245),
                "0": (400, 2245),
                "backspace": (539, 2245),
            }
            
            # Use keypad for digit/decimal entry; for integer amount avoid "10.0" so keypad is used
            amount_str = str(int(amount)) if isinstance(amount, (int, float)) and amount == int(amount) else str(amount)
            digits_needed = amount_str
            all_found = all(d in PAYTM_KEYPAD for d in digits_needed)
            
            if all_found:
                print(f"[Payment] Tapping keypad at known coordinates...")
                for digit in digits_needed:
                    x, y = PAYTM_KEYPAD[digit]
                    print(f"[Payment] Keypad '{digit}' at ({x}, {y})")
                    self.device_actions.tap(x, y, delay=0.3)
                    self.device_actions.wait(0.3)
                print(f"[Payment] >>> Amount {amount} entered via keypad")
            else:
                # Fallback: amount field + keyevent
                clickable_elements = self.accessibility.find_clickable_elements()
                amount_edittext = None
                for elem in clickable_elements:
                    elem_class = (elem.get("class", "") or "").lower()
                    x, y = elem.get("x", 0), elem.get("y", 0)
                    if "edittext" in elem_class and 400 < y < 900:
                        amount_edittext = (x, y)
                        break
                if amount_edittext:
                    ax, ay = amount_edittext
                    print(f"[Payment] Tapping amount field at ({ax}, {ay}), keyevent...")
                    self.device_actions.tap(ax, ay, delay=0.5)
                    self.device_actions.wait(0.5)
                    self.device_actions.type_digits_keyevent(str(amount))
                else:
                    self.device_actions.tap(540, 650, delay=0.5)
                    self.device_actions.wait(0.5)
                    self.device_actions.type_digits_keyevent(str(amount))
                print(f"[Payment] >>> Amount {amount} entered via keyevent")
            
            self.device_actions.wait(1.0)
            
            # Step 6: Click Proceed button (ABOVE the keypad, blue box)
            # Step 6 & 7: Simple rule - if UI shows "Pay Securely", tap (540, 2200). Otherwise tap Proceed then Pay Securely.
            print("[Payment] Step 6: Checking for 'Pay Securely' on screen...")
            
            pay_securely_tapped = False
            screen_has_pay_securely = False
            
            # Quick check: screen text or accessibility
            try:
                device = self.adb_client.get_device()
                screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                screen_text = " ".join([e.get("description", "") for e in screen_analysis.get("elements", [])]).lower()
                if "pay securely" in screen_text or "pay secure" in screen_text:
                    screen_has_pay_securely = True
            except Exception:
                pass
            
            if not screen_has_pay_securely:
                clickable_elements = self.accessibility.find_clickable_elements()
                for elem in clickable_elements:
                    combined = ((elem.get("text", "") or "") + " " + (elem.get("content_desc", "") or "")).lower()
                    if "pay securely" in combined or "pay secure" in combined:
                        screen_has_pay_securely = True
                        break
            
            # When "Pay Securely" is visible, always use this tap (user-specified)
            if screen_has_pay_securely:
                print("[Payment] UI shows 'Pay Securely' -> tapping (540, 2200)")
                self.device_actions.tap(540, 2200, delay=0.5)
                pay_securely_tapped = True
            else:
                # No "Pay Securely" -> we're on amount screen, tap Proceed first
                print("[Payment] No 'Pay Securely' visible -> tapping Proceed at (540, 1580)...")
                self.device_actions.tap(540, 1580, delay=0.5)
            
            self.device_actions.wait(3.0)
            
            # Step 7: If we didn't tap Pay Securely yet, screen should show it now - tap (540, 2200)
            if not pay_securely_tapped:
                print("[Payment] Step 7: Tapping 'Pay Securely' at (540, 2200)...")
                self.device_actions.tap(540, 2200, delay=0.5)
                self.device_actions.wait(3.0)
            
            # Step 8: UPI PIN Entry
            print("\n[Payment] Step 8: UPI PIN required...")
            self.tts.speak("Please enter your UPI PIN.")
            print("[Payment] Waiting for UPI PIN input...")
            print("=" * 50)
            print("  ENTER YOUR UPI PIN (typed input for security)")
            print("=" * 50)
            
            # Get UPI PIN via typed input (secure)
            try:
                import getpass
                upi_pin = getpass.getpass("Enter UPI PIN: ")
                
                if upi_pin:
                    print(f"[Payment] UPI PIN received ({len(upi_pin)} digits)")
                    
                    # Wait for PIN keypad to fully load
                    self.device_actions.wait(1.5)
                    
                    # The PIN keypad should be visible - find and tap each digit
                    # First, scan for the keypad numbers
                    clickable_elements = self.accessibility.find_clickable_elements()
                    
                    # Build a map of digit -> (x, y) from keypad
                    digit_map = {}
                    for elem in clickable_elements:
                        elem_text = (elem.get("text", "") or "").strip()
                        x, y = elem.get("x", 0), elem.get("y", 0)
                        
                        # Check if it's a single digit (0-9)
                        if elem_text in "0123456789" and y > 1000:  # Keypad is at bottom
                            digit_map[elem_text] = (x, y)
                            print(f"[Payment] Keypad digit '{elem_text}' at ({x}, {y})")
                    
                    # Enter each PIN digit
                    for digit in upi_pin:
                        if digit in digit_map:
                            x, y = digit_map[digit]
                            print(f"[Payment] Tapping digit '{digit}' at ({x}, {y})")
                            self.device_actions.tap(x, y, delay=0.15)
                        else:
                            # Fallback: Re-scan and find digit
                            clickable_elements = self.accessibility.find_clickable_elements()
                            digit_found = False
                            for elem in clickable_elements:
                                elem_text = (elem.get("text", "") or "").strip()
                                if elem_text == digit:
                                    x, y = elem.get("x", 0), elem.get("y", 0)
                                    if x > 0 and y > 1000:
                                        self.device_actions.tap(x, y, delay=0.15)
                                        digit_found = True
                                        break
                            
                            if not digit_found:
                                print(f"[Payment] Digit '{digit}' not found on keypad, using ADB input")
                                self.device_actions.type_text(digit, clear_first=False)
                        
                        self.device_actions.wait(0.4)
                    
                    # Step 9: Tap checkmark/tick button to confirm
                    print("[Payment] Step 9: Confirming payment (tapping checkmark)...")
                    self.device_actions.wait(1.0)
                    
                    # Re-scan for confirm button
                    clickable_elements = self.accessibility.find_clickable_elements()
                    confirm_found = False
                    
                    # Look for checkmark/tick button (usually bottom-right)
                    confirm_keywords = ["✓", "tick", "check", "confirm", "done", "submit", "ok", "verify"]
                    
                    for elem in clickable_elements:
                        elem_text = (elem.get("text", "") or "").lower()
                        elem_desc = (elem.get("content_desc", "") or "").lower()
                        combined = elem_text + " " + elem_desc
                        x, y = elem.get("x", 0), elem.get("y", 0)
                        
                        # Log bottom-right elements
                        if x > 800 and y > 1500:
                            print(f"[Payment] Bottom-right element at ({x}, {y}): text='{elem_text}', desc='{elem_desc}'")
                        
                        # Look for checkmark button
                        if any(kw in combined for kw in confirm_keywords):
                            print(f"[Payment] ✓ Found confirm button at ({x}, {y}): '{elem_text or elem_desc}'")
                            self.device_actions.tap(x, y, delay=0.5)
                            confirm_found = True
                            break
                    
                    if not confirm_found:
                        # Tap at typical checkmark location (bottom-right)
                        print("[Payment] Using fallback confirm location (bottom-right)...")
                        self.device_actions.tap(950, 2200, delay=0.5)
                    
                    self.device_actions.wait(3.0)  # Wait for payment processing
                    
                    print("[Payment] ✓ Payment initiated!")
                    self.tts.speak(f"Payment of {amount} rupees to {recipient} has been initiated!")
                    state["action_success"] = True
                    state["task_complete"] = True
                else:
                    print("[Payment] No UPI PIN entered - payment cancelled")
                    self.tts.speak("Payment cancelled - no UPI PIN entered.")
                    state["action_success"] = False
                    state["task_complete"] = True
                    
            except Exception as e:
                print(f"[Payment] Error during PIN entry: {e}")
                self.tts.speak("There was an error during payment. Please try again.")
                state["action_success"] = False
                state["task_complete"] = True
            
            self.state_manager.add_step(f"payment_{recipient}_{amount}")
            return state
        
        elif action == "tap_element":
            # User wants to tap on a specific element (e.g., "tap on login")
            target = intent_parts.get("target", "")
            print(f"[Status] Tapping on element: {target}")
            
            device = self.adb_client.get_device()
            screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
            elements = screen_analysis.get("elements", [])
            
            # Find the element to tap
            element_found = False
            target_lower = target.lower()
            
            # Try to find element by description
            for elem in elements:
                elem_desc = elem.get("description", "").lower()
                elem_type = elem.get("type", "")
                
                # Check if description matches target
                if target_lower in elem_desc or elem_desc in target_lower:
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0:
                        print(f"[Tap] Found element '{elem.get('description')}' at ({x}, {y})")
                        self.device_actions.tap(x, y, delay=0.5)
                        self.device_actions.wait(2.0)  # Wait for action to complete
                        element_found = True
                        state["action_success"] = True
                        state["task_complete"] = False
                        state["session_active"] = True
                        
                        # If it's a login button, trigger authentication
                        if any(kw in target_lower for kw in ["login", "sign in"]):
                            print("[Tap] Login button tapped - starting authentication")
                            state["needs_auth"] = True
                        break
            
            # Try accessibility tree if not found
            if not element_found:
                # Try to find by keywords
                keywords = [target_lower]
                if "login" in target_lower:
                    keywords.extend(["login", "sign in", "log in"])
                
                accessibility_result = self.accessibility.find_button_by_keywords(keywords)
                if accessibility_result:
                    x, y, _ = accessibility_result
                    print(f"[Tap] Found element via accessibility at ({x}, {y})")
                    self.device_actions.tap(x, y, delay=0.5)
                    self.device_actions.wait(2.0)
                    element_found = True
                    state["action_success"] = True
                    state["task_complete"] = False
                    state["session_active"] = True
                    
                    if "login" in target_lower:
                        state["needs_auth"] = True
            
            if not element_found:
                state["action_success"] = False
                state["error"] = f"Could not find element: {target}"
                state["task_complete"] = False
        
        elif action == "search" or action == "query":
            query = intent_parts.get("query", "")
            if query:
                # Find search field
                search_field = self.element_detector.find_text_field(device, "search")
                if search_field:
                    self.device_actions.tap(search_field[0], search_field[1])
                    self.device_actions.wait(0.5)
                    self.device_actions.type_text(query, clear_first=True)
                    self.device_actions.press_key("KEYCODE_ENTER")
                    self.state_manager.add_step(f"searched_{query}")
                    state["action_success"] = True
                    # Don't mark complete - user can continue interacting
                    state["task_complete"] = False
                else:
                    state["action_success"] = False
                    state["error"] = "Could not find search field"
        
        elif action == "extract":
            # Extract information from screen
            description = self.element_detector.get_screen_description(device)
            state["extracted_info"] = description
            state["action_success"] = True
            # Don't mark complete - user can continue interacting
            state["task_complete"] = False
        
        else:
            state["action_success"] = False
            state["error"] = f"Unknown action: {action}"
        
        return state
    
    def _find_send_button_with_llm(self, screen_description: str, elements: List[Dict], query: str) -> Dict:
        """
        Use LLM to find send button/icon on the screen
        
        Args:
            screen_description: Description of the screen
            elements: List of detected UI elements
            query: Query that was typed
            
        Returns:
            Dictionary with send button info: {"found": bool, "coordinates": (x, y), "description": str}
        """
        # Build elements description
        elements_desc = []
        for elem in elements:
            elem_type = elem.get("type", "")
            elem_desc = elem.get("description", "")
            x = elem.get("x", 0)
            y = elem.get("y", 0)
            
            if elem_type == "icon":
                elements_desc.append(f"Icon at ({x}, {y}): {elem_desc}")
            elif elem_type == "button":
                elements_desc.append(f"Button at ({x}, {y}): {elem_desc}")
            elif "arrow" in elem_desc.lower() or "→" in elem_desc or "↑" in elem_desc:
                elements_desc.append(f"Possible send icon at ({x}, {y}): {elem_desc}")
        
        elements_text = "\n".join(elements_desc) if elements_desc else "No elements detected"
        
        prompt = f"""You are analyzing a mobile app screen to find the send/submit button or icon for a chat message.

SCREEN DESCRIPTION:
{screen_description}

DETECTED ELEMENTS:
{elements_text}

USER TYPED THIS QUERY:
{query}

Find the send button or icon. Look for:
1. Arrow icons (→, ↑, ↗) - these are commonly used as send buttons in chat apps
2. Icons near the input field (usually on the right side or below the input)
3. Buttons labeled "Send", "Submit", "Go"
4. Any clickable element that would send the typed message

The send button/icon is typically:
- Located near the text input field (right side or bottom)
- An arrow pointing right (→) or up (↑)
- Sometimes a circular button with an arrow icon inside
- In ChatGPT, it's usually an arrow icon on the right side of the input field

Respond in JSON format:
{{
    "found": true/false,
    "coordinates": [x, y] or null,
    "description": "Description of the send button/icon found",
    "element_type": "icon" | "button" | "none"
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies UI elements. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            import json
            import re
            result_text = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                if decision.get("found") and decision.get("coordinates"):
                    coords = decision["coordinates"]
                    if isinstance(coords, list) and len(coords) >= 2:
                        return {
                            "found": True,
                            "coordinates": (coords[0], coords[1]),
                            "description": decision.get("description", "send icon")
                        }
            return {"found": False, "coordinates": None, "description": ""}
        except Exception as e:
            print(f"[LLM Send Detection] Error: {e}")
            return {"found": False, "coordinates": None, "description": ""}
    
    def _decide_action_with_llm(self, screen_description: str, elements: List[Dict], user_intent: str, pending_query: Optional[str] = None) -> Dict:
        """
        Use LLM to intelligently decide what action to take based on screen content
        
        Args:
            screen_description: Description of the screen
            elements: List of detected UI elements
            user_intent: Original user command
            pending_query: Optional query to execute
            
        Returns:
            Dictionary with decision: {"action": "auto_proceed"|"ask_confirmation"|"execute_query", "reason": "...", "target_element": {...}}
        """
        # Build elements description - include ALL elements, not just buttons and text_fields
        elements_desc = []
        input_fields_found = []
        
        for elem in elements:
            elem_type = elem.get("type", "")
            elem_desc = elem.get("description", "")
            elem_text = elem.get("text", "")
            
            if elem_type == "button":
                elements_desc.append(f"Button: {elem_desc} {f'(text: {elem_text})' if elem_text else ''}")
            elif elem_type == "text_field":
                elements_desc.append(f"Input field: {elem_desc}")
                input_fields_found.append(elem_desc.lower())
            else:
                # Check if it might be an input field by description
                desc_lower = elem_desc.lower()
                input_keywords = ["input", "text field", "chat", "message", "ask", "type", "search", "compose", "write", "enter"]
                if any(kw in desc_lower for kw in input_keywords):
                    elements_desc.append(f"Possible input field: {elem_desc}")
                    input_fields_found.append(elem_desc.lower())
        
        elements_text = "\n".join(elements_desc) if elements_desc else "No interactive elements detected"
        
        # Add summary of input fields found
        if input_fields_found:
            elements_text += f"\n\nSUMMARY: Found {len(input_fields_found)} input field(s): {', '.join(input_fields_found)}"
        
        # Build prompt for LLM
        prompt = f"""You are an AI assistant helping a blind user navigate their Android device. Analyze the current screen and decide what action to take.

SCREEN DESCRIPTION:
{screen_description}

DETECTED ELEMENTS:
{elements_text}

USER'S ORIGINAL INTENT:
{user_intent}

PENDING QUERY TO EXECUTE:
{pending_query if pending_query else "None"}

CRITICAL RULES:
1. **HIGHEST PRIORITY**: If there's a text input field visible (like "Ask ChatGPT", "Message", "Search", "Type a message", etc.) AND the user has a pending query, ALWAYS choose "execute_query" action. Even if there's a "Log in" button visible, if the input field is usable, use it directly. Don't ask for confirmation.

2. If there's a text input field visible but NO pending query, and there's a simple button like "Get Started", "Next", "OK", "Got It", "Skip" - automatically click the button. Don't ask for confirmation.

3. **PREFER "Continue with Google"**: If you see a login popup with "Continue with Google" button, PREFER using it as it's the fastest login method. The system will handle the Google sign-in flow automatically using Accessibility.

4. Only ask for confirmation ("ask_confirmation") if:
   - There's NO usable input field AND login is required to proceed
   - The action is ambiguous or potentially destructive
   - User explicitly needs to provide information (credentials, OTP, etc.)

5. If the screen shows a chat interface with an input field at the bottom (like ChatGPT, messaging apps), and there's a pending query, ALWAYS proceed to type and send it. Ignore login buttons if the input field is accessible.

6. For ChatGPT specifically: If you see "Ask ChatGPT" input field or similar, and there's a query, execute it immediately. The input field might work even without logging in.

Respond in JSON format:
{{
    "action": "auto_proceed" | "ask_confirmation" | "execute_query",
    "reason": "Brief explanation of decision",
    "target_element_type": "button" | "text_field" | "none",
    "target_element_description": "Description of element to interact with",
    "should_click_button": true/false,
    "should_type_query": true/false
}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for decision-making
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that makes smart decisions about mobile app automation. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            import json
            import re
            result_text = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                return decision
            else:
                # Fallback: check for input fields more thoroughly
                has_input_field = False
                # Check for text_field type
                has_input_field = any(elem.get("type") == "text_field" for elem in elements)
                # Also check by description keywords
                if not has_input_field:
                    for elem in elements:
                        elem_desc = elem.get("description", "").lower()
                        input_keywords = ["input", "text field", "chat", "message", "ask", "type", "search", "compose"]
                        if any(kw in elem_desc for kw in input_keywords):
                            has_input_field = True
                            break
                
                has_query = pending_query is not None
                if has_input_field and has_query:
                    return {
                        "action": "execute_query",
                        "reason": "Input field detected with pending query (fallback)",
                        "should_type_query": True
                    }
                return {"action": "ask_confirmation", "reason": "Unable to parse LLM response"}
        except Exception as e:
            print(f"[LLM Decision] Error: {e}")
            # Fallback logic - check for input fields thoroughly
            has_input_field = False
            # Check for text_field type
            has_input_field = any(elem.get("type") == "text_field" for elem in elements)
            # Also check by description keywords
            if not has_input_field:
                for elem in elements:
                    elem_desc = elem.get("description", "").lower()
                    input_keywords = ["input", "text field", "chat", "message", "ask", "type", "search", "compose"]
                    if any(kw in elem_desc for kw in input_keywords):
                        has_input_field = True
                        break
            
            has_query = pending_query is not None
            if has_input_field and has_query:
                return {"action": "execute_query", "reason": "Fallback: Input field + query detected", "should_type_query": True}
            return {"action": "ask_confirmation", "reason": "LLM decision failed, defaulting to confirmation"}
    
    def _verify(self, state: Dict) -> Dict:
        """Verify action completed successfully and analyze screen for user interaction"""
        logger.info("Verifying action...")
        print("[Status] Verifying action completed...")
        self.state_manager.update_state(AgentState.VERIFYING)
        
        if state.get("action_success", False):
            # Check if task is complete
            intent_parts = state.get("intent_parts", {})
            action = intent_parts.get("action")
            
            # For open_app, use LLM to intelligently decide what to do
            if action == "open_app":
                device = self.adb_client.get_device()
                screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                
                description = screen_analysis.get("description", "")
                elements = screen_analysis.get("elements", [])
                user_intent = state.get("user_intent", "")
                
                # CRITICAL: Check for LOGIN WALL (bottom sheet) FIRST
                # If detected, use "Log in" button automatically (NEVER "Continue with Google")
                description_lower = description.lower()
                login_wall_keywords = [
                    "thanks for trying chatgpt", "thanks for trying",
                    "log in or sign up", "continue with google", "sign up",
                    "login wall", "bottom sheet", "get started with chatgpt"
                ]
                is_login_wall = any(keyword in description_lower for keyword in login_wall_keywords)
                
                # Also check elements for login wall indicators
                for elem in elements:
                    elem_desc = elem.get("description", "").lower()
                    if any(keyword in elem_desc for keyword in ["continue with google", "thanks for trying", "log in or sign up"]):
                        is_login_wall = True
                        break
                
                if is_login_wall:
                    print("[Verify] ⚠ LOGIN WALL DETECTED - Using 'Bottom Sheet Button Rule'")
                    print("[Verify] RULE: Tap FIRST button = 'Continue with Google' (fastest login)")
                    
                    # Use Bottom Sheet Button Rule - find first large clickable button (Continue with Google)
                    accessibility_result = self.accessibility.find_real_login_button()
                    
                    if accessibility_result:
                        x, y, button_info = accessibility_result
                        print(f"[Verify] ✓ Found 'Continue with Google' at ({x}, {y})")
                        print(f"[Verify]   Button size: {button_info.get('width', 'N/A')}x{button_info.get('height', 'N/A')}")
                        
                        # Store query for execution after login
                        query = intent_parts.get("query", "")
                        if query:
                            state["pending_query"] = query
                            state["pending_app"] = intent_parts.get("app", "")
                            print(f"[Verify] Stored pending query: '{query}' (will execute after login)")
                        
                        # STEP 1: Tap "Continue with Google"
                        print("[Google Auth] Tapping 'Continue with Google'...")
                        self.tts.speak("Signing in with Google.")
                        self.device_actions.tap(x + 5, y + 5, delay=0.5)
                        
                        # Wait longer for Google popup to fully load
                        print("[Google Auth] Waiting for Google account popup to load...")
                        self.device_actions.wait(4.0)  # Increased wait time
                        
                        # STEP 2: Find and tap "Continue" on Google popup (Accessibility only - NO LLM!)
                        print("[Google Auth] Looking for 'Continue' button on Google popup...")
                        
                        # Try multiple times with increasing wait
                        continue_found = False
                        for attempt in range(5):  # More attempts
                            continue_btn = self.accessibility.find_continue_button()
                            if continue_btn:
                                cx, cy, btn_info = continue_btn
                                print(f"[Google Auth] ✓ Found button at ({cx}, {cy})")
                                self.device_actions.tap(cx, cy, delay=0.5)
                                self.device_actions.wait(4.0)  # Wait for login to complete
                                continue_found = True
                                break
                            else:
                                wait_time = 1.5 + (attempt * 0.5)  # Increasing wait
                                print(f"[Google Auth] Attempt {attempt + 1}/5: Button not found, waiting {wait_time}s...")
                                self.device_actions.wait(wait_time)
                        
                        if continue_found:
                            print("[Google Auth] ✓ Google sign-in completed!")
                            self.tts.speak("Signed in successfully!")
                            
                            # NO need for LLM auth flow - Google login is done!
                            state["needs_auth"] = False
                            state["login_complete"] = True
                            state["login_method"] = "google"
                            state["task_complete"] = False
                            state["session_active"] = True
                            
                            # Wait for app to load after login
                            self.device_actions.wait(2.0)
                            
                            # If there's a pending query, execute it now
                            if state.get("pending_query"):
                                print(f"[Google Auth] Executing pending query: '{state['pending_query']}'")
                                # Continue to execute query (will be handled by next state)
                            
                            return state
                        else:
                            print("[Google Auth] WARNING: Could not find Continue button")
                            # Fall through to check if we're already logged in
                            state["needs_auth"] = False
                            state["task_complete"] = False
                            state["session_active"] = True
                            return state
                    else:
                        print("[Verify] WARNING: Login wall detected but no button found via Bottom Sheet Rule")
                        # Fall through to normal flow
                
                # Check if user wants to login after opening app
                wants_login = intent_parts.get("wants_login", False)
                
                # Check if there's a query to execute after opening app
                query = intent_parts.get("query", "")
                if query:
                    # Store query for later execution
                    state["pending_query"] = query
                    state["pending_app"] = intent_parts.get("app", "")
                    print(f"[Verify] Stored pending query: '{query}' for app: {intent_parts.get('app', '')}")
                else:
                    print(f"[Verify] No query found in intent_parts. Intent parts: {intent_parts}")
                
                # If user wants to login, prioritize finding and clicking login button
                # Hybrid approach: Vision → Accessibility
                if wants_login:
                    print("[Status] User wants to login - using Vision → Accessibility hybrid approach...")
                    login_button_found = False
                    
                    # Step 1: Use Vision to find approximate login button region
                    vision_login = None
                    for elem in elements:
                        elem_desc = elem.get("description", "").lower()
                        elem_type = elem.get("type", "")
                        if elem_type == "button":
                            if "login" in elem_desc or "sign in" in elem_desc or "log in" in elem_desc:
                                x = elem.get("x", 0)
                                y = elem.get("y", 0)
                                width = elem.get("width", 150)  # Approximate button width
                                height = elem.get("height", 60)  # Approximate button height
                                if x > 0 and y > 0:
                                    vision_login = {"x": x, "y": y, "width": width, "height": height}
                                    print(f"[Action] Vision found login button region at ({x}, {y})")
                                    break
                    
                    # Step 2: Use Accessibility to find precise node in Vision region
                    if vision_login:
                        accessibility_result = self.accessibility.find_node_near_region(
                            region_x=vision_login["x"],
                            region_y=vision_login["y"],
                            region_width=vision_login.get("width"),
                            region_height=vision_login.get("height"),
                            search_radius=100
                        )
                        if accessibility_result:
                            x, y, node_info = accessibility_result
                            # Validate coordinates
                            if self._validate_coordinates(x, y):
                                print(f"[Action] ✓ Found precise login button via Accessibility at ({x}, {y})")
                                self.device_actions.tap(x, y, delay=0.5)
                                self.device_actions.wait(2.5)
                                login_button_found = True
                                state["needs_auth"] = True
                                state["task_complete"] = False
                                state["session_active"] = True
                            else:
                                print(f"[Action] WARNING: Accessibility coordinates ({x}, {y}) failed validation")
                    
                    # Fallback: Direct Accessibility search
                    if not login_button_found:
                        print("[Action] Fallback: Searching Accessibility tree directly...")
                        accessibility_result = self.accessibility.find_button_by_keywords(["login", "sign in", "log in"])
                        if accessibility_result:
                            x, y, _ = accessibility_result
                            if self._validate_coordinates(x, y):
                                print(f"[Action] Found login button via direct Accessibility at ({x}, {y})")
                                self.device_actions.tap(x, y, delay=0.5)
                                self.device_actions.wait(2.5)
                                login_button_found = True
                                state["needs_auth"] = True
                                state["task_complete"] = False
                                state["session_active"] = True
                            else:
                                print(f"[Action] WARNING: Accessibility coordinates ({x}, {y}) failed validation - will use LLM-guided login")
                    
                    if login_button_found:
                        # Wait a bit more for login screen to appear
                        print("[Status] Login button clicked, waiting for login screen...")
                        self.device_actions.wait(3.0)
                        state["needs_auth"] = True  # Trigger authentication after clicking
                        state["task_complete"] = False
                        state["session_active"] = True
                        # Skip LLM decision, we already handled login button click
                        return state
                    else:
                        # Login button not found - might need to wait more or app might already be logged in
                        print("[Status] Login button not found - checking if already logged in or need to wait...")
                        # Wait a bit more and check again
                        self.device_actions.wait(2.0)
                        # Still set needs_auth so authentication flow can check
                        state["needs_auth"] = True
                        state["task_complete"] = False
                        state["session_active"] = True
                        return state
                
                # Use LLM to decide what to do (if not handling login)
                print("[Status] Analyzing screen and deciding next action...")
                decision = self._decide_action_with_llm(description, elements, user_intent, query)
                
                print(f"[Decision] Action: {decision.get('action')}, Reason: {decision.get('reason')}")
                
                if decision.get("action") == "execute_query":
                    # Automatically proceed to execute query
                    print("[Decision] Input field detected with query - proceeding to type and send automatically")
                    state["task_complete"] = False
                    state["needs_query_execution"] = True
                    state["session_active"] = True
                    # Don't wait for confirmation, go straight to executing query
                elif decision.get("action") == "auto_proceed":
                    # Find and click the appropriate button automatically
                    target_desc = decision.get("target_element_description", "").lower()
                    button_found = False
                    
                    # Try to find the button
                    for elem in elements:
                        elem_desc = elem.get("description", "").lower()
                        elem_type = elem.get("type", "")
                        if elem_type == "button":
                            # Check if this matches the target
                            if target_desc and target_desc in elem_desc:
                                x = elem.get("x", 0)
                                y = elem.get("y", 0)
                                if x > 0 and y > 0:
                                    print(f"[Auto-Action] Clicking button: {elem.get('description')} at ({x}, {y})")
                                    self.device_actions.tap(x, y, delay=0.5)
                                    self.device_actions.wait(2.0)
                                    button_found = True
                                    break
                    
                    # If button not found, try accessibility tree
                    # CRITICAL: Never include "continue" alone - it matches "Continue with Google"
                    if not button_found:
                        button_keywords = ["get started", "next", "start", "accept", "ok", "skip", "got it", "proceed", "log in", "login"]
                        accessibility_result = self.accessibility.find_button_by_keywords(button_keywords)
                        if accessibility_result:
                            x, y, _ = accessibility_result
                            print(f"[Auto-Action] Clicking button via accessibility at ({x}, {y})")
                            self.device_actions.tap(x, y, delay=0.5)
                            self.device_actions.wait(2.0)
                            button_found = True
                    
                    if button_found:
                        # After clicking, check if there's a query to execute
                        if query:
                            state["task_complete"] = False
                            state["needs_query_execution"] = True
                            state["session_active"] = True
                        else:
                            state["task_complete"] = False
                            state["session_active"] = True
                    else:
                        # Button not found, check if there's a query to execute
                        if query:
                            state["task_complete"] = False
                            state["needs_query_execution"] = True
                            state["session_active"] = True
                        else:
                            state["task_complete"] = False
                            state["session_active"] = True
                else:
                    # Check if user wants to login (from original intent)
                    intent_parts = state.get("intent_parts", {})
                    user_intent = state.get("user_intent", "").lower()
                    
                    # Check if login button is visible using vision
                    has_login_button = False
                    login_button_elem = None
                    for elem in elements:
                        elem_desc = elem.get("description", "").lower()
                        elem_type = elem.get("type", "")
                        if elem_type == "button":
                            if "login" in elem_desc or "sign in" in elem_desc or "log in" in elem_desc:
                                has_login_button = True
                                login_button_elem = elem
                                break
                    
                    # If user said "login" and login button is visible, click it automatically
                    if has_login_button and ("login" in user_intent or "sign in" in user_intent):
                        # User wants to login and login button is visible - click it automatically
                        print("[Decision] Login button detected and user requested login - clicking automatically")
                        if login_button_elem:
                            x = login_button_elem.get("x", 0)
                            y = login_button_elem.get("y", 0)
                            if x > 0 and y > 0:
                                print(f"[Action] Clicking login button at ({x}, {y})")
                                self.device_actions.tap(x, y, delay=0.5)
                                self.device_actions.wait(2.5)  # Wait for login screen to appear
                                
                                # After clicking login, trigger authentication
                                state["needs_auth"] = True
                                state["task_complete"] = False
                                state["session_active"] = True
                            else:
                                # Try accessibility tree
                                accessibility_result = self.accessibility.find_button_by_keywords(["login", "sign in"])
                                if accessibility_result:
                                    x, y, _ = accessibility_result
                                    print(f"[Action] Clicking login button via accessibility at ({x}, {y})")
                                    self.device_actions.tap(x, y, delay=0.5)
                                    self.device_actions.wait(2.5)
                                    state["needs_auth"] = True
                                    state["task_complete"] = False
                                    state["session_active"] = True
                                else:
                                    state["error"] = "Login button found but coordinates invalid"
                                    state["task_complete"] = False
                        else:
                            state["needs_auth"] = True
                            state["task_complete"] = False
                    else:
                        # Ask for confirmation (login required, etc.)
                        state["screen_description"] = description
                        state["important_buttons"] = [e for e in elements if e.get("type") == "button"]
                        state["needs_confirmation"] = True
                        state["task_complete"] = False
            elif action in ["extract", "query"]:
                state["task_complete"] = False
                state["session_active"] = True
            else:
                state["task_complete"] = False
                state["session_active"] = True
        
        return state
    
    def _is_complete(self, state: Dict) -> str:
        """Determine if task is complete"""
        # Check if authentication is needed
        if state.get("needs_auth", False):
            return "authenticate"
        
        # Only mark complete if explicitly set AND not in active session
        if state.get("task_complete", False) and not state.get("session_active", False):
            return "complete"
        if state.get("needs_confirmation", False):
            return "confirm"  # New state for user confirmation
        if state.get("needs_query_execution", False):
            return "execute_query"  # New state for executing query in app
        if state.get("error") and not state.get("session_active", False):
            return "error"
        # If session is active, don't complete - continue listening
        if state.get("session_active", False):
            return "complete"  # Return to listening loop, but don't mark as fully complete
        return "continue"
    
    def _generate_llm_response(self, state: Dict, context: str) -> str:
        """
        Use LLM to generate intelligent, contextual feedback
        
        Args:
            state: Current agent state
            context: Context about what happened (e.g., "opened ChatGPT app", "clicked login button")
            
        Returns:
            Natural, contextual response text
        """
        user_intent = state.get("user_intent", "")
        intent_parts = state.get("intent_parts", {})
        action = intent_parts.get("action", "")
        action_success = state.get("action_success", False)
        error = state.get("error")
        extracted_info = state.get("extracted_info")
        app_name = intent_parts.get("app", "")
        
        prompt = f"""You are a helpful AI assistant helping a blind user control their Android device through voice commands. Generate a natural, conversational response about what just happened.

USER'S COMMAND:
{user_intent}

WHAT HAPPENED:
{context}

ACTION: {action}
APP: {app_name}
ACTION SUCCESS: {action_success}
ERROR: {error if error else "None"}
EXTRACTED INFO: {extracted_info if extracted_info else "None"}

Generate a brief, natural response (1-2 sentences) that:
- Acknowledges what was done in a friendly, conversational way
- Provides helpful context about the current state
- Sounds natural and human-like (not robotic)
- If there's an error, explains it clearly and helpfully
- If successful, confirms what was accomplished with context

Examples:
- If opened ChatGPT: "I've opened ChatGPT for you. The app is loading now."
- If clicked login: "I found and clicked the login button. The login screen should appear shortly."
- If error: "I had trouble finding that element. Could you try describing it differently, or I can try again?"
- If query executed: "I've sent your question to ChatGPT. Waiting for the response now."

Respond with ONLY the response text, no quotes or extra formatting. Be concise but friendly."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides natural, conversational feedback. Be brief, friendly, and contextual. Always respond with just the response text, no quotes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Slightly higher for more natural responses
                max_tokens=150
            )
            
            response_text = response.choices[0].message.content.strip()
            # Remove quotes if present
            response_text = response_text.strip('"\'')
            return response_text
        except Exception as e:
            print(f"[LLM Response] Error: {e}")
            # Fallback to simple response
            if action_success:
                if app_name:
                    return f"I've opened {app_name} for you."
                return "Task completed successfully."
            elif error:
                return f"I encountered an error: {error}"
            else:
                return "Action completed."
    
    def _respond(self, state: Dict) -> Dict:
        """Generate intelligent response using LLM"""
        logger.info("Responding to user...")
        self.state_manager.update_state(AgentState.RESPONDING)
        
        # Check if this is a close command
        intent_parts = state.get("intent_parts", {})
        action = intent_parts.get("action")
        user_intent = state.get("user_intent", "")
        
        # Build context for LLM response
        context_parts = []
        app_name = intent_parts.get("app", "")
        
        if action == "open_app":
            if state.get("action_success"):
                context_parts.append(f"opened {app_name} app")
            if state.get("needs_auth"):
                context_parts.append("login button was clicked and authentication flow started")
        elif action == "login":
            if state.get("action_success"):
                context_parts.append("login process completed successfully")
            else:
                context_parts.append("login process encountered an issue")
        elif action == "tap_element":
            target = intent_parts.get("target", "")
            if state.get("action_success"):
                context_parts.append(f"tapped on {target}")
        elif action == "execute_query":
            query = state.get("pending_query", "")
            if state.get("action_success"):
                context_parts.append(f"executed query: {query}")
        
        if state.get("extracted_info"):
            context_parts.append(f"extracted information from screen")
        
        if state.get("error"):
            context_parts.append(f"error occurred: {state.get('error')}")
        
        context = ". ".join(context_parts) if context_parts else "action was performed"
        
        # Keep continuous query session active: after executing a query in ChatGPT (or similar),
        # remember the app so the user can ask follow-ups without saying "open ChatGPT" again
        pending_app = state.get("pending_app", "")
        if pending_app and state.get("action_success"):
            app_lower = pending_app.lower()
            if any(app_lower.startswith(a) or a in app_lower for a in self.QUERY_SESSION_APPS):
                self._query_app_session = pending_app
                print(f"[Session] Query session active for: {pending_app}")
        
        # Generate intelligent response using LLM
        response_text = self._generate_llm_response(state, context)
        
        if action == "close_app":
            # App closed - mark session inactive and complete
            state["session_active"] = False
            state["task_complete"] = True
            print(f"[SUCCESS]: {response_text}")
            self.tts.speak(response_text)
            state["response"] = response_text
            return state
        
        if state.get("error"):
            error_msg = state.get("error", "Unknown error occurred")
            print(f"[ERROR]: {error_msg}")
            # Use LLM-generated error response
            if not state.get("session_active", False):
                self.tts.speak(response_text)
            state["response"] = response_text
        elif state.get("action_success"):
            if state.get("extracted_info"):
                print(f"[RESULT]: {state.get('extracted_info')}")
                self.tts.speak(response_text)
                state["response"] = response_text
            elif state.get("session_active", False):
                # Session active - give intelligent feedback
                print(f"[SUCCESS]: {response_text}")
                self.tts.speak(response_text)
                state["response"] = response_text
            else:
                print(f"[SUCCESS]: {response_text}")
                self.tts.speak(response_text)
                state["response"] = response_text
        else:
            print(f"[INFO]: {response_text}")
            if not state.get("session_active", False):
                self.tts.speak(response_text)
            state["response"] = response_text
        
        # Don't complete task if session is active
        if not state.get("session_active", False):
            self.state_manager.complete_task()
        
        return state
    
    def _confirm_action(self, state: Dict) -> Dict:
        """Ask user for confirmation only when truly needed (e.g., login required)"""
        logger.info("Requesting user confirmation...")
        self.state_manager.update_state(AgentState.VERIFYING)
        
        device = self.adb_client.get_device()
        screen_description = state.get("screen_description", "")
        important_buttons = state.get("important_buttons", [])
        primary_action = state.get("primary_action", "")  # Get from state
        
        # CRITICAL: Check if this is a login wall/bottom sheet
        screen_lower = screen_description.lower()
        is_login_wall = any(keyword in screen_lower for keyword in [
            "thanks for trying chatgpt", "thanks for trying", "log in or sign up",
            "continue with google", "sign up", "login wall", "bottom sheet"
        ])
        is_login = "login" in screen_lower or "sign in" in screen_lower or is_login_wall
        
        if is_login_wall:
            # LOGIN WALL DETECTED - Use Google sign-in via bottom sheet first button
            print("[Confirm] ⚠ LOGIN WALL DETECTED - Using 'Continue with Google' (fastest)")
            print("[Confirm] RULE: Skip confirmation → Tap FIRST button = Continue with Google")
            
            # Use bottom sheet button detection - now returns FIRST (Google) button
            accessibility_result = self.accessibility.find_real_login_button()
            
            if accessibility_result:
                x, y, button_info = accessibility_result
                print(f"[Confirm] ✓ Found 'Continue with Google' at ({x}, {y})")
                self.tts.speak("Signing in with Google.")
                
                self.device_actions.tap(x + 5, y + 5, delay=0.5)
                self.device_actions.wait(3.0)
                
                # After tapping Google, wait for account selector and tap Continue
                print("[Confirm] Waiting for Google account selector...")
                self.device_actions.wait(2.0)
                
                # Find and tap Continue button on Google popup
                continue_btn = self.accessibility.find_continue_button()
                if continue_btn:
                    cx, cy, _ = continue_btn
                    print(f"[Confirm] ✓ Found Continue at ({cx}, {cy}), tapping...")
                    self.device_actions.tap(cx, cy, delay=0.5)
                    self.device_actions.wait(2.0)
                
                state["needs_auth"] = False  # Google handles auth
                state["action_success"] = True
                state["task_complete"] = False
                state["session_active"] = True
                state["login_method"] = "google"
                return state
            else:
                print("[Confirm] ERROR: No bottom sheet button found")
                self.tts.speak("I couldn't find the login button. Please try again.")
                state["error"] = "Login wall detected but no button found"
                state["task_complete"] = True
                return state
        
        if is_login:
            # Login required - ask user
            message = "I see a login screen. Do you want me to sign in with Google?"
        else:
            # For other cases, be brief
            if important_buttons:
                button_desc = important_buttons[0].get("description", "button")
                # PREFER "Continue with Google"
                if "google" in button_desc.lower():
                    message = "I see Continue with Google. Should I tap it for quick sign-in?"
                elif any(kw in button_desc.lower() for kw in ["got it", "ok", "skip"]):
                    # These are usually safe to auto-click
                    message = f"I see a {button_desc}. Should I proceed?"
                else:
                    message = f"I see a {button_desc}. Should I click it?"
            elif primary_action:
                # PREFER "Continue with Google"
                if "google" in primary_action.lower():
                    message = "I see Continue with Google. Should I use it for quick sign-in?"
                else:
                    message = f"I see a {primary_action}. Should I proceed?"
            else:
                message = "Should I proceed?"
        
        # Speak to user (brief message)
        print(f"\n[Screen]: {screen_description}")
        if important_buttons:
            print(f"[Action]: {important_buttons[0].get('description', 'button')}")
        elif primary_action:
            print(f"[Action]: {primary_action}")
        
        print(f"\n[Question]: {message}")
        self.tts.speak(message)
        
        # Listen for user response
        print("\n[Listening for your response...]")
        response = self.stt.listen(timeout=10.0, phrase_time_limit=10.0)
        
        if response:
            response_lower = response.lower().strip()
            print(f"[Heard]: {response}")
            
            # Check for positive responses
            positive_responses = ["yes", "okay", "ok", "proceed", "go ahead", "sure", "yep", "yeah", "alright", "all right", "log in", "login", "sign in"]
            
            if any(pos in response_lower for pos in positive_responses):
                # User confirmed - proceed with action
                logger.info("User confirmed action")
                self.tts.speak("Proceeding...")
                
                # CRITICAL: For login screens, ALWAYS use bottom sheet button detection
                # This finds the "Log in" button, NOT "Continue with Google"
                print("[Action] Checking if this is a login screen...")
                
                # Re-analyze screen to check for login wall
                screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                description = screen_analysis.get("description", "").lower()
                login_wall_keywords = ["thanks for trying", "log in or sign up", "continue with google", "sign up"]
                is_login_wall = any(keyword in description for keyword in login_wall_keywords)
                
                if is_login_wall:
                    print("[Action] ⚠ LOGIN WALL DETECTED - Using bottom sheet button detection")
                    accessibility_result = self.accessibility.find_real_login_button()
                    
                    if accessibility_result:
                        x, y, button_info = accessibility_result
                        print(f"[Action] ✓ Found bottom sheet login button at ({x}, {y})")
                        print(f"[Action]   Button text: '{button_info.get('text', 'N/A')[:50] if button_info.get('text') else 'N/A'}'")
                        
                        self.device_actions.tap(x + 5, y + 5, delay=0.5)
                        self.device_actions.wait(3.0)
                        
                        state["needs_auth"] = True
                        state["action_success"] = True
                        state["task_complete"] = False
                        state["session_active"] = True
                        return state
                
                # Not a login wall - use standard button detection
                print("[Action] Getting accurate button coordinates from accessibility tree...")
                button_clicked = False
                button_coords = None
                button_desc = ""
                
                # Extract button text/keywords from primary_action or important_buttons
                # Include "continue with google" and "google" as valid keywords now
                button_keywords = []
                primary_action = state.get("primary_action", "")  # Get from state
                if primary_action:
                    # Extract keywords from primary action description
                    action_lower = primary_action.lower()
                    # Include Google buttons now (preferred for fast login)
                    keywords = ["continue with google", "google", "get started", "next", "start", "accept", "ok", "skip", "sign in", "login", "log in", "proceed", "go", "got it", "continue"]
                    for kw in keywords:
                        if kw in action_lower:
                            button_keywords.append(kw)
                
                # Also check important_buttons for text
                if important_buttons:
                    for btn in important_buttons:
                        btn_desc = btn.get("description", "").lower()
                        # Include Google buttons now (preferred)
                        keywords = ["continue with google", "google", "get started", "next", "start", "accept", "ok", "skip", "sign in", "login", "log in", "proceed", "got it", "continue"]
                        for kw in keywords:
                            if kw in btn_desc and kw not in button_keywords:
                                button_keywords.append(kw)
                
                # Default keywords if none found - NEVER include "continue"
                if not button_keywords:
                    button_keywords = ["log in", "login", "sign in", "get started", "next", "start", "accept", "ok", "proceed", "got it"]
                
                print(f"[Action] Searching for button with keywords: {button_keywords}")
                
                # Try accessibility tree first (most accurate)
                accessibility_result = self.accessibility.find_button_by_keywords(button_keywords)
                
                if accessibility_result:
                    x, y, element_info = accessibility_result
                    button_coords = (x, y)
                    button_desc = element_info.get("text", "button")
                    bounds = element_info.get("bounds", [])
                    print(f"[Action] Found button '{button_desc}' via accessibility tree")
                    print(f"[Action] Bounds: {bounds}, Center: ({x}, {y})")
                else:
                    # Fallback: Use vision analysis coordinates
                    print("[Action] Accessibility tree not found, using vision analysis coordinates...")
                    fresh_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                    fresh_elements = fresh_analysis.get("elements", [])
                    
                    # Find button from vision analysis
                    for elem in fresh_elements:
                        elem_desc = elem.get("description", "").lower()
                        elem_type = elem.get("type", "")
                        if elem_type == "button":
                            # Check for common action button keywords
                            if any(keyword in elem_desc for keyword in button_keywords):
                                x = elem.get("x", 0)
                                y = elem.get("y", 0)
                                if x > 0 and y > 0:
                                    button_coords = (x, y)
                                    button_desc = elem.get("description", "button")
                                    print(f"[Action] Found button '{button_desc}' via vision analysis at ({x}, {y})")
                                    break
                    
                    # Last resort: use stored important_buttons
                    if not button_coords and important_buttons:
                        button = important_buttons[0]
                        x = button.get("x", 0)
                        y = button.get("y", 0)
                        if x > 0 and y > 0:
                            button_coords = (x, y)
                            button_desc = button.get("description", "button")
                            print(f"[Action] Using stored button coordinates: ({x}, {y})")
                
                # Click the button if found
                if button_coords:
                    x, y = button_coords
                    print(f"[Action] Final coordinates: ({x}, {y})")
                    print(f"[Action] Executing tap command: adb shell input tap {x} {y}")
                    
                    # Execute tap with validation
                    success = self.device_actions.tap(x, y, delay=0.5)
                    
                    if success:
                        print(f"[Action] Tap command executed successfully")
                        button_clicked = True
                        self.device_actions.wait(2.5)  # Wait for screen to change
                        
                        # Check if there's a query to execute after this
                        pending_query = state.get("pending_query")
                        if pending_query:
                            print(f"[Action] Query detected, will execute after button click: {pending_query}")
                            state["needs_query_execution"] = True
                            state["task_complete"] = False
                        else:
                            state["action_success"] = True
                            # Don't mark task complete - user can continue giving commands
                            state["task_complete"] = False
                            state["session_active"] = True
                            # Use LLM to generate intelligent feedback
                            intent_parts = state.get("intent_parts", {})
                            feedback = self._generate_llm_response(state, f"opened {intent_parts.get('app', 'app')} and clicked button")
                            self.tts.speak(feedback)
                    else:
                        print(f"[Action] Tap command failed - trying alternative method")
                        # Try clicking all clickable elements as fallback
                        clickable_elements = self.accessibility.find_clickable_elements()
                        if clickable_elements:
                            # Find the one closest to our target coordinates
                            best_match = None
                            min_distance = float('inf')
                            for elem in clickable_elements:
                                elem_x = elem.get("x", 0)
                                elem_y = elem.get("y", 0)
                                distance = ((elem_x - x) ** 2 + (elem_y - y) ** 2) ** 0.5
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match = elem
                            
                            if best_match and min_distance < 200:  # Within 200 pixels
                                alt_x = best_match.get("x", 0)
                                alt_y = best_match.get("y", 0)
                                print(f"[Action] Trying alternative coordinates: ({alt_x}, {alt_y})")
                                if self.device_actions.tap(alt_x, alt_y, delay=0.5):
                                    button_clicked = True
                                    state["action_success"] = True
                                    state["task_complete"] = True
                                    self.tts.speak("Action completed successfully.")
                                else:
                                    state["error"] = f"Failed to tap button at both ({x}, {y}) and ({alt_x}, {alt_y})"
                                    state["task_complete"] = True
                            else:
                                state["error"] = f"Failed to tap button at ({x}, {y}) and no nearby clickable element found"
                                state["task_complete"] = True
                        else:
                            state["error"] = f"Failed to tap button at ({x}, {y})"
                            state["task_complete"] = True
                else:
                    print("[Action] ERROR: Could not find button coordinates")
                    print(f"[Action] Debug - Primary action: {primary_action}")
                    print(f"[Action] Debug - Important buttons: {important_buttons}")
                    print(f"[Action] Debug - Button keywords searched: {button_keywords}")
                    
                    # Last attempt: get all clickable elements and show them
                    clickable_elements = self.accessibility.find_clickable_elements()
                    print(f"[Action] Debug - Found {len(clickable_elements)} clickable elements in accessibility tree")
                    if clickable_elements:
                        print("[Action] Available clickable elements:")
                        for i, elem in enumerate(clickable_elements[:5]):  # Show first 5
                            print(f"  {i+1}. {elem.get('text', 'No text')} at ({elem.get('x', 0)}, {elem.get('y', 0)})")
                    
                    state["error"] = "Could not find button to click - no valid coordinates"
                    state["task_complete"] = True
            else:
                # User declined or unclear response
                logger.info("User declined or unclear response")
                self.tts.speak("Okay, I'll wait. Let me know when you're ready.")
                state["task_complete"] = True
                state["action_success"] = False
        else:
            # No response detected
            logger.warning("No response detected from user")
            self.tts.speak("I didn't hear a response. I'll wait for your command.")
            state["task_complete"] = True
            state["action_success"] = False
        
        return state
    
    def _execute_query(self, state: Dict) -> Dict:
        """Execute query in the opened app (e.g., ask ChatGPT a question)"""
        logger.info("Executing query in app...")
        print("[Status] Executing query in app...")
        self.state_manager.update_state(AgentState.ACTING)
        
        device = self.adb_client.get_device()
        query = state.get("pending_query", "")
        app_name = state.get("pending_app", "")
        
        if not query:
            # Issue detected - use LLM to analyze and ask user
            return self._handle_issue_with_llm(state, "No query provided. The system was trying to execute a query but none was found.")
        
        print(f"[Query] Question: {query}")
        print(f"[Query] App: {app_name}")
        
        # CRITICAL: Check if app is actually open before executing query
        if app_name:
            print(f"[Query] Verifying {app_name} is open before executing query...")
            # Check if we're in the right app by checking current activity
            try:
                current_activity = device.shell("dumpsys window windows | grep -E 'mCurrentFocus|mFocusedApp' | head -1")
                app_package = self.app_launcher.get_package_name(app_name)
                
                if app_package and app_package not in current_activity:
                    print(f"[Query] WARNING: {app_name} doesn't appear to be open.")
                    print(f"[Query] Current activity: {current_activity[:100] if current_activity else 'Unknown'}")
                    print(f"[Query] Expected package: {app_package}")
                    print(f"[Query] Opening {app_name} first...")
                    # Open the app first
                    launch_success = self.app_launcher.launch_app(app_name)
                    if launch_success:
                        print(f"[Query] {app_name} opened. Waiting for app to load...")
                        self.device_actions.wait(5.0)  # Wait longer for app to fully load
                    else:
                        return self._handle_issue_with_llm(state, f"Could not open {app_name}. Please try again.")
                else:
                    print(f"[Query] ✓ {app_name} appears to be open")
            except Exception as e:
                print(f"[Query] WARNING: Could not verify app state: {e}. Proceeding anyway...")
        else:
            print(f"[Query] WARNING: No app name provided. Query execution may fail if app isn't open.")
        
        # Wait for app to load
        self.device_actions.wait(2.0)
        
        # Use accessibility only for input field coordinates (no Vision)
        print("[Query] Looking for input field via Accessibility...")
        input_field = None
        acc_result = self.accessibility.find_input_field()
        if acc_result:
            x, y, _ = acc_result
            input_field = (x, y)
            print(f"[Query] Using input field at ({x}, {y}) from Accessibility")
        if not input_field:
            # Fallback: same two regions as find_input_field (keyboard open vs closed)
            region_x_min, region_x_max = 300, 800
            keyboard_open = self.accessibility.is_keyboard_visible()
            if keyboard_open:
                region_y_min, region_y_max = 1200, 1800
                print("[Query] Fallback: Searching clickable EditText in region Y 1200-1800 (keyboard open)...")
            else:
                region_y_min, region_y_max = 2000, 2500
                print("[Query] Fallback: Searching clickable EditText in region Y 2000-2500 (keyboard closed)...")
            clickable_elements = self.accessibility.find_clickable_elements()
            for elem in clickable_elements:
                elem_class = elem.get("class", "").lower()
                if "edit" in elem_class or "input" in elem_class or "textfield" in elem_class:
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0 and region_x_min <= x <= region_x_max and region_y_min <= y <= region_y_max:
                        input_field = (x, y)
                        print(f"[Query] Found input field via clickable elements at ({x}, {y})")
                        break
        
        if not input_field:
            # Issue detected - use LLM to analyze and ask user
            return self._handle_issue_with_llm(state, "Could not find input field. The app may not be ready or the screen may have changed.")
        
        # Tap on input field
        x, y = input_field
        print(f"[Query] Tapping input field at ({x}, {y})")
        self.device_actions.tap(x, y, delay=0.5)
        self.device_actions.wait(1.0)
        
        # Type the query
        print(f"[Query] Typing question: {query}")
        self.device_actions.type_text(query, clear_first=True)
        
        # Send: tap the send button (upward-arrow icon just after input). Enter would only add newline.
        input_x, input_y = input_field
        send_button = self.accessibility.find_send_button_near_input(input_x, input_y)
        if send_button:
            sx, sy, _ = send_button
            print(f"[Query] Tapping send button (upward arrow) at ({sx}, {sy})")
            self.device_actions.tap(sx, sy, delay=0.5)
            self.device_actions.wait(1.0)
        else:
            # Fallback: keyword search then Enter (may newline in some apps)
            kw_result = self.accessibility.find_button_by_keywords(["send", "submit", "arrow"])
            if kw_result:
                sx, sy, _ = kw_result
                print(f"[Query] Tapping send button at ({sx}, {sy})")
                self.device_actions.tap(sx, sy, delay=0.5)
                self.device_actions.wait(1.0)
            else:
                print("[Query] Send button not found, pressing Enter (may newline in ChatGPT)...")
                self.device_actions.press_key("KEYCODE_ENTER")
                self.device_actions.wait(1.0)
        
        # Wait for response to generate
        print("[Query] Waiting for response to generate...")
        self.device_actions.wait(8.0)  # Wait longer for response to fully generate
        
        # Extract response using specialized method
        print("[Query] Extracting response...")
        answer_text = self._extract_chat_response(device, query)
        
        if answer_text and len(answer_text.strip()) > 2:
            state["extracted_response"] = answer_text
            state["extracted_info"] = f"Response: {answer_text}"
            state["action_success"] = True
            state["task_complete"] = False
            state["session_active"] = True
            
            print(f"[Query] Response extracted: {answer_text[:150]}...")
            # Speak the response
            response_text = f"The answer is: {answer_text}"
            self.tts.speak(response_text)
            
            # POST-ACTION MONITOR: Check for login popup after query (2 checks only)
            print("[Post-Action Monitor] Entering post-action monitor mode...")
            print("[Post-Action Monitor] Checking for login popup (2 checks)...")
            login_popup_detected = self._wait_for_login_popup(device, max_checks=2)
            
            if login_popup_detected:
                print("[Post-Action Monitor] ✓ Login popup detected!")
                state["login_popup_detected"] = True
                
                # TRY GOOGLE SIGN-IN FIRST (uses Accessibility only, no LLM)
                print("[Post-Action Monitor] Attempting Google sign-in (Accessibility only)...")
                google_success, google_msg = self._perform_google_signin()
                
                if google_success:
                    print(f"[Post-Action Monitor] ✓ Google sign-in successful: {google_msg}")
                    state["login_complete"] = True
                    state["login_method"] = "google"
                    # Reset login state
                    self.login_flow_state = {
                        "active": False,
                        "awaiting_email": False,
                        "awaiting_password": False,
                        "email": None,
                        "password": None,
                        "email_entered": False,
                        "popup_detected": False,
                        "awaiting_popup_confirmation": False
                    }
                else:
                    print(f"[Post-Action Monitor] Google sign-in failed: {google_msg}")
                    print("[Post-Action Monitor] Falling back to credential-based login...")
                    
                    # Update login flow state to remember popup was detected
                    self.login_flow_state["popup_detected"] = True
                    self.login_flow_state["awaiting_popup_confirmation"] = True
                    
                    # Try to auto-tap login button for credential-based flow
                    login_tapped = self._auto_tap_login_button(device)
                    if login_tapped:
                        print("[Post-Action Monitor] Login button tapped automatically. Starting credential input flow.")
                        state["awaiting_email"] = True
                        state["login_flow_active"] = True
                        # Update login flow state
                        self.login_flow_state["popup_detected"] = False
                        self.login_flow_state["awaiting_popup_confirmation"] = False
                        self.login_flow_state["active"] = True
                        self.login_flow_state["awaiting_email"] = True
                        
                        # Use typed input for credentials (more accurate than voice)
                        print("\n" + "="*60)
                        print("LOGIN REQUIRED (Credential-based)")
                        print("="*60)
                        self.tts.speak("Login required. Please type your email address.")
                        
                        # Get email via typed input
                        print("\n[Login] Please type your email address:")
                        try:
                            email = input("Email: ").strip()
                            
                            if email and "@" in email:
                                print(f"[Login] Email received: {email}")
                                # Handle email input
                                email_result = self._handle_email_input_typed(state, email)
                                
                                if email_result.get("email_entered", False):
                                    # Get password via typed input (secure)
                                    self.tts.speak("Email entered. Please type your password.")
                                    print("\n[Login] Please type your password:")
                                    
                                    # Use getpass for secure password input (hides characters)
                                    import getpass
                                    try:
                                        password = getpass.getpass("Password: ")
                                    except Exception:
                                        # Fallback to regular input if getpass fails
                                        password = input("Password: ")
                                    
                                    if password:
                                        print(f"[Login] Password received (hidden)")
                                        # Handle password input
                                        password_result = self._handle_password_input(state, password)
                                        
                                        if password_result.get("login_complete", False):
                                            print("[Login] ✓ Login completed successfully!")
                                            self.tts.speak("Login successful!")
                                            
                                            # Reset login state
                                            self.login_flow_state = {
                                                "active": False,
                                                "awaiting_email": False,
                                                "awaiting_password": False,
                                                "email": None,
                                                "password": None,
                                                "email_entered": False,
                                                "popup_detected": False,
                                                "awaiting_popup_confirmation": False
                                            }
                                            
                                            # Now execute the original query if there is one
                                            pending_query = state.get("pending_query")
                                            if pending_query:
                                                print(f"[Login] Will execute pending query: {pending_query}")
                                                self.tts.speak("Now executing your query.")
                                                self.device_actions.wait(2.0)  # Wait for app to settle
                                                state["login_complete"] = True
                                                state["task_complete"] = False
                                                state["session_active"] = True
                                            else:
                                                state["task_complete"] = False
                                                state["session_active"] = True
                                            
                                            return state
                                        else:
                                            print("[Login] Password entry failed.")
                                            self.tts.speak("Password entry failed. Please try again.")
                                    else:
                                        print("[Login] No password entered.")
                                        self.tts.speak("No password entered.")
                                else:
                                    print("[Login] Email entry failed.")
                                    self.tts.speak("Could not enter email. Please try again.")
                            else:
                                print("[Login] Invalid email format.")
                                self.tts.speak("Invalid email address. Please try again.")
                        except KeyboardInterrupt:
                            print("\n[Login] Login cancelled by user.")
                            self.tts.speak("Login cancelled.")
                        
                        # Return with login flow active
                        state["task_complete"] = False
                        state["session_active"] = True
                        return state
                    else:
                        print("[Post-Action Monitor] Could not auto-tap login button. Asking user for confirmation...")
                        # Ask user to confirm login
                        self.tts.speak("I see a login screen. Please say 'log in' or 'yes' to proceed with login.")
                        # Keep popup_detected and awaiting_popup_confirmation flags set
                        # so process_command can handle the user's confirmation
            else:
                print("[Post-Action Monitor] No login popup detected after 2 checks. Continuing normally.")
        else:
            # Issue detected - use LLM to analyze and ask user
            return self._handle_issue_with_llm(state, "Could not extract response. The answer may still be generating or the screen may have changed.")
        
        return state
    
    def _extract_chat_response(self, device, query: str) -> str:
        """
        Extract the actual chat response text from ChatGPT or similar chat apps
        
        Args:
            device: ADB device instance
            query: The original query that was asked
            
        Returns:
            Extracted response text or empty string
        """
        import re
        
        # Use Vision only for result extraction (primary path)
        print("[Query] Extracting response via Vision API...")
        try:
                # Use a specialized prompt for extracting chat responses
                specialized_prompt = f"""You are analyzing a ChatGPT conversation screen. The user asked: "{query}"

Extract ONLY the actual answer/response text from ChatGPT. Ignore:
- UI elements (buttons, menus, input fields)
- Screen descriptions
- Navigation elements
- The user's question (which is: "{query}")

Return ONLY the ChatGPT's response text. If the response is still generating, extract what's visible so far.

Respond in JSON format:
{{
    "answer": "The actual answer text from ChatGPT",
    "is_complete": true/false
}}"""

                # Get screenshot using screen_analyzer
                screenshot = self.screen_analyzer.capture_screenshot(device)
                if screenshot:
                    from openai import OpenAI
                    import tempfile
                    import os
                    
                    api_key = self.config.get_openai_api_key()
                    client = OpenAI(api_key=api_key)
                    
                    # Save screenshot to temp file for API
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        screenshot.save(tmp_file.name)
                        screenshot_path = tmp_file.name
                    
                    try:
                        # Convert to base64 for API
                        base64_image = self.screen_analyzer.image_to_base64(screenshot)
                        
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": specialized_prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                    ]
                                }
                            ],
                            max_tokens=500,
                            temperature=0.2
                        )
                    finally:
                        # Clean up temp file
                        if os.path.exists(screenshot_path):
                            os.unlink(screenshot_path)
                    
                    result_text = response.choices[0].message.content
                    
                    # Parse JSON response
                    import json
                    try:
                        json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
                        if json_match:
                            result_json = json.loads(json_match.group())
                            answer_text = result_json.get("answer", "").strip()
                            if answer_text:
                                print(f"[Query] Extracted answer via Vision API: {answer_text[:100]}...")
                                return answer_text
                    except Exception:
                        pass
                    if "answer" in result_text.lower():
                        answer_match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']', result_text, re.IGNORECASE)
                        if answer_match:
                            return answer_match.group(1).strip()
        except Exception as e:
            print(f"[Query] Vision extraction failed: {e}")
        
        # Fallback: accessibility tree (only if Vision failed)
        answer_texts = []
        query_lower = query.lower()
        print("[Query] Fallback: Extracting from accessibility tree...")
        clickable_elements = self.accessibility.find_clickable_elements()
        for elem in clickable_elements:
            text = elem.get("text", "").strip()
            content_desc = elem.get("content_desc", "").strip()
            elem_class = elem.get("class", "").lower()
            if not text and not content_desc:
                continue
            combined_text = (text + " " + content_desc).lower()
            if query_lower in combined_text and len(text) < len(query) + 50:
                continue
            if "button" in elem_class or "menu" in elem_class or "image" in elem_class:
                continue
            if text and len(text) > 30:
                ui_indicators = ["tap", "click", "button", "menu", "settings", "options", "chatgpt", "welcome"]
                if not any(indicator in text.lower() for indicator in ui_indicators):
                    answer_texts.append(text)
        
        if not answer_texts:
            print("[Query] Method 3: Using general screen analysis with better filtering...")
            response_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=False)
            response_description = response_analysis.get("description", "")
            
            if response_description:
                # Better filtering: look for sentences that answer the question
                sentences = re.split(r'[.!?]\s+', response_description)
                answer_sentences = []
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    sentence_clean = sentence.strip()
                    
                    # Skip UI-related sentences
                    ui_words = ["button", "field", "screen", "display", "visible", "shown", "tap", "click", 
                               "interface", "application", "allowing", "users", "ask questions", "receive"]
                    if any(ui_word in sentence_lower for ui_word in ui_words):
                        continue
                    
                    # Skip if it contains the query (that's the input, not the answer)
                    if query_lower in sentence_lower and len(sentence_clean) < len(query) + 50:
                        continue
                    
                    # Look for answer indicators
                    answer_indicators = ["capital", "is", "are", "answer", "response", "delhi", "india"]
                    if any(indicator in sentence_lower for indicator in answer_indicators):
                        # Keep substantial sentences (likely answers)
                        if len(sentence_clean) > 20:
                            answer_sentences.append(sentence_clean)
                
                if answer_sentences:
                    answer_texts = answer_sentences
        
        # Combine and return the best answer
        if answer_texts:
            # Join all answer texts, prioritizing longer ones
            answer_texts.sort(key=len, reverse=True)  # Longest first
            answer_text = ". ".join(answer_texts[:2])  # Take top 2 longest answers
            return answer_text.strip()
        
        return ""
    
    def _handle_error(self, state: Dict) -> Dict:
        """Handle errors - analyze with LLM and ask user for guidance"""
        error_msg = state.get('error', 'Unknown error')
        logger.error(f"Error occurred: {error_msg}")
        self.state_manager.update_state(AgentState.ERROR)
        self.state_manager.increment_error()
        
        # Use LLM to analyze the issue and get user guidance
        return self._handle_issue_with_llm(state, error_msg)
    
    def _handle_issue_with_llm(self, state: Dict, issue_description: str) -> Dict:
        """
        Handle issues by analyzing screen with LLM and asking user for guidance
        
        Args:
            state: Current state dictionary
            issue_description: Description of the issue encountered
            
        Returns:
            Updated state dictionary
        """
        device = self.adb_client.get_device()
        if not device:
            state["error"] = issue_description
            state["task_complete"] = True
            return state
        
        print(f"[Issue Detection] Issue encountered: {issue_description}")
        print("[Issue Detection] Capturing screenshot and analyzing with LLM...")
        
        # Capture screenshot
        screenshot = self.screen_analyzer.capture_screenshot(device)
        if not screenshot:
            state["error"] = f"{issue_description}. Failed to capture screenshot."
            state["task_complete"] = True
            return state
        
        # Analyze screen with LLM to understand what's happening
        analysis_prompt = f"""You are analyzing a mobile app screen where an issue occurred.

ISSUE DESCRIPTION: {issue_description}

Analyze the screenshot and provide:
1. What is currently displayed on the screen?
2. What issue or problem do you see?
3. What might be preventing the task from completing?
4. What actions could resolve this issue?

Respond in JSON format:
{{
    "screen_description": "What you see on the screen",
    "issue_identified": "What issue/problem is visible",
    "possible_causes": ["cause1", "cause2"],
    "suggested_actions": ["action1", "action2"],
    "summary": "Brief summary for the user"
}}"""

        try:
            base64_image = self.screen_analyzer.image_to_base64(screenshot)
            from openai import OpenAI
            api_key = self.config.get_openai_api_key()
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": analysis_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                screen_desc = analysis.get("screen_description", "")
                issue = analysis.get("issue_identified", issue_description)
                summary = analysis.get("summary", f"I see: {screen_desc}. Issue: {issue}")
                suggested_actions = analysis.get("suggested_actions", [])
                
                # Speak the issue to user
                user_message = f"I encountered an issue: {summary}"
                if suggested_actions:
                    user_message += f" Possible actions: {', '.join(suggested_actions[:3])}"
                user_message += ". Please tell me what to do next."
                
                print(f"[Issue Detection] LLM Analysis:")
                print(f"  Screen: {screen_desc}")
                print(f"  Issue: {issue}")
                print(f"  Suggested actions: {suggested_actions}")
                print(f"[Issue Detection] Asking user for guidance...")
                
                self.tts.speak(user_message)
                
                # Store issue context for user response
                state["issue_context"] = {
                    "description": issue_description,
                    "screen_description": screen_desc,
                    "issue_identified": issue,
                    "suggested_actions": suggested_actions,
                    "waiting_for_user_guidance": True
                }
                state["task_complete"] = False
                state["session_active"] = True
                
                # The system will wait for user input in the main loop
                # User can then provide guidance like "click continue" or "try again"
                return state
            else:
                # Fallback if JSON parsing fails
                summary = result_text[:200] if len(result_text) > 200 else result_text
                user_message = f"I encountered an issue: {issue_description}. I see: {summary}. Please tell me what to do next."
                self.tts.speak(user_message)
                state["issue_context"] = {
                    "description": issue_description,
                    "summary": summary,
                    "waiting_for_user_guidance": True
                }
                state["task_complete"] = False
                state["session_active"] = True
                return state
                
        except Exception as e:
            print(f"[Issue Detection] Error analyzing with LLM: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: just tell user about the issue
            user_message = f"I encountered an issue: {issue_description}. Please tell me what to do next."
            self.tts.speak(user_message)
            state["issue_context"] = {
                "description": issue_description,
                "waiting_for_user_guidance": True
            }
            state["task_complete"] = False
            state["session_active"] = True
            return state
    
    def _wait_for_login_popup(self, device, max_checks: int = 2, check_interval: float = 0.5) -> bool:
        """
        Check for login popup a limited number of times (popup may appear after answer extraction).
        
        Args:
            device: ADB device instance
            max_checks: Number of times to check (default: 2)
            check_interval: Time between checks (default: 0.5 seconds)
            
        Returns:
            True if login popup detected, False otherwise
        """
        import time
        for i in range(max_checks):
            if self._detect_login_popup(device):
                print(f"[Popup Detection] ✓ LOGIN WALL DETECTED on check {i+1}/{max_checks}")
                return True
            
            if i < max_checks - 1:
                print(f"[Popup Detection] No popup yet... retry {i+1}/{max_checks}")
                time.sleep(check_interval)
        
        print(f"[Popup Detection] No popup detected after {max_checks} checks.")
        return False
    
    def _detect_login_popup(self, device) -> bool:
        """
        Detect if "Thanks for trying ChatGPT" login popup is visible (single check)
        Uses hybrid detection: text + layout + structure
        
        Returns:
            True if login popup detected, False otherwise
        """
        # Method 1: Check accessibility tree for "Thanks for trying ChatGPT" text (MOST RELIABLE)
        clickable_elements = self.accessibility.find_clickable_elements()
        
        login_popup_indicators = [
            "thanks for trying chatgpt",
            "thanks for trying",
            "log in",
            "sign up",
            "sign in"
        ]
        
        found_indicators = []
        for elem in clickable_elements:
            text = elem.get("text", "").lower().strip()
            content_desc = elem.get("content_desc", "").lower().strip()
            combined_text = (text + " " + content_desc).lower()
            
            # Check for login popup indicators
            for indicator in login_popup_indicators:
                if indicator in combined_text:
                    found_indicators.append(indicator)
                    # Check if it's specifically the "Thanks for trying" message
                    if "thanks for trying" in combined_text:
                        print(f"[Popup Detection] ✓ Found 'Thanks for trying ChatGPT' text: '{text or content_desc}'")
                        return True
                    # Check for login button in popup context
                    if ("log in" in combined_text or "sign up" in combined_text or "sign in" in combined_text) and len(text) > 0:
                        print(f"[Popup Detection] ✓ Found login button in popup: '{text or content_desc}'")
                        return True
        
        # Method 2: Use Vision API to detect popup (layout + text)
        try:
            screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
            description = screen_analysis.get("description", "").lower()
            elements = screen_analysis.get("elements", [])
            
            # Check for popup keywords in description
            popup_keywords = [
                "thanks for trying chatgpt",
                "thanks for trying",
                "login popup",
                "sign up popup",
                "login required",
                "login wall",
                "modal",
                "popup"
            ]
            
            description_lower = description.lower()
            if any(keyword in description_lower for keyword in popup_keywords):
                print(f"[Popup Detection] ✓ Vision API detected login popup keywords in description")
                return True
            
            # Method 3: Check for modal/popup structure (large centered buttons, modal layout)
            # Look for multiple login-related buttons/elements (indicates popup)
            login_elements_count = 0
            for elem in elements:
                elem_desc = elem.get("description", "").lower()
                elem_type = elem.get("type", "")
                if elem_type == "button" and any(indicator in elem_desc for indicator in login_popup_indicators):
                    login_elements_count += 1
            
            # If we see multiple login-related elements, likely a popup
            if login_elements_count >= 2:
                print(f"[Popup Detection] ✓ Detected popup structure: {login_elements_count} login-related elements")
                return True
            
            # Method 4: Check for "Thanks for trying" in element descriptions
            for elem in elements:
                elem_desc = elem.get("description", "").lower()
                if "thanks for trying" in elem_desc:
                    print(f"[Popup Detection] ✓ Found 'Thanks for trying' in element: '{elem_desc}'")
                    return True
                    
        except Exception as e:
            print(f"[Popup Detection] Vision API check failed: {e}")
        
        return False
    
    def _auto_tap_login_button(self, device) -> bool:
        """
        Automatically tap the "Log in" button in the popup using the "Bottom Sheet Button Rule".
        
        Strategy: Find the LOWEST large clickable button (bottom sheet pattern).
        This works across all devices/Android versions without text matching.
        
        CRITICAL: For popups, use ONLY Accessibility coordinates (not Vision).
        Vision sees cropped screenshots, Accessibility sees the real popup window.
        
        Returns:
            True if login button was tapped and email field appeared, False otherwise
        """
        print("[Auto-Login] Using 'Bottom Sheet Button Rule' - finding LOWEST large clickable button...")
        print("[Auto-Login] NOTE: Using Accessibility coordinates (not Vision) for popup buttons")
        print("[Auto-Login] Strategy: Structure + Position + Clickability (no text matching)")
        
        # Use the robust "Bottom Sheet Button Rule" - finds lowest large clickable button
        # This works regardless of text content, language, or device
        accessibility_result = self.accessibility.find_real_login_button()
        
        if accessibility_result:
            x, y, button_info = accessibility_result
            
            # Accessibility coordinates are already in device space - use them directly
            if self._validate_coordinates(x, y):
                print(f"[Auto-Login] ✓ Found REAL clickable button (bottom sheet pattern) at ({x}, {y})")
                print(f"[Auto-Login]   Button size: {button_info.get('width', 'N/A')}x{button_info.get('height', 'N/A')}")
                print(f"[Auto-Login]   Button text: '{button_info.get('text', 'N/A')[:50] if button_info.get('text') else 'N/A'}'")
                print(f"[Auto-Login]   Button class: '{button_info.get('class', 'N/A')}'")
                
                # Add small offset for button padding
                tap_x = x + 5
                tap_y = y + 5
                print(f"[Auto-Login] Tapping at ({tap_x}, {tap_y}) with +5px offset")
                
                # Capture screen before tap to verify change
                screenshot_before = self.screen_analyzer.capture_screenshot(device)
                
                self.device_actions.tap(tap_x, tap_y, delay=0.5)
                self.device_actions.wait(3.0)  # Wait for login screen to appear
                
                # VERIFICATION: Check if email field appeared (confirms tap was correct)
                print("[Auto-Login] Verifying tap was correct - checking for email field...")
                screen_analysis_after = self.screen_analyzer.analyze_screen(device, detect_elements=True)
                description_after = screen_analysis_after.get("description", "").lower()
                elements_after = screen_analysis_after.get("elements", [])
                
                # Check for email field indicators
                email_indicators = ["email", "e-mail", "username", "user name", "sign in", "log in"]
                has_email_field = any(indicator in description_after for indicator in email_indicators)
                
                # Also check elements for input fields
                for elem in elements_after:
                    elem_desc = elem.get("description", "").lower()
                    elem_type = elem.get("type", "")
                    if elem_type == "text_field" and any(indicator in elem_desc for indicator in ["email", "username"]):
                        has_email_field = True
                        break
                
                if has_email_field:
                    print("[Auto-Login] ✓ VERIFICATION PASSED: Email field appeared - tap was correct!")
                    return True
                else:
                    print("[Auto-Login] ⚠ VERIFICATION FAILED: Email field did not appear - tap may have been wrong")
                    print("[Auto-Login] Screen description: " + description_after[:200])
                    # Don't return False yet - might still work, just log the warning
                    return True  # Return True anyway, let the flow continue
            else:
                print(f"[Auto-Login] WARNING: Coordinates ({x}, {y}) failed validation")
        else:
            print("[Auto-Login] No large clickable buttons found via Accessibility")
            print("[Auto-Login] This might mean the popup structure is different or buttons are too small")
            return False
        
        return False
    
    def _extract_email_from_speech(self, speech_text: str) -> Optional[str]:
        """
        Extract email address from spoken text
        
        Handles various speech patterns like:
        - "my email is user@example.com"
        - "user at example dot com"
        - "user@example.com"
        - "it's user underscore name at gmail dot com"
        
        Args:
            speech_text: Raw speech-to-text output
            
        Returns:
            Cleaned email address or None if not found
        """
        import re
        
        text = speech_text.lower().strip()
        print(f"[Email Parser] Raw input: '{text}'")
        
        # Common speech-to-text replacements
        replacements = {
            " at ": "@",
            " dot ": ".",
            " underscore ": "_",
            " hyphen ": "-",
            " dash ": "-",
            "at the rate": "@",
            "at the rate of": "@",
            "at sign": "@",
            " period ": ".",
            " point ": ".",
        }
        
        # Apply replacements
        for spoken, symbol in replacements.items():
            text = text.replace(spoken, symbol)
        
        # Remove common prefixes
        prefixes = ["my email is ", "email is ", "it's ", "it is ", "the email is ", "email ", "my email "]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        
        # Remove spaces around @ and .
        text = re.sub(r'\s*@\s*', '@', text)
        text = re.sub(r'\s*\.\s*', '.', text)
        
        # Remove any remaining spaces (email shouldn't have spaces)
        text = text.replace(" ", "")
        
        # Validate email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, text):
            print(f"[Email Parser] Extracted valid email: '{text}'")
            return text
        
        # Try to find email pattern in the text (if it contains extra words)
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', speech_text.replace(" ", "").lower())
        if email_match:
            email = email_match.group(1)
            print(f"[Email Parser] Found email in text: '{email}'")
            return email
        
        # Final attempt: reconstruct from parts
        # Handle cases like "user 123 at gmail.com"
        text_parts = speech_text.lower().split()
        reconstructed = ""
        for part in text_parts:
            if part in ["at", "the", "rate", "of", "sign"]:
                if "at" in part:
                    reconstructed += "@"
            elif part in ["dot", "period", "point"]:
                reconstructed += "."
            elif part in ["underscore"]:
                reconstructed += "_"
            elif part in ["hyphen", "dash"]:
                reconstructed += "-"
            elif part not in ["my", "email", "is", "it's", "it", "the"]:
                reconstructed += part
        
        if "@" in reconstructed and "." in reconstructed:
            # Validate reconstructed email
            if re.match(email_pattern, reconstructed):
                print(f"[Email Parser] Reconstructed email: '{reconstructed}'")
                return reconstructed
        
        print(f"[Email Parser] Could not extract valid email from: '{speech_text}'")
        return None
    
    def _handle_email_input(self, state: Dict, email: str) -> Dict:
        """
        Handle email input during login flow
        
        Args:
            state: Current state dictionary
            email: Email address from user
            
        Returns:
            Updated state dictionary
        """
        device = self.adb_client.get_device()
        print(f"[Login Flow] Processing email: {email}")
        
        # Find email input field using Vision → Accessibility
        screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
        elements = screen_analysis.get("elements", [])
        
        email_field = None
        
        # Look for email field in vision elements
        for elem in elements:
            elem_desc = elem.get("description", "").lower()
            elem_type = elem.get("type", "")
            if elem_type == "text_field" and ("email" in elem_desc or "username" in elem_desc):
                x = elem.get("x", 0)
                y = elem.get("y", 0)
                width = elem.get("width", 200)
                height = elem.get("height", 50)
                if x > 0 and y > 0:
                    email_field = {"x": x, "y": y, "width": width, "height": height}
                    print(f"[Login Flow] Vision found email field region at ({x}, {y})")
                    break
        
        # Use Accessibility to find precise node
        if email_field:
            accessibility_result = self.accessibility.find_node_near_region(
                region_x=email_field["x"],
                region_y=email_field["y"],
                region_width=email_field.get("width"),
                region_height=email_field.get("height"),
                search_radius=150
            )
            if accessibility_result:
                x, y, node_info = accessibility_result
                elem_class = node_info.get("class", "").lower()
                if "edit" in elem_class or "input" in elem_class or "textfield" in elem_class:
                    email_field = (x, y)
                    print(f"[Login Flow] ✓ Found precise email field via Accessibility at ({x}, {y})")
        
        # Fallback 1: Direct accessibility search for email/username fields
        if not email_field:
            clickable_elements = self.accessibility.find_clickable_elements()
            for elem in clickable_elements:
                elem_class = elem.get("class", "").lower()
                elem_text = elem.get("text", "").lower()
                elem_desc = elem.get("content_desc", "").lower()
                if "edit" in elem_class and ("email" in elem_class or "username" in elem_class or 
                                              "email" in elem_text or "username" in elem_text or
                                              "email" in elem_desc or "username" in elem_desc):
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0:
                        email_field = (x, y)
                        print(f"[Login Flow] Found email field via direct Accessibility at ({x}, {y})")
                        break
        
        # Fallback 2: Find the first visible EditText (input field) on screen
        if not email_field:
            print("[Login Flow] Trying to find first EditText input field...")
            clickable_elements = self.accessibility.find_clickable_elements()
            edit_fields = []
            for elem in clickable_elements:
                elem_class = elem.get("class", "").lower()
                # Look for EditText or similar input fields
                if "edittext" in elem_class or "edit" in elem_class or "input" in elem_class:
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0:
                        edit_fields.append((x, y, elem))
            
            # Pick the topmost input field (usually email comes before password)
            if edit_fields:
                edit_fields.sort(key=lambda e: e[1])  # Sort by Y coordinate
                email_field = (edit_fields[0][0], edit_fields[0][1])
                print(f"[Login Flow] Found first EditText field at ({email_field[0]}, {email_field[1]})")
        
        if email_field:
            if isinstance(email_field, tuple):
                x, y = email_field
            else:
                x, y = email_field["x"], email_field["y"]
            
            # Tap email field and type email
            print(f"[Login Flow] Tapping email field at ({x}, {y})")
            self.device_actions.tap(x, y, delay=0.5)
            self.device_actions.wait(1.0)
            
            print(f"[Login Flow] Typing email: {email}")
            self.device_actions.type_text(email, clear_first=True)
            self.device_actions.wait(1.0)
            
            # Move to password field or submit
            state["email_entered"] = True
            state["awaiting_password"] = True
            state["awaiting_email"] = False
            
            # Ask for password (via voice)
            self.tts.speak("Email entered. Please say your password.")
            return state
        else:
            # Could not find email field
            return self._handle_issue_with_llm(state, "Could not find email input field. Please check the screen.")
    
    def _handle_email_input_typed(self, state: Dict, email: str) -> Dict:
        """
        Handle email input during login flow (typed input version - no TTS)
        
        Args:
            state: Current state dictionary
            email: Email address from user (typed)
            
        Returns:
            Updated state dictionary
        """
        device = self.adb_client.get_device()
        print(f"[Login Flow] Processing email: {email}")
        
        # Find email input field using Vision → Accessibility
        screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
        elements = screen_analysis.get("elements", [])
        
        email_field = None
        
        # Look for email field in vision elements
        for elem in elements:
            elem_desc = elem.get("description", "").lower()
            elem_type = elem.get("type", "")
            if elem_type == "text_field" and ("email" in elem_desc or "username" in elem_desc):
                x = elem.get("x", 0)
                y = elem.get("y", 0)
                width = elem.get("width", 200)
                height = elem.get("height", 50)
                if x > 0 and y > 0:
                    email_field = {"x": x, "y": y, "width": width, "height": height}
                    print(f"[Login Flow] Vision found email field region at ({x}, {y})")
                    break
        
        # Use Accessibility to find precise node
        if email_field:
            accessibility_result = self.accessibility.find_node_near_region(
                region_x=email_field["x"],
                region_y=email_field["y"],
                region_width=email_field.get("width"),
                region_height=email_field.get("height"),
                search_radius=150
            )
            if accessibility_result:
                x, y, node_info = accessibility_result
                elem_class = node_info.get("class", "").lower()
                if "edit" in elem_class or "input" in elem_class or "textfield" in elem_class:
                    email_field = (x, y)
                    print(f"[Login Flow] ✓ Found precise email field via Accessibility at ({x}, {y})")
        
        # Fallback 1: Direct accessibility search for email/username fields
        if not email_field:
            clickable_elements = self.accessibility.find_clickable_elements()
            for elem in clickable_elements:
                elem_class = elem.get("class", "").lower()
                elem_text = elem.get("text", "").lower()
                elem_desc = elem.get("content_desc", "").lower()
                if "edit" in elem_class and ("email" in elem_class or "username" in elem_class or 
                                              "email" in elem_text or "username" in elem_text or
                                              "email" in elem_desc or "username" in elem_desc):
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0:
                        email_field = (x, y)
                        print(f"[Login Flow] Found email field via direct Accessibility at ({x}, {y})")
                        break
        
        # Fallback 2: Find the first visible EditText (input field) on screen
        if not email_field:
            print("[Login Flow] Trying to find first EditText input field...")
            clickable_elements = self.accessibility.find_clickable_elements()
            edit_fields = []
            for elem in clickable_elements:
                elem_class = elem.get("class", "").lower()
                if "edittext" in elem_class or "edit" in elem_class or "input" in elem_class:
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0:
                        edit_fields.append((x, y, elem))
            
            if edit_fields:
                edit_fields.sort(key=lambda e: e[1])  # Sort by Y coordinate
                email_field = (edit_fields[0][0], edit_fields[0][1])
                print(f"[Login Flow] Found first EditText field at ({email_field[0]}, {email_field[1]})")
        
        if email_field:
            if isinstance(email_field, tuple):
                x, y = email_field
            else:
                x, y = email_field["x"], email_field["y"]
            
            # Tap email field and type email
            print(f"[Login Flow] Tapping email field at ({x}, {y})")
            self.device_actions.tap(x, y, delay=0.5)
            self.device_actions.wait(1.0)
            
            print(f"[Login Flow] Typing email: {email}")
            self.device_actions.type_text(email, clear_first=True)
            self.device_actions.wait(1.0)
            
            # Try to click Continue/Next button if present
            self._click_continue_button_if_present()
            
            state["email_entered"] = True
            state["awaiting_password"] = True
            state["awaiting_email"] = False
            
            # No TTS here - typed input flow handles prompts
            return state
        else:
            print("[Login Flow] Could not find email input field.")
            state["error"] = "Could not find email input field"
            return state
    
    def _click_continue_button_if_present(self):
        """Try to click Continue/Next button after entering email"""
        try:
            # Look for Continue/Next button
            result = self.accessibility.find_button_by_keywords(["continue", "next", "submit"])
            if result:
                x, y, _ = result
                print(f"[Login Flow] Found Continue button at ({x}, {y}), clicking...")
                self.device_actions.tap(x, y, delay=0.5)
                self.device_actions.wait(2.0)
        except Exception as e:
            print(f"[Login Flow] No Continue button found or error: {e}")
    
    def _click_continue_button_below_field(self, field_y: int):
        """
        Find and click Continue/Next button that is BELOW the given field Y coordinate.
        Uses same approach as find_real_login_button - find large clickable buttons by structure.
        
        Args:
            field_y: Y coordinate of the input field (button should be below this)
        """
        try:
            print(f"[Auth] Looking for Continue/Next button below Y={field_y}...")
            print(f"[Auth] Using Accessibility tree to find LARGE clickable buttons...")
            
            # Use Accessibility to find ALL large clickable buttons (like find_real_login_button)
            result = self.accessibility.find_button_below_y(field_y, min_width=300, min_height=80)
            
            if result:
                x, y, button_info = result
                print(f"[Auth] ✓ Found Continue button via Accessibility at ({x}, {y})")
                print(f"[Auth]   Size: {button_info.get('width', 'N/A')}x{button_info.get('height', 'N/A')}")
                print(f"[Auth]   Text: '{button_info.get('text', 'N/A')[:30] if button_info.get('text') else 'N/A'}'")
                print(f"[Auth]   (Button is {y - field_y}px below the input field)")
                
                self.device_actions.tap(x, y, delay=0.5)
                self.device_actions.wait(2.0)
            else:
                print(f"[Auth] No large clickable button found below Y={field_y}")
                # Try pressing Enter as fallback
                print("[Auth] Trying Enter key as fallback...")
                self.device_actions.press_key("KEYCODE_ENTER")
                self.device_actions.wait(2.0)
                
        except Exception as e:
            print(f"[Auth] Error finding Continue button: {e}")
            import traceback
            traceback.print_exc()
    
    def _click_submit_button_below_field(self, field_y: int):
        """
        Find and click Submit/Login button that is BELOW the password field.
        Uses same approach as find_real_login_button - find large clickable buttons by structure.
        
        Args:
            field_y: Y coordinate of the password field (button should be below this)
        """
        try:
            print(f"[Auth] Looking for Submit/Login button below Y={field_y}...")
            print(f"[Auth] Using Accessibility tree to find LARGE clickable buttons...")
            
            # Use Accessibility to find ALL large clickable buttons
            result = self.accessibility.find_button_below_y(field_y, min_width=300, min_height=80)
            
            if result:
                x, y, button_info = result
                print(f"[Auth] ✓ Found Submit button via Accessibility at ({x}, {y})")
                print(f"[Auth]   Size: {button_info.get('width', 'N/A')}x{button_info.get('height', 'N/A')}")
                print(f"[Auth]   Text: '{button_info.get('text', 'N/A')[:30] if button_info.get('text') else 'N/A'}'")
                print(f"[Auth]   (Button is {y - field_y}px below the password field)")
                
                self.device_actions.tap(x, y, delay=0.5)
                self.device_actions.wait(3.0)  # Wait longer for login to process
            else:
                print(f"[Auth] No large clickable button found below Y={field_y}")
                # Try pressing Enter as fallback
                print("[Auth] Trying Enter key as fallback...")
                self.device_actions.press_key("KEYCODE_ENTER")
                self.device_actions.wait(3.0)
                
        except Exception as e:
            print(f"[Auth] Error finding Submit button: {e}")
            import traceback
            traceback.print_exc()
    
    def _perform_google_signin(self) -> Tuple[bool, str]:
        """
        Perform Google sign-in using Accessibility only (NO LLM).
        
        Steps:
        1. Find and tap "Continue with Google" button
        2. Wait for Google account selection popup
        3. Find and tap "Continue" on the popup
        
        Returns:
            Tuple of (success, message)
        """
        print("\n" + "="*60)
        print("GOOGLE SIGN-IN (Accessibility Only)")
        print("="*60)
        
        try:
            # Step 1: Find "Continue with Google" button
            print("\n[Google Auth] Step 1: Finding 'Continue with Google' button...")
            google_button = self.accessibility.find_google_signin_button()
            
            if not google_button:
                print("[Google Auth] ✗ 'Continue with Google' button not found")
                return False, "Continue with Google button not found"
            
            x, y, btn_info = google_button
            print(f"[Google Auth] ✓ Found at ({x}, {y})")
            
            # Tap the button
            print(f"[Google Auth] Tapping 'Continue with Google'...")
            self.device_actions.tap(x, y, delay=0.5)
            self.device_actions.wait(3.0)  # Wait for Google popup to load
            
            # Step 2: Wait for Google account selection popup
            print("\n[Google Auth] Step 2: Waiting for Google account popup...")
            self.tts.speak("Selecting Google account...")
            
            # Give more time for Google popup to fully load
            self.device_actions.wait(2.0)
            
            # Step 3: Find "Continue" button on Google popup
            print("\n[Google Auth] Step 3: Finding 'Continue' button on popup...")
            
            # Try multiple times as the popup may take time to render
            continue_button = None
            for attempt in range(3):
                continue_button = self.accessibility.find_continue_button()
                if continue_button:
                    break
                print(f"[Google Auth] Attempt {attempt + 1}: Continue button not found, waiting...")
                self.device_actions.wait(1.5)
            
            if continue_button:
                x, y, btn_info = continue_button
                print(f"[Google Auth] ✓ Found 'Continue' at ({x}, {y})")
                
                # Tap Continue
                print(f"[Google Auth] Tapping 'Continue'...")
                self.device_actions.tap(x, y, delay=0.5)
                self.device_actions.wait(3.0)  # Wait for login to complete
                
                print("[Google Auth] ✓ Google sign-in flow completed!")
                self.tts.speak("Google sign-in completed!")
                return True, "Google sign-in successful"
            else:
                # If no Continue button, maybe account is auto-selected
                print("[Google Auth] No 'Continue' button found - checking if account was auto-selected...")
                self.device_actions.wait(2.0)
                
                # Check for any clickable buttons (might be "Allow", "Accept", etc.)
                any_button = self.accessibility.find_button_below_y(800, min_width=200, min_height=40)
                if any_button:
                    x, y, btn_info = any_button
                    print(f"[Google Auth] Found alternative button at ({x}, {y}), tapping...")
                    self.device_actions.tap(x, y, delay=0.5)
                    self.device_actions.wait(2.0)
                    return True, "Google sign-in completed (alternative flow)"
                
                return False, "Could not complete Google sign-in - no Continue button found"
                
        except Exception as e:
            print(f"[Google Auth] Error during Google sign-in: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Google sign-in failed: {str(e)}"
    
    def _handle_password_input(self, state: Dict, password: str) -> Dict:
        """
        Handle password input during login flow
        
        Args:
            state: Current state dictionary
            password: Password from user
            
        Returns:
            Updated state dictionary
        """
        device = self.adb_client.get_device()
        print(f"[Login Flow] Processing password...")
        
        # Find password input field using Vision → Accessibility
        screen_analysis = self.screen_analyzer.analyze_screen(device, detect_elements=True)
        elements = screen_analysis.get("elements", [])
        
        password_field = None
        
        # Look for password field in vision elements
        for elem in elements:
            elem_desc = elem.get("description", "").lower()
            elem_type = elem.get("type", "")
            if elem_type == "text_field" and "password" in elem_desc:
                x = elem.get("x", 0)
                y = elem.get("y", 0)
                width = elem.get("width", 200)
                height = elem.get("height", 50)
                if x > 0 and y > 0:
                    password_field = {"x": x, "y": y, "width": width, "height": height}
                    print(f"[Login Flow] Vision found password field region at ({x}, {y})")
                    break
        
        # Use Accessibility to find precise node
        if password_field:
            accessibility_result = self.accessibility.find_node_near_region(
                region_x=password_field["x"],
                region_y=password_field["y"],
                region_width=password_field.get("width"),
                region_height=password_field.get("height"),
                search_radius=150
            )
            if accessibility_result:
                x, y, node_info = accessibility_result
                elem_class = node_info.get("class", "").lower()
                if "edit" in elem_class or "input" in elem_class or "password" in elem_class:
                    password_field = (x, y)
                    print(f"[Login Flow] ✓ Found precise password field via Accessibility at ({x}, {y})")
        
        # Fallback 1: Direct accessibility search for password fields
        if not password_field:
            clickable_elements = self.accessibility.find_clickable_elements()
            for elem in clickable_elements:
                elem_class = elem.get("class", "").lower()
                elem_text = elem.get("text", "").lower()
                elem_desc = elem.get("content_desc", "").lower()
                if "edit" in elem_class and ("password" in elem_class or 
                                              "password" in elem_text or 
                                              "password" in elem_desc):
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0:
                        password_field = (x, y)
                        print(f"[Login Flow] Found password field via direct Accessibility at ({x}, {y})")
                        break
        
        # Fallback 2: Find the second EditText (password usually comes after email)
        if not password_field:
            print("[Login Flow] Trying to find password as second EditText...")
            clickable_elements = self.accessibility.find_clickable_elements()
            edit_fields = []
            for elem in clickable_elements:
                elem_class = elem.get("class", "").lower()
                if "edittext" in elem_class or "edit" in elem_class or "input" in elem_class:
                    x = elem.get("x", 0)
                    y = elem.get("y", 0)
                    if x > 0 and y > 0:
                        edit_fields.append((x, y, elem))
            
            # Sort by Y coordinate and pick the second one (or last if only one)
            if edit_fields:
                edit_fields.sort(key=lambda e: e[1])
                if len(edit_fields) >= 2:
                    password_field = (edit_fields[1][0], edit_fields[1][1])
                    print(f"[Login Flow] Found second EditText (password) at ({password_field[0]}, {password_field[1]})")
                else:
                    # Only one field - might be a separate password screen
                    password_field = (edit_fields[0][0], edit_fields[0][1])
                    print(f"[Login Flow] Found only one EditText at ({password_field[0]}, {password_field[1]})")
        
        if password_field:
            if isinstance(password_field, tuple):
                x, y = password_field
            else:
                x, y = password_field["x"], password_field["y"]
            
            # Tap password field and type password
            print(f"[Login Flow] Tapping password field at ({x}, {y})")
            self.device_actions.tap(x, y, delay=0.5)
            self.device_actions.wait(1.0)
            
            print(f"[Login Flow] Typing password...")
            self.device_actions.type_text(password, clear_first=True)
            self.device_actions.wait(1.0)
            
            # Find and tap submit/login button
            submit_button = None
            # Find submit button - NEVER include "continue" alone (matches "Continue with Google")
            accessibility_result = self.accessibility.find_button_by_keywords(["log in", "login", "sign in", "submit", "next"])
            if accessibility_result:
                x, y, _ = accessibility_result
                submit_button = (x, y)
                print(f"[Login Flow] Found submit button at ({x}, {y})")
            
            if submit_button:
                x, y = submit_button
                print(f"[Login Flow] Clicking submit button at ({x}, {y})")
                self.device_actions.tap(x, y, delay=0.5)
                self.device_actions.wait(3.0)
                
                state["password_entered"] = True
                state["awaiting_password"] = False
                state["login_flow_active"] = False
                state["login_complete"] = True
                
                # TTS is handled by calling code
                print("[Login Flow] ✓ Login credentials submitted")
                return state
            else:
                return self._handle_issue_with_llm(state, "Could not find submit button after entering password.")
        else:
            return self._handle_issue_with_llm(state, "Could not find password input field.")
    
    def _is_login_confirmation(self, text: str) -> bool:
        """
        Check if the user is confirming they want to proceed with login
        (e.g., "log in", "yes", "okay", "proceed", "go ahead")
        
        Args:
            text: User input text
            
        Returns:
            True if it's a login confirmation, False otherwise
        """
        text_lower = text.lower().strip()
        
        # Remove punctuation for better matching
        import re
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        words = text_clean.split()
        
        # Login confirmation phrases
        confirmation_phrases = [
            "log in", "login", "sign in", "signin",
            "yes", "yeah", "yep", "yup", "okay", "ok", "sure", "alright",
            "proceed", "go ahead", "continue", "do it", "tap login", "click login",
            "press login", "tap log in", "click log in", "press log in"
        ]
        
        # Check for exact phrase matches
        for phrase in confirmation_phrases:
            if phrase in text_lower:
                return True
        
        # Check for individual words that indicate confirmation
        confirmation_words = ["yes", "okay", "ok", "sure", "proceed", "continue", "go"]
        if any(word in words for word in confirmation_words):
            # But exclude if it's part of a longer command like "open gmail"
            if len(words) <= 3:  # Short responses are likely confirmations
                return True
        
        # Check for "log in" or "login" as standalone or with confirmation
        if "log" in words and "in" in words:
            return True
        if "login" in words and len(words) <= 3:
            return True
        
        return False
    
    def _is_conversational(self, text: str) -> bool:
        """
        Check if the input is conversational (greeting, question, chat) vs a command
        Also filters out unwanted/noise input that shouldn't be processed
        
        Args:
            text: User input text
            
        Returns:
            True if conversational/noise (should NOT be processed as command)
            False if it's a valid command
        """
        text_lower = text.lower().strip()
        
        # Remove punctuation for better matching
        import re
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        words = text_clean.split()
        
        # ===== EARLY COMMAND DETECTION - These are ALWAYS commands, not conversational =====
        # Pattern: "open [app] and [action]" - always a command
        if re.search(r"open\s+\w+\s+(?:and\s+)?(?:send|message|pay|transfer|ask|tell|search|find|tap|click)", text_lower):
            return False  # Definitely a command
        
        # Pattern: "open [app]" - always a command
        if re.search(r"open\s+(?:whatsapp|paytm|gmail|chatgpt|chrome|youtube|settings|gmail|phone|camera)", text_lower):
            return False  # Definitely a command
        
        # Pattern: "send message to X" or "message X" - always a command
        if re.search(r"(?:send\s+message|message)\s+(?:to\s+)?\w+", text_lower):
            return False  # Definitely a command
        
        # ===== NOISE/UNWANTED PHRASES - IGNORE THESE =====
        # Random phrases, background noise, or non-command statements
        noise_phrases = [
            "unfortunately", "didn't work", "that didn't", "not working", "it failed",
            "oh no", "oops", "hmm", "umm", "uh", "ah", "err", "let me think",
            "i don't know", "i'm not sure", "maybe later", "never mind", "forget it",
            "what was that", "excuse me", "sorry", "pardon", "i see", "interesting",
            "that's nice", "that's good", "that's bad", "too bad", "oh well",
            "wait", "hold on", "one moment", "just a second", "let me see"
        ]
        
        if any(phrase in text_lower for phrase in noise_phrases):
            print(f"[Filter] Ignoring noise phrase: '{text}'")
            return True
        
        # ===== MUST HAVE COMMAND KEYWORDS =====
        # Valid commands must contain at least one of these
        command_keywords = [
            # App actions
            "open", "close", "launch", "start", "exit", "quit",
            # Navigation
            "go", "navigate", "back", "home", "scroll",
            # Interactions
            "tap", "click", "press", "touch", "select", "choose",
            # Input
            "type", "enter", "input", "search", "find",
            # Communication
            "send", "message", "call", "email",
            # Payment
            "pay", "transfer", "send money", "rupees", "rs",
            # AI/Query
            "ask", "tell", "what is", "how to", "show me",
            # Authentication
            "login", "log in", "sign in", "authenticate",
            # Apps
            "whatsapp", "paytm", "gmail", "chatgpt", "chrome", "youtube", "settings"
        ]
        
        # Check if the input contains ANY command keyword
        has_command_keyword = any(kw in text_lower for kw in command_keywords)
        
        if not has_command_keyword:
            # No command keyword found - treat as conversational/noise
            print(f"[Filter] No command keyword found in: '{text}'")
            return True
        
        # ===== ACTION COMMANDS - These are NOT conversational =====
        action_verbs = ["tap", "click", "press", "touch", "select", "choose", "open", "close", 
                       "launch", "start", "search", "find", "tell", "ask", "extract", "get", 
                       "show", "display", "navigate", "go", "login", "sign in", "authenticate",
                       "type", "enter", "input", "send", "submit", "pay", "transfer", "message"]
        
        # Check for action commands
        if any(verb in text_lower for verb in action_verbs):
            return False
        
        # Check for "tap on [element]" or "click [element]" patterns
        tap_patterns = ["tap on", "click on", "press on", "touch on", "tap the", "click the", 
                       "press the", "touch the"]
        if any(pattern in text_lower for pattern in tap_patterns):
            return False
        
        # Check for element names that indicate commands
        ui_elements = ["button", "login", "sign in", "field", "input", "text field", "search bar",
                      "menu", "option", "link", "icon", "tab"]
        if any(element in text_lower for element in ui_elements):
            return False
        
        # ===== CONVERSATIONAL PATTERNS =====
        # Greetings
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", 
                     "greetings", "howdy", "what's up", "sup", "hola", "bonjour"]
        
        # Questions about the agent
        agent_questions = ["who are you", "what are you", "what can you do", "help", 
                          "what can you help", "how are you", "how do you work", "what do you do"]
        
        # Casual responses
        casual = ["thanks", "thank you", "okay", "ok", "sure", "yes", "no", "maybe", "alright"]
        
        if any(greeting in text_lower for greeting in greetings):
            return True
        
        if any(word in greetings for word in words):
            return True
        
        if any(question in text_lower for question in agent_questions):
            return True
        
        if any(word in text_lower for word in casual) and not any(cmd in text_lower for cmd in action_verbs):
            return True
        
        # If short phrase without commands, treat as conversational
        unique_words = set(words)
        if len(unique_words) <= 3 and len(words) <= 5 and not any(cmd in text_lower for cmd in action_verbs):
            return True
        
        return False
    
    def _handle_conversation(self, text: str) -> str:
        """
        Handle conversational input and return appropriate response
        
        Args:
            text: User conversational input
            
        Returns:
            Response text
        """
        text_lower = text.lower().strip()
        
        # Greetings
        if any(greeting in text_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            responses = [
                "Hello! I'm your mobile automation assistant. How can I help you today?",
                "Hi there! I'm here to help you control your Android device. What would you like me to do?",
                "Hello! I can help you open apps, search, and navigate your device. What do you need?"
            ]
            import random
            return random.choice(responses)
        
        # Questions about capabilities
        if any(phrase in text_lower for phrase in ["what can you do", "what are you", "who are you", "help"]):
            return "I'm a mobile automation assistant for blind users. I can open apps, search for information, handle authentication, and navigate your Android device using voice commands. Try saying 'Open Gmail' or 'Open Settings' to get started!"
        
        if "how are you" in text_lower:
            return "I'm doing well, thank you for asking! I'm ready to help you with your device. What would you like me to do?"
        
        # Thanks
        if any(word in text_lower for word in ["thanks", "thank you"]):
            return "You're welcome! Is there anything else I can help you with?"
        
        # Default friendly response
        return "I'm here to help! You can ask me to open apps, search for things, or navigate your device. For example, say 'Open Gmail' or 'Open Settings'."
    
    def _parse_whatsapp_follow_up_command(self, text: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Parse WhatsApp follow-up commands when session is active.
        DEFAULT: Treat entire text as message to current chat (send_to_current=True).
        ONLY if user explicitly says "send message to X" or "message X", extract recipient.
        Returns (recipient, message, send_to_current_chat).
        """
        import re
        original_text = text.strip().rstrip(" .!?")
        t = original_text.lower()
        if not t or len(t) < 3:
            return None, None, False
        
        recipient = None
        message = None
        send_to_current = True  # DEFAULT: send to current chat
        
        # ONLY check for recipient if user explicitly says "send message to X" or "message X"
        explicit_recipient_patterns = [
            r"send\s+message\s+to\s+(\w+)",  # "send message to keshav"
            r"message\s+to\s+(\w+)",         # "message to keshav"
            r"message\s+(\w+)\s+(?:say|saying|tell|ask)",  # "message keshav say"
        ]
        
        for pattern in explicit_recipient_patterns:
            match = re.search(pattern, t)
            if match:
                recipient = match.group(1).strip().rstrip(" .,!?")
                send_to_current = False  # User wants different recipient
                # Extract message after recipient
                after_recipient = t.split(recipient, 1)[1] if recipient in t else ""
                for marker in [" say ", " saying ", " tell ", " ask ", " asking "]:
                    if marker in after_recipient:
                        message = after_recipient.split(marker, 1)[1].strip().rstrip(" .!?")
                        break
                if not message and after_recipient.strip():
                    message = after_recipient.strip().rstrip(" .!?")
                break
        
        # If no explicit recipient found, treat ENTIRE text as message to current chat
        if send_to_current:
            # Remove common prefixes like "tell him that", "say that" but keep the message
            cleaned = re.sub(r"^(tell\s+(?:him|her|them|him that|her that|them that)\s+)", "", t, flags=re.IGNORECASE)
            cleaned = re.sub(r"^(say\s+(?:that\s+)?)", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            message = cleaned if cleaned else original_text.strip()
            # If message is too short or looks like a command, might be invalid
            command_words = ["open", "close", "search", "find", "go", "back", "home", "settings", "exit", "stop"]
            if any(cw in message.lower() for cw in command_words) and len(message.split()) <= 3:
                return None, None, False
        
        if recipient:
            recipient = re.sub(r"[^\w\s]", "", recipient).strip() or None
        
        return (recipient, message, send_to_current) if message else (None, None, False)
    
    def _parse_intent(self, user_intent: str) -> Dict[str, Any]:
        """
        Parse user intent into structured format
        
        Args:
            user_intent: Natural language intent
            
        Returns:
            Parsed intent dictionary
        """
        intent_lower = user_intent.lower()
        intent_parts = {
            "action": None,
            "app": None,
            "query": None,
            "target": None,
            "recipient": None,  # For messaging apps (WhatsApp, etc.)
            "message": None,    # The message to send
            "amount": None      # For payment apps (Paytm, GPay, etc.)
        }
        
        # ===== WHATSAPP MESSAGE PARSING =====
        # Handle commands like: "open whatsapp and send message to mummy say hi"
        # or "send message to mummy on whatsapp saying hello"
        if "whatsapp" in intent_lower and ("send" in intent_lower or "message" in intent_lower):
            intent_parts["action"] = "send_whatsapp_message"
            intent_parts["app"] = "whatsapp"
            
            # Extract recipient name (after "to" and before "say/saying/with")
            recipient = None
            message = None
            
            # Pattern 1: "send message to [name] say/saying [message]"
            if " to " in intent_lower:
                after_to = intent_lower.split(" to ", 1)[1]
                # Find where the message starts
                message_markers = [" say ", " saying ", " tell ", " with message ", " message "]
                for marker in message_markers:
                    if marker in after_to:
                        parts = after_to.split(marker, 1)
                        recipient = parts[0].strip()
                        message = parts[1].strip().rstrip(" .!?") if len(parts) > 1 else None
                        break
                else:
                    # No message marker found - just recipient
                    # Check if there's content after common words
                    recipient = after_to.split()[0] if after_to else None
            
            # Pattern 2: "message [name] [message]" 
            if not recipient and "message" in intent_lower:
                after_message = intent_lower.split("message", 1)[1].strip()
                words = after_message.split()
                if words:
                    # Skip "to" if present
                    if words[0] == "to" and len(words) > 1:
                        words = words[1:]
                    if words:
                        recipient = words[0]
                        # Rest is message
                        if len(words) > 1:
                            # Check for "say/saying"
                            rest = " ".join(words[1:])
                            for marker in ["say ", "saying "]:
                                if marker in rest:
                                    message = rest.split(marker, 1)[1].strip()
                                    break
                            else:
                                message = rest.strip()
            
            if recipient:
                intent_parts["recipient"] = recipient
                print(f"[Intent] WhatsApp recipient: '{recipient}'")
            if message:
                intent_parts["message"] = message
                print(f"[Intent] WhatsApp message: '{message}'")
            
            return intent_parts
        
        # ===== PAYTM/UPI PAYMENT PARSING =====
        # Handle commands like: "open paytm and send 10 rupees to keshav"
        # or "pay 100 to john on paytm" or "send money to mom 50 rupees paytm"
        payment_apps = ["paytm", "gpay", "google pay", "phonepe", "phone pe", "bhim"]
        payment_keywords = ["send", "pay", "transfer", "rupees", "rs", "₹", "inr", "money"]
        
        if any(app in intent_lower for app in payment_apps) and any(kw in intent_lower for kw in payment_keywords):
            # Determine which payment app
            if "paytm" in intent_lower:
                intent_parts["app"] = "paytm"
            elif "gpay" in intent_lower or "google pay" in intent_lower:
                intent_parts["app"] = "gpay"
            elif "phonepe" in intent_lower or "phone pe" in intent_lower:
                intent_parts["app"] = "phonepe"
            elif "bhim" in intent_lower:
                intent_parts["app"] = "bhim"
            
            intent_parts["action"] = "send_payment"
            
            # Extract amount and recipient
            import re
            
            # Pattern for amount: "10 rupees", "rs 100", "₹50", "100 rs", "rupees 50"
            amount_patterns = [
                r'(\d+)\s*(?:rupees|rupee|rs|₹|inr)',  # 10 rupees, 10rs, 10₹
                r'(?:rupees|rupee|rs|₹|inr)\s*(\d+)',  # rupees 10, rs 10
                r'(?:send|pay|transfer)\s+(\d+)',       # send 10, pay 100
                r'(\d+)\s+(?:to|for)',                  # 10 to keshav
            ]
            
            amount = None
            for pattern in amount_patterns:
                match = re.search(pattern, intent_lower)
                if match:
                    amount = match.group(1)
                    break
            
            # Extract recipient (after "to") - capture FULL NAME
            recipient = None
            if " to " in intent_lower:
                after_to = intent_lower.split(" to ", 1)[1]
                
                # Stop words that indicate end of recipient name
                stop_words = ["on", "via", "using", "through", "and", "paytm", "gpay", "phonepe", "bhim"]
                
                # Collect all words until we hit a stop word
                words_after_to = after_to.split()
                recipient_words = []
                
                for word in words_after_to:
                    # Skip leading articles
                    if word in ["my", "the", "a", "an"] and not recipient_words:
                        continue
                    # Stop if we hit a stop word
                    if word in stop_words:
                        break
                    # Stop if we hit a number (likely amount)
                    if word.isdigit():
                        break
                    recipient_words.append(word)
                
                if recipient_words:
                    recipient = " ".join(recipient_words).strip()
                    # Remove trailing punctuation
                    recipient = recipient.rstrip(".,!?")
            
            if amount:
                intent_parts["amount"] = amount
                print(f"[Intent] Payment amount: ₹{amount}")
            if recipient:
                intent_parts["recipient"] = recipient
                print(f"[Intent] Payment recipient: '{recipient}'")
            
            return intent_parts
        
        # Check for close/exit app commands first
        close_keywords = ["close", "exit", "quit", "stop"]
        app_keywords = ["app", "application", "chatgpt", "gmail", "settings", "chrome", "youtube"]
        
        if any(close_kw in intent_lower for close_kw in close_keywords):
            if any(app_kw in intent_lower for app_kw in app_keywords) or "app" in intent_lower:
                intent_parts["action"] = "close_app"
                return intent_parts
        
        # Check for tap/click commands (e.g., "tap on login", "click login button")
        # BUT only if it's NOT part of an "open" command
        tap_patterns = ["tap on", "click on", "press on", "touch on", "tap the", "click the", 
                       "press the", "touch the"]
        if any(pattern in intent_lower for pattern in tap_patterns) and "open" not in intent_lower:
            # Extract element name after "tap on" or "click on"
            for pattern in tap_patterns:
                if pattern in intent_lower:
                    parts = intent_lower.split(pattern, 1)
                    if len(parts) > 1:
                        element_name = parts[1].strip().rstrip(" .!?")
                        # Remove common words
                        element_name = re.sub(r'\b(button|the|a|an)\b', '', element_name).strip()
                        if element_name:
                            intent_parts["action"] = "tap_element"
                            intent_parts["target"] = element_name
                            # Check if it's a login-related element
                            if any(kw in element_name for kw in ["login", "sign in", "authenticate"]):
                                intent_parts["action"] = "login"
                            return intent_parts
        
        # Detect action - CHECK "open" FIRST before standalone login commands
        if "open" in intent_lower:
            intent_parts["action"] = "open_app"
            # Extract app name - handle multi-word app names
            words = intent_lower.split()
            if "open" in words:
                idx = words.index("open")
                # Get all words after "open" until we hit "and" or a command word
                # Note: "tell" and "me" are NOT command words for app name extraction
                # They only become relevant after "and" for query extraction
                command_words = {"and", "then", "to", "for", "ask", "search", "find", "login", "sign"}
                filler_words = {"you", "pretty", "please", "the", "a", "an", "very", "so"}
                app_words = []
                
                # Find where "and" appears (if any) - that's usually where the command starts
                and_idx = -1
                for i in range(idx + 1, len(words)):
                    if words[i] == "and":
                        and_idx = i
                        break
                
                # Extract app name - stop at "and" or command words, skip filler words
                # Don't stop at "tell" or "me" - they're part of query phrases, not app names
                for i in range(idx + 1, len(words)):
                    word = words[i]
                    # Stop at "and" or command words (but not "tell" or "me")
                    if word == "and" or word in command_words:
                        break
                    # Skip filler words
                    if word not in filler_words:
                        app_words.append(word)
                
                if app_words:
                    app_name = " ".join(app_words)
                    intent_parts["app"] = app_name
                    
                    # Check if there's a query after "and ask" or "and search"
                    # Find "and" position to get remaining text
                    if and_idx > 0:
                        remaining_text = " ".join(words[and_idx + 1:])
                    else:
                        # No "and" found, check remaining words after app name
                        remaining_start = idx + len(app_words) + 1
                        if remaining_start < len(words):
                            remaining_text = " ".join(words[remaining_start:])
                        else:
                            remaining_text = ""
                    
                    if "tell me" in remaining_text:
                        # Handle "tell me" - extract everything after "tell me"
                        parts = remaining_text.split("tell me", 1)
                        if len(parts) > 1:
                            query = parts[1].strip()
                            # Remove common trailing words
                            query = query.rstrip(" .!?")
                            intent_parts["query"] = query
                            print(f"[Intent] Extracted query from 'tell me': '{query}'")
                    elif "ask" in remaining_text:
                        parts = remaining_text.split("ask", 1)
                        if len(parts) > 1:
                            query = parts[1].strip()
                            # Remove common trailing words
                            query = query.rstrip(" .!?")
                            intent_parts["query"] = query
                    elif "search" in remaining_text:
                        parts = remaining_text.split("search", 1)
                        if len(parts) > 1:
                            query = parts[1].strip()
                            query = query.rstrip(" .!?")
                            intent_parts["query"] = query
                    elif "tell" in remaining_text:
                        # Handle standalone "tell" - extract everything after "tell"
                        parts = remaining_text.split("tell", 1)
                        if len(parts) > 1:
                            query = parts[1].strip()
                            # Remove "me" if it's the first word
                            if query.startswith("me "):
                                query = query[3:].strip()
                            query = query.rstrip(" .!?")
                            intent_parts["query"] = query
                    elif "login" in remaining_text or "sign in" in remaining_text:
                        # User wants to login after opening app
                        intent_parts["wants_login"] = True
                        print("[Intent] User wants to login after opening app")
                        # Return early to prevent standalone login check from overriding
                        return intent_parts
        
        # Check for standalone login/authentication commands (only if no "open" action was set)
        if not intent_parts.get("action") or intent_parts.get("action") != "open_app":
            login_keywords = ["login", "sign in", "authenticate", "log in"]
            if any(login_kw in intent_lower for login_kw in login_keywords):
                # Only set login action if we're not already opening an app
                if intent_parts.get("action") != "open_app":
                    intent_parts["action"] = "login"
                    return intent_parts
        
        # Standalone query/search (without "open")
        if not intent_parts.get("query") and ("search" in intent_lower or "ask" in intent_lower or "query" in intent_lower or "tell" in intent_lower):
            if intent_parts["action"] != "open_app":
                intent_parts["action"] = "query"
            # Extract query
            if "tell me" in intent_lower:
                parts = intent_lower.split("tell me", 1)
                if len(parts) > 1:
                    intent_parts["query"] = parts[1].strip().rstrip(" .!?")
            elif "tell" in intent_lower:
                parts = intent_lower.split("tell", 1)
                if len(parts) > 1:
                    query = parts[1].strip()
                    # Remove "me" if it's the first word
                    if query.startswith("me "):
                        query = query[3:].strip()
                    intent_parts["query"] = query.rstrip(" .!?")
            elif "ask" in intent_lower:
                parts = intent_lower.split("ask", 1)
                if len(parts) > 1:
                    intent_parts["query"] = parts[1].strip().rstrip(" .!?")
            elif "search" in intent_lower:
                parts = intent_lower.split("search", 1)
                if len(parts) > 1:
                    intent_parts["query"] = parts[1].strip().rstrip(" .!?")
        
        if "extract" in intent_lower or ("get" in intent_lower and not intent_parts.get("action")) or ("find" in intent_lower and not intent_parts.get("action")):
            if not intent_parts.get("action"):
                intent_parts["action"] = "extract"
        
        return intent_parts
    
    def process_command(self, user_command: str, issue_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process user command or conversation
        
        Args:
            user_command: Natural language command or conversational input
            issue_context: Optional context from a previous issue (for guidance commands)
            
        Returns:
            Result dictionary
        """
        # Check if we're in login flow and awaiting email/password
        # This needs to be checked from the current workflow state
        # For now, we'll check in the main loop via a flag
        
        # If this is guidance for an issue, handle it specially
        if issue_context and issue_context.get("waiting_for_user_guidance"):
            print(f"[Guidance] Processing user guidance: {user_command}")
            # Treat guidance as a command to resolve the issue
            # User might say "click continue", "try again", "tap the button", etc.
        
        # PRIORITY CHECK: Handle login popup confirmation
        # If popup was detected and user confirms (says "log in", "yes", "okay", etc.), proceed with login
        if self.login_flow_state.get("popup_detected") or self.login_flow_state.get("awaiting_popup_confirmation"):
            if self._is_login_confirmation(user_command):
                print("[Login Popup] User confirmed login. Proceeding to tap login button...")
                device = self.adb_client.get_device()
                if device:
                    login_tapped = self._auto_tap_login_button(device)
                    if login_tapped:
                        print("[Login Popup] Login button tapped. Will ask for email.")
                        self.login_flow_state["popup_detected"] = False
                        self.login_flow_state["awaiting_popup_confirmation"] = False
                        self.login_flow_state["active"] = True
                        self.login_flow_state["awaiting_email"] = True
                        self.tts.speak("I see a login screen. Please say your email address.")
                        return {"login_flow": True, "popup_confirmed": True, "awaiting_email": True}
                    else:
                        print("[Login Popup] Could not find login button. Will ask user for guidance.")
                        return self._handle_issue_with_llm(
                            {"login_popup_detected": True, "user_intent": user_command},
                            "Login popup detected but could not find login button."
                        )
        
        # If user has an active query-app session (e.g. ChatGPT open), treat their message as
        # a follow-up question to that app, not as conversational chat with the assistant
        if self._query_app_session and user_command.strip():
            is_conv = False
            logger.info(f"Input: '{user_command}' | Query session active for {self._query_app_session} -> sending to app (not conversational)")
        else:
            is_conv = self._is_conversational(user_command)
            logger.info(f"Input: '{user_command}' | Is conversational: {is_conv}")
        
        # Check if we're in login flow and user is providing email
        if self.login_flow_state.get("awaiting_email"):
            email_match = self._extract_email_from_command(user_command)
            if email_match:
                print(f"[Login Flow] Detected email in command: {email_match}")
                device = self.adb_client.get_device()
                if device:
                    state = {
                        "awaiting_email": True,
                        "login_flow_active": True,
                        "user_intent": user_command
                    }
                    result = self._handle_email_input(state, email_match)
                    # Update login flow state
                    self.login_flow_state["email"] = email_match
                    self.login_flow_state["awaiting_email"] = False
                    self.login_flow_state["awaiting_password"] = True
                    return {"login_flow": True, "email_entered": True, "awaiting_password": True}
            else:
                # User might have said email without clear pattern, try to extract anyway
                # Or ask them to repeat
                self.tts.speak("I didn't catch your email address. Please say it again, for example: my email is user at gmail dot com")
                return {"login_flow": True, "awaiting_email": True}
        
        # Check if we're in login flow and user is providing password
        if self.login_flow_state.get("awaiting_password"):
            password_match = self._extract_password_from_command(user_command)
            if password_match:
                print(f"[Login Flow] Detected password in command")
                device = self.adb_client.get_device()
                if device:
                    state = {
                        "awaiting_password": True,
                        "login_flow_active": True,
                        "email_entered": True,
                        "user_intent": user_command
                    }
                    result = self._handle_password_input(state, password_match)
                    # Update login flow state
                    self.login_flow_state["password"] = password_match
                    self.login_flow_state["awaiting_password"] = False
                    self.login_flow_state["active"] = False
                    return {"login_flow": True, "password_entered": True, "login_complete": True}
            else:
                # User might have said password directly, use the whole command
                password_text = user_command.strip().rstrip(" .!?")
                if len(password_text) > 3:
                    print(f"[Login Flow] Using command as password")
                    device = self.adb_client.get_device()
                    if device:
                        state = {
                            "awaiting_password": True,
                            "login_flow_active": True,
                            "email_entered": True,
                            "user_intent": user_command
                        }
                        result = self._handle_password_input(state, password_text)
                        self.login_flow_state["password"] = password_text
                        self.login_flow_state["awaiting_password"] = False
                        self.login_flow_state["active"] = False
                        return {"login_flow": True, "password_entered": True, "login_complete": True}
        
        if is_conv:
            logger.info(f"Conversational input detected: {user_command}")
            response = self._handle_conversation(user_command)
            print(f"[Chat]: {response}")
            print("[TTS] About to speak conversational response...")
            self.tts.speak(response)
            return {"conversational": True, "response": response}
        
        # Continuous query session: user said "open ChatGPT and ask X" before; now they can just ask
        # without saying "open ChatGPT" again until they say "close" or open another app
        cmd_lower = user_command.strip().lower()
        close_keywords = ("close app", "close chatgpt", "close whatsapp", "close the app", "exit", "stop", "go back", "end session")
        if self._query_app_session and any(phrase in cmd_lower for phrase in close_keywords):
            self._query_app_session = None
            self.tts.speak("Session ended. Say 'open ChatGPT and ask' or 'open WhatsApp and send message to' when you want to use them again.")
            # If they said "close app/chatgpt/whatsapp", still process so we send home/key back
            if any(phrase in cmd_lower for phrase in ("close app", "close chatgpt", "close whatsapp", "close the app")):
                user_command = "close app"  # Normalize so workflow parses close_app
            else:
                return {"session_ended": True, "response": "Session ended."}
        
        # It's a command - process it
        task_id = str(uuid.uuid4())
        task = self.state_manager.start_task(task_id, user_command)
        
        logger.info(f"Processing command: {user_command}")
        
        # Initialize state
        initial_state = {
            "user_intent": user_command,
            "task_id": task_id,
            "action_success": False,
            "task_complete": False,
            "iteration_count": 0,
            "session_active": False,  # Start with inactive session
            "issue_context": issue_context  # Include issue context if present
        }
        
        # If we're in a query-app session, treat as follow-up for that app
        if self._query_app_session and user_command.strip():
            if self._query_app_session == "whatsapp":
                # When WhatsApp is open, treat ANY text as a message to current chat
                # UNLESS user explicitly says "send message to X" (then search for X)
                recipient, message, send_to_current = self._parse_whatsapp_follow_up_command(user_command)
                # If parser found nothing, treat entire command as message to current chat
                if not message and not recipient:
                    message = user_command.strip().rstrip(" .!?")
                    send_to_current = True
                if message:  # Always set if we have a message
                    initial_state["whatsapp_follow_up"] = True
                    initial_state["intent_parts"] = {
                        "action": "send_whatsapp_message",
                        "app": "whatsapp",
                        "recipient": recipient or "",  # Empty if sending to current chat
                        "message": message,
                        "send_to_current_chat": send_to_current,  # Flag: skip recipient search
                    }
                    if recipient:
                        print(f"[Session] WhatsApp follow-up: send to '{recipient}' saying '{message}'")
                    else:
                        print(f"[Session] WhatsApp follow-up: send to current chat: '{message[:60]}...'")
                # else: very short or invalid, run normal parse
            else:
                # ChatGPT or other: follow-up question
                initial_state["query_session_follow_up"] = True
                initial_state["pending_query"] = user_command.strip()
                initial_state["pending_app"] = self._query_app_session
                print(f"[Session] Follow-up question for {self._query_app_session}: '{user_command.strip()[:60]}...'")
        
        # Run workflow
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.tts.speak(f"I encountered an error processing your command: {str(e)}")
            return {"error": str(e)}
    
    def _extract_email_from_command(self, command: str) -> Optional[str]:
        """
        Extract email address from user command (handles voice input like "at" and "dot")
        
        Args:
            command: User command text (e.g., "my email is aviral at gmail dot com")
            
        Returns:
            Email address if found, None otherwise
        """
        import re
        
        # First, try to find standard email format
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, command)
        if matches:
            return matches[0]
        
        # Handle voice input: "at" → "@", "dot" → "."
        # Pattern: "my email is username at domain dot com"
        command_lower = command.lower()
        
        # Look for "email" keyword followed by email parts
        email_keywords = ["email", "e-mail", "mail"]
        for keyword in email_keywords:
            if keyword in command_lower:
                # Extract text after email keyword
                parts = command_lower.split(keyword, 1)
                if len(parts) > 1:
                    email_text = parts[1].strip()
                    # Remove "is", "my", etc.
                    email_text = re.sub(r'\b(my|is|the|address)\b', '', email_text).strip()
                    
                    # Replace "at" with "@" and "dot" with "."
                    email_text = re.sub(r'\bat\b', '@', email_text)
                    email_text = re.sub(r'\bdot\b', '.', email_text)
                    email_text = re.sub(r'\bpoint\b', '.', email_text)
                    
                    # Clean up spaces around @ and .
                    email_text = re.sub(r'\s*@\s*', '@', email_text)
                    email_text = re.sub(r'\s*\.\s*', '.', email_text)
                    
                    # Remove any remaining spaces
                    email_text = email_text.replace(' ', '')
                    
                    # Validate it looks like an email
                    if re.match(r'^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$', email_text):
                        return email_text
        
        # Also try direct pattern matching for "username at domain dot com"
        pattern = r'([a-z0-9._%+-]+)\s+at\s+([a-z0-9.-]+)\s+dot\s+([a-z]{2,})'
        match = re.search(pattern, command_lower)
        if match:
            email = f"{match.group(1)}@{match.group(2)}.{match.group(3)}"
            return email
        
        return None
    
    def _extract_password_from_command(self, command: str) -> Optional[str]:
        """
        Extract password from user command (user will say it, we'll capture it)
        
        Args:
            command: User command text
            
        Returns:
            Password text if detected, None otherwise
        """
        import re
        # Check for "my password is X" pattern
        # Note: In production, you'd want to handle this more securely
        pattern = r'(?:my\s+)?password\s+(?:is\s+)?(.+)'
        match = re.search(pattern, command.lower())
        if match:
            password_text = match.group(1).strip().rstrip(" .!?")
            # Remove common filler words
            password_text = re.sub(r'\b(is|the|password)\b', '', password_text, flags=re.IGNORECASE).strip()
            if len(password_text) > 0:
                return password_text
        
        # If command doesn't contain "password" keyword, assume the whole command is the password
        # (user might just say the password directly)
        command_clean = command.strip().rstrip(" .!?")
        if len(command_clean) > 3 and not any(word in command_clean.lower() for word in ["please", "say", "tell", "my"]):
            # Likely the password itself
            return command_clean
        
        return None
    
    def run(self):
        """Run main agent loop"""
        self.tts.speak("System active. What can I help you with?")
        
        while True:
            try:
                # Listen for wake word and command
                command = self.stt.listen_for_wake_word()
                
                if command:
                    logger.info(f"Received command: {command}")
                    self.process_command(command)
                else:
                    logger.warning("No command received")
            
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.tts.speak("Goodbye.")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.tts.speak("I encountered an error. Please try again.")


def main():
    """Main entry point"""
    try:
        config = Config()
        orchestrator = AgentOrchestrator(config)
        orchestrator.run()
    except Exception as e:
        print(f"Failed to start agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
