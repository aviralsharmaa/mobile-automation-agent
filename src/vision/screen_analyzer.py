"""
Screen Analyzer - OpenAI GPT-4o vision integration for screen understanding
"""
import base64
import io
from typing import Dict, Optional, List
from PIL import Image
from openai import OpenAI
from ppadb.device import Device


class ScreenAnalyzer:
    """Screen analysis using OpenAI GPT-4o vision"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize screen analyzer
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini for speed, gpt-4o for accuracy)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.fast_model = "gpt-4o-mini"  # Use mini for faster analysis (2-5 seconds)
        self.accurate_model = "gpt-4o"  # Use full model only when needed (5-10 seconds)
    
    def capture_screenshot(self, device: Device) -> Optional[Image.Image]:
        """
        Capture screenshot from device
        
        Args:
            device: ADB device instance
            
        Returns:
            PIL Image or None if failed
        """
        try:
            screenshot_data = device.screencap()
            image = Image.open(io.BytesIO(screenshot_data))
            return image
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def analyze_screen(
        self,
        device: Device,
        prompt: Optional[str] = None,
        detect_elements: bool = True
    ) -> Dict:
        """
        Analyze screen and get description
        
        Args:
            device: ADB device instance
            prompt: Custom prompt (uses default if None)
            detect_elements: Whether to detect UI elements
            
        Returns:
            Dictionary with description and detected elements
        """
        # Capture screenshot
        screenshot = self.capture_screenshot(device)
        if not screenshot:
            return {"error": "Failed to capture screenshot"}
        
        # Convert to base64
        base64_image = self.image_to_base64(screenshot)
        
        # Default prompt for blind users
        if not prompt:
            prompt = self._get_default_prompt(detect_elements)
        
        try:
            # Use faster model (gpt-4o-mini) for most cases, unless specifically requested
            model_to_use = self.fast_model if detect_elements else self.model
            
            # Call OpenAI vision API with optimized settings
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
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
                max_tokens=800 if detect_elements else 500,  # Reduced tokens for faster response
                temperature=0.2  # Lower temperature for faster, more consistent results
            )
            
            result_text = response.choices[0].message.content
            
            # Parse response
            return self._parse_response(result_text, detect_elements)
            
        except Exception as e:
            print(f"Error analyzing screen: {e}")
            return {"error": str(e)}
    
    def analyze_screen_for_response(self, device: Device) -> Dict:
        """
        Analyze screen specifically to extract ChatGPT or AI assistant response text
        
        Args:
            device: ADB device instance
            
        Returns:
            Dictionary with extracted response text
        """
        screenshot = self.capture_screenshot(device)
        if not screenshot:
            return {"error": "Failed to capture screenshot"}
        
        base64_image = self.image_to_base64(screenshot)
        
        prompt = """I am a blind user using a ChatGPT mobile app. Analyze this screen and extract ONLY the actual response text that ChatGPT has generated.

Focus on:
- The main answer/response text displayed by ChatGPT
- Ignore UI elements, buttons, input fields, navigation bars
- Ignore the user's question (if visible)
- Extract the actual answer content that ChatGPT provided

If ChatGPT is still generating a response (typing indicator visible), indicate that the response is incomplete.

Respond with JSON format:
{
    "response_text": "The actual answer text from ChatGPT, or empty string if no response yet",
    "is_complete": true/false,
    "description": "Brief description of what's on screen"
}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
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
                max_tokens=2000,  # More tokens for longer responses
                temperature=0.2  # Very low temperature for accurate extraction
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            import re
            
            # Try to extract JSON
            json_match = re.search(r'\{[^{}]*"response_text"[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return parsed
                except:
                    pass
            
            # Fallback: return description
            return {
                "response_text": result_text,
                "is_complete": True,
                "description": result_text
            }
            
        except Exception as e:
            print(f"Error analyzing screen for response: {e}")
            return {"error": str(e)}
    
    def _get_default_prompt(self, detect_elements: bool) -> str:
        """Get default prompt for screen analysis - optimized for speed"""
        base_prompt = """Analyze this Android screen. Provide a concise description (1-2 sentences).

Focus on:
- Main content/purpose
- Important buttons: "Continue", "Get Started", "Next", "Start", "Accept", "OK", "Skip", "Sign In", "Login"
- Input fields (text fields, search bars)
- Login screen? Stage: "initial" (both fields), "email_only", "password_only", "other", "none"
- Popups/alerts
- PRIMARY action button

"""
        
        if detect_elements:
            base_prompt += """
Identify clickable elements with EXACT center coordinates (x, y) for 1080x2400 screen:
- Element type: "button", "text_field", "icon", "other"
- Description and coordinates
- Prioritize: "Continue", "Get Started", "Login", "Sign In", arrow icons (→, ↑)
- Y coordinate: 150-250 for top-right buttons (NOT 100-120 - that's status bar)
- Coordinates are CENTER of element
"""
        
        base_prompt += """
Respond in JSON format (be concise):
{
    "description": "Brief description",
    "is_login_screen": true/false,
    "login_stage": "initial|email_only|password_only|other|none",
    "has_email_field": true/false,
    "has_password_field": true/false,
    "has_popup": true/false,
    "primary_action": "Primary button description",
    "primary_action_type": "button|text_field|link|other",
    "elements": [
        {"description": "element", "x": 100, "y": 200, "type": "button|text_field|icon|other", "text": "text if visible"}
    ]
}
"""
        
        return base_prompt
    
    def _parse_response(self, response_text: str, detect_elements: bool) -> Dict:
        """
        Parse GPT-4o response
        
        Args:
            response_text: Response from GPT-4o
            detect_elements: Whether elements were requested
            
        Returns:
            Parsed dictionary
        """
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return parsed
            except json.JSONDecodeError:
                pass
        
        # Fallback: return text description
        text_lower = response_text.lower()
        is_login = "login" in text_lower or "sign in" in text_lower
        has_password = "password" in text_lower and ("field" in text_lower or "enter" in text_lower)
        has_email = "email" in text_lower and ("field" in text_lower or "enter" in text_lower)
        
        # Determine login stage
        login_stage = "none"
        if is_login:
            if has_password and not has_email:
                login_stage = "password_only"
            elif has_email and not has_password:
                login_stage = "email_only"
            elif has_email and has_password:
                login_stage = "initial"
            else:
                login_stage = "other"
        
        return {
            "description": response_text,
            "is_login_screen": is_login,
            "login_stage": login_stage,
            "has_email_field": has_email,
            "has_password_field": has_password,
            "has_popup": "popup" in text_lower or "alert" in text_lower,
            "elements": []
        }
    
    def detect_login_screen(self, device: Device) -> bool:
        """
        Quickly detect if current screen is a login screen
        
        Args:
            device: ADB device instance
            
        Returns:
            True if login screen detected
        """
        result = self.analyze_screen(device, detect_elements=False)
        return result.get("is_login_screen", False)
    
    def detect_popup(self, device: Device) -> bool:
        """
        Detect if there's a popup or alert
        
        Args:
            device: ADB device instance
            
        Returns:
            True if popup detected
        """
        result = self.analyze_screen(device, detect_elements=False)
        return result.get("has_popup", False)
