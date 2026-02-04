"""
Element Detector - UI element grounding and coordinate extraction
"""
from typing import Dict, List, Optional, Tuple
from src.vision.screen_analyzer import ScreenAnalyzer
from ppadb.device import Device


class ElementDetector:
    """UI element detection and coordinate mapping"""
    
    def __init__(self, screen_analyzer: ScreenAnalyzer):
        """
        Initialize element detector
        
        Args:
            screen_analyzer: ScreenAnalyzer instance
        """
        self.screen_analyzer = screen_analyzer
    
    def find_element(
        self,
        device: Device,
        description: str,
        element_type: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Find element by description and return coordinates
        
        Args:
            device: ADB device instance
            description: Description of element to find (e.g., "login button", "search bar")
            element_type: Optional element type filter (button, text_field, etc.)
            
        Returns:
            Tuple of (x, y) coordinates or None if not found
        """
        # Analyze screen with element detection
        result = self.screen_analyzer.analyze_screen(device, detect_elements=True)
        
        if "error" in result:
            return None
        
        elements = result.get("elements", [])
        
        # Search for matching element
        description_lower = description.lower()
        for element in elements:
            elem_desc = element.get("description", "").lower()
            elem_type = element.get("type", "")
            
            # Check if description matches
            if description_lower in elem_desc or elem_desc in description_lower:
                # Check type filter if specified
                if element_type and elem_type != element_type:
                    continue
                
                x = element.get("x", 0)
                y = element.get("y", 0)
                
                if x > 0 and y > 0:
                    return (x, y)
        
        return None
    
    def find_all_elements(
        self,
        device: Device,
        element_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Get all detected elements on screen
        
        Args:
            device: ADB device instance
            element_type: Optional filter by type
            
        Returns:
            List of element dictionaries
        """
        result = self.screen_analyzer.analyze_screen(device, detect_elements=True)
        
        if "error" in result:
            return []
        
        elements = result.get("elements", [])
        
        if element_type:
            elements = [e for e in elements if e.get("type") == element_type]
        
        return elements
    
    def find_text_field(self, device: Device, field_description: Optional[str] = None) -> Optional[Tuple[int, int]]:
        """
        Find text input field
        
        Args:
            device: ADB device instance
            field_description: Optional description (e.g., "email", "password", "search")
            
        Returns:
            Coordinates of text field or None
        """
        if field_description:
            return self.find_element(device, field_description, element_type="text_field")
        else:
            # Find first text field
            elements = self.find_all_elements(device, element_type="text_field")
            if elements:
                return (elements[0].get("x", 0), elements[0].get("y", 0))
        return None
    
    def find_button(
        self,
        device: Device,
        button_text: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find button by text/description
        
        Args:
            device: ADB device instance
            button_text: Button text or description
            
        Returns:
            Coordinates of button or None
        """
        return self.find_element(device, button_text, element_type="button")
    
    def get_screen_description(self, device: Device) -> str:
        """
        Get text description of current screen
        
        Args:
            device: ADB device instance
            
        Returns:
            Screen description text
        """
        result = self.screen_analyzer.analyze_screen(device, detect_elements=False)
        return result.get("description", "Unable to analyze screen")
    
    def find_element_by_hierarchy(
        self,
        device: Device,
        position: str,
        element_type: Optional[str] = None
    ) -> Optional[Tuple[int, int]]:
        """
        Find element by position in hierarchy (e.g., "first", "second", "top", "bottom")
        
        Args:
            device: ADB device instance
            position: Position description (e.g., "first button", "top email", "last item")
            element_type: Optional element type filter
            
        Returns:
            Coordinates of element or None
        """
        elements = self.find_all_elements(device, element_type)
        
        if not elements:
            return None
        
        position_lower = position.lower()
        
        # Sort by Y coordinate (top to bottom)
        sorted_elements = sorted(elements, key=lambda e: e.get("y", 0))
        
        if "first" in position_lower or "top" in position_lower:
            elem = sorted_elements[0]
        elif "last" in position_lower or "bottom" in position_lower:
            elem = sorted_elements[-1]
        elif "second" in position_lower:
            if len(sorted_elements) > 1:
                elem = sorted_elements[1]
            else:
                return None
        else:
            # Default to first
            elem = sorted_elements[0]
        
        return (elem.get("x", 0), elem.get("y", 0))
