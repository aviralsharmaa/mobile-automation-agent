"""
Accessibility Tree Parser - Get accurate UI element coordinates using uiautomator
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from ppadb.device import Device


class AccessibilityTree:
    """Parse Android accessibility tree for accurate element coordinates"""
    
    def __init__(self, device: Device):
        """
        Initialize accessibility tree parser
        
        Args:
            device: ADB device instance
        """
        self.device = device
    
    def get_tree(self) -> Optional[str]:
        """
        Get accessibility tree dump
        
        Returns:
            XML string of accessibility tree or None
        """
        try:
            # Get UI hierarchy dump
            tree = self.device.shell("uiautomator dump /dev/tty")
            return tree
        except Exception as e:
            print(f"Error getting accessibility tree: {e}")
            return None
    
    def get_tree_file(self) -> Optional[str]:
        """
        Get accessibility tree from file (more reliable)
        
        Returns:
            XML string of accessibility tree or None
        """
        try:
            # Dump to file first
            self.device.shell("uiautomator dump /sdcard/window_dump.xml")
            # Read the file
            tree = self.device.shell("cat /sdcard/window_dump.xml")
            return tree
        except Exception as e:
            print(f"Error getting accessibility tree from file: {e}")
            return None
    
    def find_element_by_text(
        self,
        text: str,
        element_type: Optional[str] = None
    ) -> Optional[Tuple[int, int, Dict]]:
        """
        Find element by text using accessibility tree
        
        Args:
            text: Text to search for (e.g., "Continue", "Get Started")
            element_type: Optional type filter ("button", "clickable", etc.)
            
        Returns:
            Tuple of (x, y, bounds_dict) or None if not found
        """
        tree = self.get_tree_file()
        if not tree:
            return None
        
        text_lower = text.lower()
        
        try:
            # Parse XML to find element with matching text
            # Look for pattern like: <node text="Continue" bounds="[540,1800][1080,2000]" clickable="true"/>
            pattern = rf'<node[^>]*text=["\']([^"\']*{re.escape(text)}[^"\']*)["\'][^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\']'
            
            matches = re.finditer(pattern, tree, re.IGNORECASE)
            
            for match in matches:
                found_text = match.group(1)
                x1 = int(match.group(2))
                y1 = int(match.group(3))
                x2 = int(match.group(4))
                y2 = int(match.group(5))
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check if clickable
                node_text = match.group(0)
                is_clickable = 'clickable="true"' in node_text.lower()
                
                if element_type == "button" and not is_clickable:
                    continue
                
                return (center_x, center_y, {
                    "text": found_text,
                    "bounds": [x1, y1, x2, y2],
                    "center": [center_x, center_y]
                })
            
            # Try case-insensitive search
            pattern_lower = rf'<node[^>]*text=["\']([^"\']*{re.escape(text_lower)}[^"\']*)["\'][^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\']'
            matches = re.finditer(pattern_lower, tree, re.IGNORECASE)
            
            for match in matches:
                found_text = match.group(1)
                x1 = int(match.group(2))
                y1 = int(match.group(3))
                x2 = int(match.group(4))
                y2 = int(match.group(5))
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                node_text = match.group(0)
                is_clickable = 'clickable="true"' in node_text.lower()
                
                if element_type == "button" and not is_clickable:
                    continue
                
                return (center_x, center_y, {
                    "text": found_text,
                    "bounds": [x1, y1, x2, y2],
                    "center": [center_x, center_y]
                })
            
        except Exception as e:
            print(f"Error parsing accessibility tree: {e}")
        
        return None
    
    def find_clickable_elements(self) -> List[Dict]:
        """
        Find all clickable elements in accessibility tree with full attribute extraction
        
        Returns:
            List of element dictionaries with coordinates and attributes
        """
        tree = self.get_tree_file()
        if not tree:
            return []
        
        elements = []
        
        try:
            # Find all clickable nodes - improved pattern to capture full node
            pattern = r'<node[^>]*clickable="true"[^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\'][^>]*>'
            
            matches = re.finditer(pattern, tree, re.IGNORECASE)
            
            for match in matches:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Extract all available attributes
                node_text = match.group(0)
                
                # Extract text
                text_match = re.search(r'text=["\']([^"\']*)["\']', node_text)
                text = text_match.group(1) if text_match else ""
                
                # Extract content-desc (for icon buttons)
                desc_match = re.search(r'content-desc=["\']([^"\']*)["\']', node_text)
                content_desc = desc_match.group(1) if desc_match else ""
                
                # Extract class/type
                class_match = re.search(r'class=["\']([^"\']*)["\']', node_text)
                element_class = class_match.group(1) if class_match else ""
                
                # Extract resource-id
                resource_id_match = re.search(r'resource-id=["\']([^"\']*)["\']', node_text)
                resource_id = resource_id_match.group(1) if resource_id_match else ""
                
                # Extract package name
                package_match = re.search(r'package=["\']([^"\']*)["\']', node_text)
                package = package_match.group(1) if package_match else ""
                
                elements.append({
                    "text": text,
                    "content_desc": content_desc,
                    "class": element_class,
                    "resource_id": resource_id,
                    "package": package,
                    "bounds": [x1, y1, x2, y2],
                    "center": [center_x, center_y],
                    "x": center_x,
                    "y": center_y,
                    "width": x2 - x1,
                    "height": y2 - y1
                })
        
        except Exception as e:
            print(f"[Accessibility] Error parsing clickable elements: {e}")
            import traceback
            traceback.print_exc()
        
        return elements
    
    def is_keyboard_visible(self) -> bool:
        """
        Detect if the soft keyboard (IME) is currently visible.
        Uses dumpsys input_method and looks for mInputShown=true.
        
        Returns:
            True if keyboard is open, False otherwise
        """
        try:
            out = self.device.shell("dumpsys input_method")
            if out and "mInputShown=true" in out:
                return True
            # Some devices use mShowRequested or other flags
            if out and "mInputShown=false" in out:
                return False
            return False
        except Exception as e:
            print(f"[Accessibility] Keyboard check failed: {e}")
            return False
    
    def find_input_field(self) -> Optional[Tuple[int, int, Dict]]:
        """
        Find the best text input field (EditText) for chat/search (e.g. "Ask ChatGPT").
        Uses two regions based on keyboard state:
        - Keyboard OPEN:  X 300-800, Y 1200-1800 (field above keyboard, ~608,1427)
        - Keyboard CLOSED: X 300-800, Y 2000-2500 (field at bottom ~540,2400)
        
        Returns:
            Tuple of (center_x, center_y, element_info) or None
        """
        tree = self.get_tree_file()
        if not tree:
            return None
        keyboard_open = self.is_keyboard_visible()
        region_x_min, region_x_max = 300, 800
        if keyboard_open:
            region_y_min, region_y_max = 1200, 1800
            target_y = 1500
            print("[Accessibility] Keyboard open: using input region Y 1200-1800")
        else:
            region_y_min, region_y_max = 2000, 2500
            target_y = 2200
            print("[Accessibility] Keyboard closed: using input region Y 2000-2500 (bottom)")
        # Match any node with bounds (input fields may be clickable or focusable)
        pattern = r'<node[^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\'][^>]*>'
        candidates = []
        try:
            for match in re.finditer(pattern, tree, re.IGNORECASE):
                x1, y1 = int(match.group(1)), int(match.group(2))
                x2, y2 = int(match.group(3)), int(match.group(4))
                node_tag = match.group(0)
                class_match = re.search(r'class=["\']([^"\']*)["\']', node_tag)
                class_name = (class_match.group(1) or "").lower()
                is_input = any(t in class_name for t in [
                    "edittext", "textfield", "textinput", "edit", "editabletext",
                    "autocomplete", "searchview", "input"
                ])
                if not is_input:
                    continue
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                if width < 50 or height < 30:  # skip tiny elements
                    continue
                hint = ""
                hint_m = re.search(r'hint=["\']([^"\']*)["\']', node_tag)
                if hint_m:
                    hint = hint_m.group(1)
                text = ""
                text_m = re.search(r'text=["\']([^"\']*)["\']', node_tag)
                if text_m:
                    text = text_m.group(1)
                candidates.append({
                    "x": center_x, "y": center_y,
                    "bounds": [x1, y1, x2, y2],
                    "width": width, "height": height,
                    "class": class_name, "hint": hint, "text": text,
                })
            if not candidates:
                return None
            in_region = [c for c in candidates if region_x_min <= c["x"] <= region_x_max and region_y_min <= c["y"] <= region_y_max]
            if not in_region:
                # Try the other region (keyboard state may be wrong)
                other_y_min, other_y_max = (2000, 2500) if keyboard_open else (1200, 1800)
                other_target = 2200 if keyboard_open else 1500
                in_region = [c for c in candidates if region_x_min <= c["x"] <= region_x_max and other_y_min <= c["y"] <= other_y_max]
                if in_region:
                    target_y = other_target
                    print(f"[Accessibility] Using alternate region Y {other_y_min}-{other_y_max}")
            if not in_region:
                return None
            candidates = in_region
            def dist(e):
                return (e["x"] - 540) ** 2 + (e["y"] - target_y) ** 2
            candidates.sort(key=dist)
            best = candidates[0]
            print(f"[Accessibility] Found input field at ({best['x']}, {best['y']}), hint: '{best.get('hint', '')}'")
            return (best["x"], best["y"], best)
        except Exception as e:
            print(f"[Accessibility] Error in find_input_field: {e}")
            return None
    
    def find_send_button_near_input(self, input_x: int, input_y: int) -> Optional[Tuple[int, int, Dict]]:
        """
        Find the send button just after the input field (e.g. ChatGPT's upward-arrow send icon).
        Prefers elements whose content-desc or text contains upward arrow (↑), "send", or "submit".
        
        Args:
            input_x: Center X of the input field
            input_y: Center Y of the input field
            
        Returns:
            (x, y, element_info) or None
        """
        elements = self.find_clickable_elements()
        if not elements:
            return None
        # Upward-arrow / send indicators (ChatGPT uses ↑ for send)
        send_indicators = ("↑", "send", "submit", "arrow", "up")
        candidates = []
        for elem in elements:
            x = elem.get("x", 0)
            y = elem.get("y", 0)
            bounds = elem.get("bounds", [])
            if x <= 0 or y <= 0:
                continue
            # Must be to the right of input (just after it), same row
            if x <= input_x + 20:
                continue
            if x - input_x > 450:
                continue
            if abs(y - input_y) > 120:
                continue
            w = bounds[2] - bounds[0] if len(bounds) >= 4 else 80
            h = bounds[3] - bounds[1] if len(bounds) >= 4 else 80
            if w < 20 or w > 180 or h < 20 or h > 180:
                continue
            raw_desc = elem.get("content_desc") or ""
            raw_text = elem.get("text") or ""
            desc = raw_desc.lower()
            text = raw_text.lower()
            combined = desc + " " + text
            has_send_indicator = any(ind in combined for ind in ("send", "submit", "arrow", "up"))
            # Prefer upward arrow (unicode ↑) - check raw strings
            has_up_arrow = "↑" in raw_desc or "↑" in raw_text
            distance = (x - input_x) ** 2 + (y - input_y) ** 2
            candidates.append({
                "x": x, "y": y,
                "distance": distance,
                "has_up_arrow": has_up_arrow,
                "has_send_indicator": has_send_indicator,
                "desc": (raw_desc or raw_text or "").strip() or "(no desc)",
            })
        if candidates:
            # Prefer: 1) has ↑, 2) has send/submit/arrow/up, 3) closest to input
            candidates.sort(key=lambda c: (
                not c["has_up_arrow"],
                not c["has_send_indicator"],
                c["distance"],
            ))
            best = candidates[0]
            print(f"[Accessibility] Send button near input: ({best['x']}, {best['y']}), desc: '{best['desc'][:40]}'")
            return (best["x"], best["y"], best)
        
        # Known-good fallback: ChatGPT send button (upward arrow) at (990, 1427)
        CHATGPT_SEND_BUTTON = (990, 1427)
        print(f"[Accessibility] Using known send button coordinates: {CHATGPT_SEND_BUTTON}")
        return (CHATGPT_SEND_BUTTON[0], CHATGPT_SEND_BUTTON[1], {"x": CHATGPT_SEND_BUTTON[0], "y": CHATGPT_SEND_BUTTON[1], "desc": "send (fallback)"})
    
    # WhatsApp green circle send button: right of input, X=1080, Y between 1200-1800
    WHATSAPP_SEND_REGION = (1000, 1080, 1200, 1800)  # x_min, x_max, y_min, y_max
    WHATSAPP_SEND_FALLBACK = (1080, 1500)
    
    def find_whatsapp_send_button(self) -> Optional[Tuple[int, int, Dict]]:
        """
        Find the green circle send button in WhatsApp (right of input field).
        Region: X 1000-1080, Y 1200-1800. Do not use Enter - it does not send.
        
        Returns:
            (x, y, info) or fallback (1080, 1500)
        """
        elements = self.find_clickable_elements()
        x_min, x_max, y_min, y_max = self.WHATSAPP_SEND_REGION
        candidates = []
        for elem in elements:
            x, y = elem.get("x", 0), elem.get("y", 0)
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            if "edittext" in (elem.get("class") or "").lower():
                continue
            desc = (elem.get("content_desc") or "").lower()
            cls = (elem.get("class") or "").lower()
            priority = 4 if "send" in desc else (3 if "image" in cls or "button" in cls else 2)
            candidates.append({"x": x, "y": y, "priority": priority})
        if candidates:
            candidates.sort(key=lambda c: (-c["priority"], -c["x"]))
            best = candidates[0]
            print(f"[Accessibility] WhatsApp send button at ({best['x']}, {best['y']})")
            return (best["x"], best["y"], best)
        print(f"[Accessibility] WhatsApp send fallback: {self.WHATSAPP_SEND_FALLBACK}")
        fx, fy = self.WHATSAPP_SEND_FALLBACK
        return (fx, fy, {"x": fx, "y": fy, "desc": "send (fallback)"})
    
    def find_node_near_region(
        self,
        region_x: int,
        region_y: int,
        region_width: Optional[int] = None,
        region_height: Optional[int] = None,
        search_radius: int = 100
    ) -> Optional[Tuple[int, int, Dict]]:
        """
        Find the nearest clickable Accessibility node within a Vision-provided region.
        This implements the Vision → Accessibility hybrid approach.
        
        Args:
            region_x: X coordinate of region center (from Vision)
            region_y: Y coordinate of region center (from Vision)
            region_width: Optional width of region (if Vision provided bounding box)
            region_height: Optional height of region (if Vision provided bounding box)
            search_radius: Maximum distance to search from region center (default 100px)
            
        Returns:
            Tuple of (x, y, element_dict) of the nearest clickable node, or None if not found
        """
        all_clickable = self.find_clickable_elements()
        if not all_clickable:
            return None
        
        best_match = None
        min_distance = float('inf')
        
        # If Vision provided a bounding box, check if nodes are within it
        if region_width and region_height:
            region_x1 = region_x - region_width // 2
            region_y1 = region_y - region_height // 2
            region_x2 = region_x + region_width // 2
            region_y2 = region_y + region_height // 2
            
            for elem in all_clickable:
                elem_x = elem.get("x", 0)
                elem_y = elem.get("y", 0)
                bounds = elem.get("bounds", [])
                
                if len(bounds) >= 4:
                    elem_x1, elem_y1, elem_x2, elem_y2 = bounds[0], bounds[1], bounds[2], bounds[3]
                    
                    # Check if node overlaps with Vision region
                    overlaps = not (elem_x2 < region_x1 or elem_x1 > region_x2 or 
                                   elem_y2 < region_y1 or elem_y1 > region_y2)
                    
                    if overlaps:
                        # Calculate distance from Vision center to node center
                        distance = ((elem_x - region_x) ** 2 + (elem_y - region_y) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            best_match = elem
        
        # If no node found in bounding box, search by radius
        if not best_match:
            for elem in all_clickable:
                elem_x = elem.get("x", 0)
                elem_y = elem.get("y", 0)
                
                # Calculate distance from Vision center
                distance = ((elem_x - region_x) ** 2 + (elem_y - region_y) ** 2) ** 0.5
                
                if distance <= search_radius and distance < min_distance:
                    min_distance = distance
                    best_match = elem
        
        if best_match:
            x = best_match.get("x", 0)
            y = best_match.get("y", 0)
            print(f"[Accessibility] Found nearest node at ({x}, {y}), distance: {min_distance:.1f}px from Vision region ({region_x}, {region_y})")
            return (x, y, best_match)
        
        return None
    
    def find_real_login_button(self, keywords: List[str] = None) -> Optional[Tuple[int, int, Dict]]:
        """
        Find the REAL clickable login button using the "Bottom Sheet Button Rule".
        
        Strategy: Find the LOWEST large clickable button on screen (bottom sheet pattern).
        This works across all devices/Android versions without text matching.
        
        Rules:
        - Must be clickable=true
        - Must be reasonably large (width > 300px, height > 80px) - real UI buttons
        - Returns the button with LOWEST Y coordinate (bottom-most on screen)
        - NO text matching needed - works in any language
        
        Args:
            keywords: Ignored (kept for compatibility, but not used)
            
        Returns:
            Tuple of (x, y, element_dict) of the bottom-most large clickable button, or None
        """
        tree = self.get_tree_file()
        if not tree:
            print("[Accessibility] Tree file is None")
            return None
        
        clickable_buttons = []
        
        try:
            # Pattern to match ALL clickable nodes with bounds (no text requirement)
            # Format: <node ... clickable="true" ... bounds="[x1,y1][x2,y2]" ...>
            # We need to handle cases where attributes can be in any order
            pattern = r'<node[^>]*clickable="true"[^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\']'
            matches = re.finditer(pattern, tree, re.IGNORECASE)
            
            for match in matches:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                
                # Calculate dimensions
                width = x2 - x1
                height = y2 - y1
                
                # Only keep reasonably large elements (real UI buttons, not small icons)
                # Typical bottom sheet buttons: width 300-1000px, height 80-150px
                if width > 300 and height > 80:
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Try to extract class name and text for logging (optional)
                    node_start = match.start()
                    node_end = match.end()
                    node_tag = tree[node_start:node_start + 500]  # Get node tag for parsing
                    
                    class_name = ""
                    text_content = ""
                    
                    # Extract class if available
                    class_match = re.search(r'class=["\']([^"\']*)["\']', node_tag)
                    if class_match:
                        class_name = class_match.group(1)
                    
                    # Extract text if available (for logging only)
                    text_match = re.search(r'text=["\']([^"\']*)["\']', node_tag)
                    if text_match:
                        text_content = text_match.group(1)
                    
                    # CRITICAL: Skip input fields - they are NOT buttons!
                    # EditText, TextField, TextInput are input fields, not clickable buttons
                    class_lower = class_name.lower()
                    is_input_field = any(input_type in class_lower for input_type in [
                        "edittext", "textfield", "textinput", "edit", "input", 
                        "autocomplete", "searchview", "editabletext"
                    ])
                    
                    if is_input_field:
                        print(f"[Accessibility]   SKIPPING input field at ({center_x}, {center_y}), class: {class_name}")
                        continue  # Skip input fields
                    
                    clickable_buttons.append({
                        "x": center_x,
                        "y": center_y,
                        "bounds": [x1, y1, x2, y2],
                        "width": width,
                        "height": height,
                        "text": text_content,
                        "class": class_name
                    })
            
            if clickable_buttons:
                print(f"[Accessibility] Found {len(clickable_buttons)} large clickable buttons")
                # Sort by Y coordinate (ascending) to show in order from top to bottom
                clickable_buttons.sort(key=lambda b: b["y"])
                for i, btn in enumerate(clickable_buttons):
                    print(f"[Accessibility]   {i+1}. Button at ({btn['x']}, {btn['y']}), size: {btn['width']}x{btn['height']}, text: '{btn['text'][:50]}'")
                
                # Pick the FIRST/TOP button (lowest Y coordinate) - this is "Continue with Google"
                best_button = clickable_buttons[0]  # First = topmost = Continue with Google
                
                print(f"[Accessibility] ✓ Selected FIRST/TOP button (Continue with Google pattern):")
                print(f"[Accessibility]   Position: ({best_button['x']}, {best_button['y']})")
                print(f"[Accessibility]   Size: {best_button['width']}x{best_button['height']}")
                print(f"[Accessibility]   Text: '{best_button['text'][:50] if best_button['text'] else 'N/A'}'")
                print(f"[Accessibility]   Class: {best_button['class']}")
                
                return (best_button["x"], best_button["y"], best_button)
            else:
                print(f"[Accessibility] No large clickable buttons found (need width > 300px, height > 80px)")
                return None
                
        except Exception as e:
            print(f"[Accessibility] Error finding real login button: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_button_below_y(self, min_y: int, min_width: int = 300, min_height: int = 80) -> Optional[Tuple[int, int, Dict]]:
        """
        Find the FIRST large clickable button BELOW a given Y coordinate.
        Uses structure/position detection like find_real_login_button.
        
        Args:
            min_y: Minimum Y coordinate (button must be below this)
            min_width: Minimum button width in pixels
            min_height: Minimum button height in pixels
            
        Returns:
            Tuple of (x, y, element_dict) of the first large button below min_y, or None
        """
        tree = self.get_tree_file()
        if not tree:
            print("[Accessibility] Tree file is None")
            return None
        
        clickable_buttons = []
        
        try:
            # Pattern to match ALL clickable nodes with bounds
            pattern = r'<node[^>]*clickable="true"[^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\']'
            matches = re.finditer(pattern, tree, re.IGNORECASE)
            
            for match in matches:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                
                # Calculate dimensions
                width = x2 - x1
                height = y2 - y1
                center_y = (y1 + y2) // 2
                
                # Must be BELOW min_y (at least 50px below)
                if center_y <= min_y + 50:
                    continue
                
                # Only keep reasonably large elements
                if width >= min_width and height >= min_height:
                    center_x = (x1 + x2) // 2
                    
                    # Extract class and text
                    node_start = match.start()
                    node_tag = tree[node_start:node_start + 500]
                    
                    class_name = ""
                    text_content = ""
                    
                    class_match = re.search(r'class=["\']([^"\']*)["\']', node_tag)
                    if class_match:
                        class_name = class_match.group(1)
                    
                    text_match = re.search(r'text=["\']([^"\']*)["\']', node_tag)
                    if text_match:
                        text_content = text_match.group(1)
                    
                    # Skip input fields (EditText)
                    class_lower = class_name.lower()
                    if any(input_type in class_lower for input_type in ["edittext", "textfield", "edit", "input"]):
                        continue
                    
                    clickable_buttons.append({
                        "x": center_x,
                        "y": center_y,
                        "bounds": [x1, y1, x2, y2],
                        "width": width,
                        "height": height,
                        "text": text_content,
                        "class": class_name
                    })
            
            if clickable_buttons:
                # Sort by Y and return the FIRST (closest to input field)
                clickable_buttons.sort(key=lambda b: b["y"])
                
                print(f"[Accessibility] Found {len(clickable_buttons)} buttons below Y={min_y}:")
                for i, btn in enumerate(clickable_buttons[:3]):  # Show first 3
                    print(f"[Accessibility]   {i+1}. Button at ({btn['x']}, {btn['y']}), size: {btn['width']}x{btn['height']}")
                
                best_button = clickable_buttons[0]  # First = closest to field
                return (best_button["x"], best_button["y"], best_button)
            else:
                print(f"[Accessibility] No large clickable buttons found below Y={min_y}")
                return None
                
        except Exception as e:
            print(f"[Accessibility] Error finding button below Y: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_google_signin_button(self) -> Optional[Tuple[int, int, Dict]]:
        """
        Find "Continue with Google" button using Accessibility tree.
        NO LLM needed - pure text/content_desc matching.
        
        Returns:
            Tuple of (x, y, element_dict) or None
        """
        tree = self.get_tree_file()
        if not tree:
            print("[Accessibility] Tree file is None")
            return None
        
        print("[Accessibility] Searching for 'Continue with Google' button...")
        
        try:
            # Pattern to find clickable nodes with bounds
            pattern = r'<node[^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\'][^>]*'
            
            # Also search for text or content-desc containing "google" or "continue with google"
            google_keywords = ["continue with google", "sign in with google", "google"]
            
            candidates = []
            
            for match in re.finditer(pattern, tree, re.IGNORECASE):
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                
                # Get the full node string for analysis
                node_start = match.start()
                node_str = tree[node_start:node_start + 800]
                
                # Check if clickable
                if 'clickable="true"' not in node_str:
                    continue
                
                # Extract text and content-desc
                text_match = re.search(r'text=["\']([^"\']*)["\']', node_str)
                desc_match = re.search(r'content-desc=["\']([^"\']*)["\']', node_str)
                
                text = text_match.group(1).lower() if text_match else ""
                desc = desc_match.group(1).lower() if desc_match else ""
                combined = text + " " + desc
                
                # Check for Google sign-in keywords
                if any(kw in combined for kw in google_keywords):
                    width = x2 - x1
                    height = y2 - y1
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Must be reasonably sized button
                    if width > 200 and height > 40:
                        candidates.append({
                            "x": center_x,
                            "y": center_y,
                            "bounds": [x1, y1, x2, y2],
                            "width": width,
                            "height": height,
                            "text": text_match.group(1) if text_match else "",
                            "desc": desc_match.group(1) if desc_match else ""
                        })
            
            if candidates:
                # Sort by Y (pick the first/highest on bottom sheet - usually the Google button)
                candidates.sort(key=lambda c: c["y"])
                best = candidates[0]
                print(f"[Accessibility] ✓ Found 'Continue with Google' at ({best['x']}, {best['y']})")
                print(f"[Accessibility]   Text: '{best['text']}'")
                print(f"[Accessibility]   Size: {best['width']}x{best['height']}")
                return (best["x"], best["y"], best)
            
            print("[Accessibility] 'Continue with Google' button not found")
            return None
            
        except Exception as e:
            print(f"[Accessibility] Error finding Google button: {e}")
            return None
    
    def find_continue_button(self) -> Optional[Tuple[int, int, Dict]]:
        """
        Find "Continue" button on Google account selection popup.
        Uses POSITION-BASED detection since text may not be in accessibility tree.
        
        Strategy:
        1. First try to find button with "Continue" text
        2. If not found, find ANY clickable button in the popup region (Y > 400)
        3. Prefer buttons that are centered horizontally and in lower portion
        
        Returns:
            Tuple of (x, y, element_dict) or None
        """
        tree = self.get_tree_file()
        if not tree:
            print("[Accessibility] Tree file is None")
            return None
        
        print("[Accessibility] Searching for 'Continue' button (with position fallback)...")
        
        try:
            # Pattern to find clickable nodes with bounds
            pattern = r'<node[^>]*clickable="true"[^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\']'
            
            text_matches = []  # Buttons with "Continue" text
            position_matches = []  # Any button in the right position
            
            for match in re.finditer(pattern, tree, re.IGNORECASE):
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Skip tiny elements
                if width < 80 or height < 30:
                    continue
                
                # Get node content for text matching
                node_start = match.start()
                node_str = tree[node_start:node_start + 800]
                
                # Extract text and content-desc
                text_match_inner = re.search(r'text=["\']([^"\']*)["\']', node_str)
                desc_match = re.search(r'content-desc=["\']([^"\']*)["\']', node_str)
                class_match = re.search(r'class=["\']([^"\']*)["\']', node_str)
                
                text = text_match_inner.group(1) if text_match_inner else ""
                desc = desc_match.group(1) if desc_match else ""
                class_name = class_match.group(1) if class_match else ""
                
                text_lower = text.lower()
                desc_lower = desc.lower()
                combined = text_lower + " " + desc_lower
                
                # Skip input fields
                if any(inp in class_name.lower() for inp in ["edittext", "edit", "input"]):
                    continue
                
                button_info = {
                    "x": center_x,
                    "y": center_y,
                    "bounds": [x1, y1, x2, y2],
                    "width": width,
                    "height": height,
                    "text": text,
                    "desc": desc,
                    "class": class_name
                }
                
                # METHOD 1: Look for "Continue" text
                continue_keywords = ["continue", "next", "proceed", "ok", "confirm", "done"]
                if any(kw in combined for kw in continue_keywords):
                    text_matches.append(button_info)
                    print(f"[Accessibility]   Found text match: '{text}' at ({center_x}, {center_y})")
                
                # METHOD 2: Position-based - look for button in popup area
                # Google popup Continue button is typically:
                # - In lower portion of popup (Y > 350)
                # - Reasonably sized (not a tiny icon)
                # - Can be small width if it's just the "Continue" button
                if center_y > 350 and width > 80 and height > 35:
                    position_matches.append(button_info)
            
            # PRIORITY 1: Use text matches if found
            if text_matches:
                # Sort by Y (prefer lower buttons)
                text_matches.sort(key=lambda c: -c["y"])
                best = text_matches[0]
                print(f"[Accessibility] ✓ Found 'Continue' button (text match) at ({best['x']}, {best['y']})")
                print(f"[Accessibility]   Text: '{best['text']}'")
                print(f"[Accessibility]   Size: {best['width']}x{best['height']}")
                return (best["x"], best["y"], best)
            
            # PRIORITY 2: Use position-based matches (any button in popup area)
            if position_matches:
                print(f"[Accessibility] No text match found, using position-based detection...")
                print(f"[Accessibility] Found {len(position_matches)} buttons in popup area:")
                for i, btn in enumerate(position_matches[:5]):
                    print(f"[Accessibility]   {i+1}. ({btn['x']}, {btn['y']}), size: {btn['width']}x{btn['height']}, text: '{btn['text'][:20] if btn['text'] else 'N/A'}'")
                
                # Sort by: RIGHT-BOTTOM priority
                # 1. Higher Y = lower on screen (bottom)
                # 2. Higher X = right side
                # Google "Continue" button is typically at right-bottom of popup
                position_matches.sort(key=lambda c: (-c["y"], -c["x"]))
                best = position_matches[0]
                
                print(f"[Accessibility] ✓ Selected button at ({best['x']}, {best['y']}) - RIGHT-BOTTOM of popup")
                print(f"[Accessibility]   Size: {best['width']}x{best['height']}")
                return (best["x"], best["y"], best)
            
            print("[Accessibility] 'Continue' button not found (no text or position matches)")
            return None
            
        except Exception as e:
            print(f"[Accessibility] Error finding Continue button: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_button_by_keywords(self, keywords: List[str]) -> Optional[Tuple[int, int, Dict]]:
        """
        Find button matching any of the keywords - improved to find buttons even without text
        
        Args:
            keywords: List of keywords to search for (e.g., ["login", "sign in", "log in", "Log in", "Sign in", "Sign In", "Sign in", "Sign In"])
            
        Returns:
            Tuple of (x, y, element_dict) or None
        """
        tree = self.get_tree_file()
        if not tree:
            print("[Accessibility] Tree file is None")
            return None
        
        keywords_lower = [k.lower() for k in keywords]
        print(f"[Accessibility] Searching for buttons with keywords: {keywords}")
        
        try:
            # METHOD 1: Find clickable nodes with text attribute (direct)
            pattern_with_text = r'<node[^>]*clickable="true"[^>]*text=["\']([^"\']*)["\'][^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\']'
            
            matches = re.finditer(pattern_with_text, tree, re.IGNORECASE)
            
            for match in matches:
                text = match.group(1).lower()
                if text and any(keyword in text for keyword in keywords_lower):
                    x1 = int(match.group(2))
                    y1 = int(match.group(3))
                    x2 = int(match.group(4))
                    y2 = int(match.group(5))
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    print(f"[Accessibility] Found button by direct text: '{match.group(1)}' at ({center_x}, {center_y})")
                    return (center_x, center_y, {
                        "text": match.group(1),
                        "bounds": [x1, y1, x2, y2],
                        "center": [center_x, center_y]
                    })
            
            # METHOD 1B: Find clickable nodes and check their CHILD nodes for text
            # Pattern: <node clickable="true" bounds="...">...<node text="Log in"/>...</node>
            print("[Accessibility] Checking child nodes for text...")
            
            # More flexible pattern - clickable and bounds can be in any order
            pattern_clickable_with_bounds = r'<node[^>]*(?:clickable="true")[^>]*(?:bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\'])[^>]*>'
            matches = re.finditer(pattern_clickable_with_bounds, tree, re.IGNORECASE)
            
            for match in matches:
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                
                # Find the end of this node's opening tag
                start_pos = match.end()
                
                # Search within the next section for child nodes with text
                # Look ahead up to 3000 chars (should cover nested structure)
                search_end = min(start_pos + 3000, len(tree))
                node_section = tree[start_pos:search_end]
                
                # Look for text="..." in child nodes within this section
                # Pattern: <node[^>]*text=["']([^"']*)["']
                child_text_pattern = r'<node[^>]*text=["\']([^"\']*)["\']'
                child_matches = re.finditer(child_text_pattern, node_section, re.IGNORECASE)
                
                for child_match in child_matches:
                    child_text = child_match.group(1).lower().strip()
                    if child_text and any(keyword in child_text for keyword in keywords_lower):
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        print(f"[Accessibility] ✓ Found button by child node text: '{child_match.group(1)}' at ({center_x}, {center_y})")
                        return (center_x, center_y, {
                            "text": child_match.group(1),
                            "bounds": [x1, y1, x2, y2],
                            "center": [center_x, center_y]
                        })
            
            # METHOD 1C: Alternative approach - find text nodes first, then find parent clickable node
            print("[Accessibility] Finding text nodes first, then parent clickable nodes...")
            
            # Find all nodes with text matching keywords (case-insensitive)
            text_pattern = r'<node[^>]*text=["\']([^"\']*)["\']'
            text_matches = list(re.finditer(text_pattern, tree, re.IGNORECASE))
            
            for text_match in text_matches:
                text = text_match.group(1).lower().strip()
                original_text = text_match.group(1).strip()
                
                if text and any(keyword in text for keyword in keywords_lower):
                    text_pos = text_match.start()
                    
                    # Look backwards to find the nearest clickable parent node
                    # Search backwards up to 8000 chars to find parent (larger range for nested structures)
                    search_start = max(0, text_pos - 8000)
                    section_before = tree[search_start:text_pos]
                    
                    # Find the last clickable node before this text node (more flexible pattern)
                    clickable_pattern = r'<node[^>]*(?:clickable="true")[^>]*(?:bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\'])[^>]*>'
                    clickable_matches = list(re.finditer(clickable_pattern, section_before, re.IGNORECASE))
                    
                    if clickable_matches:
                        # Use the last (closest) clickable node
                        parent_match = clickable_matches[-1]
                        x1 = int(parent_match.group(1))
                        y1 = int(parent_match.group(2))
                        x2 = int(parent_match.group(3))
                        y2 = int(parent_match.group(4))
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        print(f"[Accessibility] ✓ Found button by parent search: '{original_text}' at ({center_x}, {center_y})")
                        return (center_x, center_y, {
                            "text": original_text,
                            "bounds": [x1, y1, x2, y2],
                            "center": [center_x, center_y]
                        })
            
            # METHOD 2: Find clickable nodes with content-desc attribute (for icon buttons)
            pattern_with_desc = r'<node[^>]*clickable="true"[^>]*content-desc=["\']([^"\']*)["\'][^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\']'
            
            matches = re.finditer(pattern_with_desc, tree, re.IGNORECASE)
            
            for match in matches:
                content_desc = match.group(1).lower()
                x1 = int(match.group(2))
                y1 = int(match.group(3))
                x2 = int(match.group(4))
                y2 = int(match.group(5))
                
                # Check if content-desc matches any keyword
                if any(keyword in content_desc for keyword in keywords_lower):
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    print(f"[Accessibility] Found button by content-desc: '{match.group(1)}' at ({center_x}, {center_y})")
                    return (center_x, center_y, {
                        "text": match.group(1),
                        "bounds": [x1, y1, x2, y2],
                        "center": [center_x, center_y]
                    })
            
            # METHOD 3: Find all clickable nodes and check their class/resource-id
            # Look for Button, ImageButton classes or resource-id containing keywords
            pattern_clickable = r'<node[^>]*clickable="true"[^>]*bounds=["\']\[(\d+),(\d+)\]\[(\d+),(\d+)\]["\'][^>]*>'
            
            matches = re.finditer(pattern_clickable, tree, re.IGNORECASE)
            
            for match in matches:
                node_text = match.group(0)
                x1 = int(match.group(1))
                y1 = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                
                # Extract class, resource-id, and text
                class_match = re.search(r'class=["\']([^"\']*)["\']', node_text)
                resource_id_match = re.search(r'resource-id=["\']([^"\']*)["\']', node_text)
                text_match = re.search(r'text=["\']([^"\']*)["\']', node_text)
                desc_match = re.search(r'content-desc=["\']([^"\']*)["\']', node_text)
                
                element_class = class_match.group(1).lower() if class_match else ""
                resource_id = resource_id_match.group(1).lower() if resource_id_match else ""
                text = text_match.group(1).lower() if text_match else ""
                content_desc = desc_match.group(1).lower() if desc_match else ""
                
                # Check if it's a button class
                is_button = "button" in element_class or "imagebutton" in element_class
                
                # Check if any attribute matches keywords
                all_text = f"{text} {content_desc} {resource_id}".lower()
                
                if is_button and any(keyword in all_text for keyword in keywords_lower):
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    found_text = text_match.group(1) if text_match else (desc_match.group(1) if desc_match else resource_id_match.group(1) if resource_id_match else "button")
                    print(f"[Accessibility] Found button by class/resource-id: '{found_text}' (class: {element_class}) at ({center_x}, {center_y})")
                    return (center_x, center_y, {
                        "text": found_text,
                        "bounds": [x1, y1, x2, y2],
                        "center": [center_x, center_y],
                        "class": element_class
                    })
            
            # METHOD 4: Find clickable elements without text but with button-like characteristics
            # (e.g., large clickable areas that might be login buttons)
            print("[Accessibility] No button found with text/content-desc, trying to find by position...")
            
            # Get all clickable elements and look for ones that might be login buttons
            # Login buttons are usually:
            # - In the middle-bottom area of screen
            # - Reasonable size (not too small, not too large)
            # - Have button-like classes
            
            all_clickable = self.find_clickable_elements()
            screen_height = 2400  # Pixel 7A height
            screen_width = 1080   # Pixel 7A width
            
            for elem in all_clickable:
                x = elem.get("x", 0)
                y = elem.get("y", 0)
                bounds = elem.get("bounds", [])
                elem_class = elem.get("class", "").lower()
                text = elem.get("text", "").lower()
                
                if len(bounds) >= 4:
                    width = bounds[2] - bounds[0]
                    height = bounds[3] - bounds[1]
                    
                    # Check if it's button-like (has button in class or reasonable size)
                    is_button_like = "button" in elem_class or (width > 200 and height > 50 and width < 1000 and height < 200)
                    
                    # Check if it's in a reasonable position (not at top corners, usually middle-bottom)
                    is_reasonable_position = y > 300 and y < screen_height - 200 and x > 100 and x < screen_width - 100
                    
                    # If we have keywords and this looks like a button, check if any keyword appears in text/class
                    if is_button_like and is_reasonable_position:
                        all_attrs = f"{text} {elem_class}".lower()
                        if any(keyword in all_attrs for keyword in keywords_lower):
                            print(f"[Accessibility] Found potential login button by position/class: '{text or elem_class}' at ({x}, {y})")
                            return (x, y, {
                                "text": text or "button",
                                "bounds": bounds,
                                "center": [x, y],
                                "class": elem_class
                            })
            
            print("[Accessibility] No login button found with any method")
            # Debug: Print first few clickable elements for debugging
            if all_clickable:
                print(f"[Accessibility] Debug: Found {len(all_clickable)} clickable elements. First 10:")
                for i, elem in enumerate(all_clickable[:10]):
                    text = elem.get('text', '') or elem.get('content_desc', '')
                    print(f"  {i+1}. Text: '{text}', Class: '{elem.get('class', '')}', Pos: ({elem.get('x', 0)}, {elem.get('y', 0)})")
        
        except Exception as e:
            print(f"[Accessibility] Error finding button by keywords: {e}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def debug_print_tree(self, max_elements: int = 20) -> None:
        """
        Debug method to print all clickable elements in the tree
        
        Args:
            max_elements: Maximum number of elements to print
        """
        elements = self.find_clickable_elements()
        print(f"[Accessibility Debug] Found {len(elements)} clickable elements:")
        
        for i, elem in enumerate(elements[:max_elements]):
            text = elem.get("text", "") or elem.get("content_desc", "")
            elem_class = elem.get("class", "")
            x = elem.get("x", 0)
            y = elem.get("y", 0)
            
            display_text = text or elem_class or "no text"
            print(f"  {i+1}. '{display_text}' | Class: {elem_class} | Pos: ({x}, {y})")
        
        if len(elements) > max_elements:
            print(f"  ... and {len(elements) - max_elements} more elements")
