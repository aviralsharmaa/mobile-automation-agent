"""
Speech-to-Text Module - Voice input processing
"""
import os
import re
import speech_recognition as sr
from typing import Optional, Callable


class SpeechToText:
    """Speech-to-text conversion using multiple backends"""
    
    # Common misrecognitions correction dictionary
    # Expanded with many variations for better accuracy
    CORRECTIONS = {
        # ChatGPT variations (most common misrecognitions)
        "openchargegpt": "open chatgpt",  # When "open chatgpt" is heard as one word
        "openchargept": "open chatgpt",
        "openchargebt": "open chatgpt",
        "openchatgpt": "open chatgpt",
        "openjgpt": "open chatgpt",
        "chargegpt": "chatgpt",
        "charge pt": "chatgpt",
        "charge p t": "chatgpt",
        "charge gpt": "chatgpt",
        "charge bt": "chatgpt",  # Common Whisper misrecognition
        "charger bt": "chatgpt",
        "charger pt": "chatgpt",
        "charger p t": "chatgpt",
        "charger gpt": "chatgpt",
        "charge b t": "chatgpt",
        "charge b": "chatgpt",
        "charger": "chatgpt",
        "chart gpt": "chatgpt",
        "chat gpt": "chatgpt",
        "chat g p t": "chatgpt",
        "chat g p": "chatgpt",
        "chat g": "chatgpt",
        "charge p": "chatgpt",
        "jgpt": "chatgpt",
        "j gpt": "chatgpt",
        "j g p t": "chatgpt",
        "j g p": "chatgpt",
        "chad gpt": "chatgpt",
        "chad pt": "chatgpt",
        "chad p t": "chatgpt",
        "chatter gpt": "chatgpt",
        "chatter pt": "chatgpt",
        # Gmail variations
        "g mail": "gmail",
        "g male": "gmail",
        "gmit": "gmail",
        "g mit": "gmail",
        "g male": "gmail",
        "g mayle": "gmail",
        "g meil": "gmail",
        "g meel": "gmail",
        "gmail": "gmail",
        # WhatsApp variations
        "whats app": "whatsapp",
        "what's app": "whatsapp",
        "whatsapp": "whatsapp",
        "what app": "whatsapp",
        "whats up": "whatsapp",  # Only if context suggests app
        # YouTube variations
        "you tube": "youtube",
        "youtube": "youtube",
        "u tube": "youtube",
        "you toob": "youtube",
        # Play Store variations
        "playstore": "play store",
        "play store": "play store",
        "playstore": "play store",
        # Settings variations
        "settings": "settings",
        "setting": "settings",
        "setings": "settings",
        # Chrome variations
        "chrome": "chrome",
        "google chrome": "chrome",
        # Maps variations
        "maps": "maps",
        "google maps": "maps",
        "map": "maps",
    }
    
    def __init__(self, use_whisper: bool = False, wake_word: str = "hey assistant"):
        """
        Initialize STT
        
        Args:
            use_whisper: Use OpenAI Whisper API instead of Google SpeechRecognition
            wake_word: Wake word to listen for (default: "hey assistant")
        """
        self.recognizer = sr.Recognizer()
        self.use_whisper = use_whisper
        self.wake_word = wake_word.lower()
        self.microphone = None
        
        # Configure recognizer for better accuracy and complete sentence capture
        # Energy threshold - minimum audio energy to consider as speech
        self.recognizer.energy_threshold = 4000  # Default is 300, increase for better filtering
        # Pause threshold - seconds of non-speaking audio before phrase ends
        # Increased to 1.5 seconds to wait longer for complete sentences (like ChatGPT)
        self.recognizer.pause_threshold = 1.5  # Wait longer for pauses (default 0.8)
        # Dynamic energy threshold - adjust based on ambient noise
        self.recognizer.dynamic_energy_threshold = True
        # Non-speaking duration - seconds of non-speaking audio to keep on both sides
        self.recognizer.non_speaking_duration = 0.5
        
        # Adjust for ambient noise
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            print(f"Warning: Could not initialize microphone: {e}")
    
    def listen(self, timeout: float = 3.0, phrase_time_limit: float = 30.0) -> Optional[str]:
        """
        Listen for speech input - waits for complete sentence
        
        Args:
            timeout: Maximum seconds to wait for speech to start (reduced since we want to start quickly)
            phrase_time_limit: Maximum seconds for a phrase (increased for complete sentences)
            
        Returns:
            Recognized text or None if failed
        """
        if not self.microphone:
            print("Microphone not available")
            return None
        
        try:
            with self.microphone as source:
                # Quick ambient noise adjustment (don't wait too long)
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                
                print("Listening... (speak your complete command, pause when done)")
                
                # Listen with longer phrase_time_limit to capture complete sentences
                # pause_threshold (1.2s) means it waits 1.2 seconds of silence before ending
                # This allows for natural pauses in speech without cutting off
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                
                print("Processing speech...")
            
            return self._recognize(audio)
            
        except sr.WaitTimeoutError:
            print("No speech detected - try speaking again")
            return None
        except Exception as e:
            print(f"Error listening: {e}")
            return None
    
    def listen_from_file(self, file_path: str) -> Optional[str]:
        """
        Recognize speech from an audio file
        
        Args:
            file_path: Path to audio file (WAV, MP3, etc.)
            
        Returns:
            Recognized text or None if failed
        """
        try:
            # Load audio file
            audio_file = sr.AudioFile(file_path)
            with audio_file as source:
                audio = self.recognizer.record(source)
            
            print(f"Processing audio file: {file_path}")
            return self._recognize(audio)
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None
    
    def _correct_text(self, text: str) -> str:
        """
        Apply corrections to recognized text with fuzzy matching for app names
        
        Args:
            text: Recognized text
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        original_text = text
        text_lower = text.lower().strip()
        
        # Known app names for context-aware correction
        known_apps = ["chatgpt", "gmail", "whatsapp", "youtube", "settings", "chrome", "maps", "camera", "phone"]
        
        # Apply corrections (case-insensitive) - try longest matches first
        # Sort by length descending to match longer phrases first
        sorted_corrections = sorted(self.CORRECTIONS.items(), key=lambda x: len(x[0]), reverse=True)
        
        for wrong, correct in sorted_corrections:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(wrong) + r'\b'
            text_lower = re.sub(pattern, correct, text_lower, flags=re.IGNORECASE)
        
        # Fuzzy matching for common misrecognitions
        # Check if text contains "open" command and try to fix app names
        if "open" in text_lower:
            words = text_lower.split()
            open_idx = -1
            for i, word in enumerate(words):
                if word == "open":
                    open_idx = i
                    break
            
            if open_idx >= 0 and open_idx + 1 < len(words):
                # Get the word(s) after "open"
                potential_app = words[open_idx + 1]
                
                # Fuzzy match against known apps
                for app in known_apps:
                    # Check similarity (simple Levenshtein-like check)
                    similarity = self._word_similarity(potential_app, app)
                    if similarity > 0.6:  # 60% similarity threshold
                        words[open_idx + 1] = app
                        text_lower = ' '.join(words)
                        print(f"[Auto-corrected] '{potential_app}' -> '{app}' (similarity: {similarity:.2f})")
                        break
                
                # Specific fixes for common Whisper misrecognitions
                fixes = {
                    "chargerbt": "chatgpt",
                    "charger": "chatgpt",  # If followed by nothing or common words
                    "chargebt": "chatgpt",
                    "charge": "chatgpt",  # Only if context suggests it
                }
                
                if potential_app in fixes:
                    words[open_idx + 1] = fixes[potential_app]
                    text_lower = ' '.join(words)
                    print(f"[Auto-corrected] '{potential_app}' -> '{fixes[potential_app]}'")
        
        # Preserve original case structure for non-app-name parts
        if text_lower != original_text.lower():
            # Reconstruct with proper capitalization
            corrected_words = text_lower.split()
            words = original_text.split()
            
            # Preserve capitalization of first word
            if words and words[0][0].isupper() and corrected_words:
                corrected_words[0] = corrected_words[0].capitalize()
            
            return ' '.join(corrected_words)
        
        return text
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate simple similarity between two words
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not word1 or not word2:
            return 0.0
        
        # Remove spaces and convert to lowercase
        w1 = word1.replace(" ", "").lower()
        w2 = word2.replace(" ", "").lower()
        
        if w1 == w2:
            return 1.0
        
        # Check if one contains the other
        if w1 in w2 or w2 in w1:
            return 0.8
        
        # Simple character overlap
        set1 = set(w1)
        set2 = set(w2)
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _recognize(self, audio) -> Optional[str]:
        """
        Recognize speech from audio
        
        Args:
            audio: AudioData from recognizer
            
        Returns:
            Recognized text or None
        """
        try:
            if self.use_whisper:
                # Use OpenAI Whisper API (more accurate)
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("[WARNING] OPENAI_API_KEY not set, falling back to Google SpeechRecognition")
                    text = self._recognize_google(audio)
                else:
                    # Use Whisper API
                    client = openai.OpenAI(api_key=api_key)
                    
                    # Save audio to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio.get_wav_data())
                        tmp_path = tmp_file.name
                    
                    try:
                        with open(tmp_path, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file,
                                language="en"  # Specify English for better accuracy
                            )
                        text = transcript.text
                    finally:
                        os.unlink(tmp_path)
            else:
                # Use Google SpeechRecognition (free, no API key needed)
                text = self._recognize_google(audio)
            
            # Apply corrections to fix common misrecognitions
            if text:
                corrected = self._correct_text(text)
                if corrected != text:
                    print(f"[Corrected]: '{text}' -> '{corrected}'")
                return corrected
            
            return text
                
        except Exception as e:
            print(f"Error recognizing speech: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _recognize_google(self, audio) -> Optional[str]:
        """Recognize using Google Speech Recognition"""
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None
    
    def listen_for_wake_word(self, callback: Optional[Callable] = None) -> Optional[str]:
        """
        Listen continuously for wake word, then listen for command
        
        Args:
            callback: Optional callback function when wake word detected
            
        Returns:
            Command text after wake word or None
        """
        print(f"Listening for wake word: '{self.wake_word}'...")
        
        while True:
            # Listen for wake word with shorter timeout
            text = self.listen(timeout=1.0, phrase_time_limit=5.0)
            
            if text:
                text_lower = text.lower()
                if self.wake_word in text_lower:
                    if callback:
                        callback()
                    
                    # Extract command after wake word
                    command = text_lower.split(self.wake_word, 1)
                    if len(command) > 1 and command[1].strip():
                        return command[1].strip()
                    
                    # If wake word detected but no command, listen for next phrase with longer timeout
                    print("Wake word detected. Listening for command...")
                    return self.listen(timeout=3.0, phrase_time_limit=20.0)
    
    def continuous_listen(self, callback: Callable[[str], None]):
        """
        Continuously listen and call callback with recognized text
        
        Args:
            callback: Function to call with recognized text
        """
        print("Starting continuous listening...")
        while True:
            text = self.listen()
            if text:
                callback(text)
