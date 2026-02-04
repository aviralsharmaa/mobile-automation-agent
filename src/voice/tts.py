"""
Text-to-Speech Module - Voice output using pyttsx3
"""
import pyttsx3
from typing import Optional


class TextToSpeech:
    """Text-to-speech conversion using pyttsx3 (simple and accurate)"""
    
    def __init__(self, speech_rate: int = 150, volume: float = 1.0):
        """
        Initialize TTS with pyttsx3
        
        Args:
            speech_rate: Words per minute (default: 150)
            volume: Volume level 0.0 to 1.0 (default: 1.0)
        """
        self.engine = None
        
        try:
            print("[TTS] Initializing pyttsx3 engine...")
            self.engine = pyttsx3.init()
            print(f"[TTS] pyttsx3 engine initialized: {self.engine}")
            self._configure_engine(speech_rate, volume)
            print("[TTS] pyttsx3 engine configured successfully")
        except Exception as e:
            print(f"[TTS ERROR] Could not initialize pyttsx3: {e}")
            import traceback
            traceback.print_exc()
            self.engine = None
    
    def _configure_engine(self, speech_rate: int = 150, volume: float = 1.0):
        """Configure pyttsx3 engine settings"""
        if not self.engine:
            return
        
        try:
            # Set speech rate (words per minute)
            self.engine.setProperty('rate', speech_rate)
            
            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
            # Try to set a more natural voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available (often sounds more natural)
                voice_set = False
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        voice_set = True
                        break
                if not voice_set and voices:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
        except Exception as e:
            print(f"[TTS WARNING] Error configuring engine: {e}")
    
    def speak(self, text: str, interrupt: bool = True):
        """
        Speak text using pyttsx3
        
        Args:
            text: Text to speak
            interrupt: Whether to interrupt current speech
        """
        if not text:
            return
        
        if not self.engine:
            print("[TTS ERROR] TTS engine not available - audio will not play")
            print(f"AI: {text}")  # At least print the text
            return
        
        print(f"AI: {text}")
        
        try:
            if interrupt:
                self.engine.stop()
            
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] Error speaking: {e}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        """Stop current speech"""
        if self.engine:
            self.engine.stop()
    
    def set_rate(self, rate: int):
        """
        Set speech rate
        
        Args:
            rate: Words per minute (default: 150)
        """
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume: float):
        """
        Set volume
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.engine:
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
