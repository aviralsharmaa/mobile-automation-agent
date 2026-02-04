"""
Quick test script to verify ElevenLabs TTS is working
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.voice.tts import TextToSpeech

def test_elevenlabs():
    """Test ElevenLabs TTS"""
    print("Testing ElevenLabs TTS configuration...")
    
    # Check if API key is set
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or api_key == "your-elevenlabs-key-here":
        print("[ERROR] ELEVENLABS_API_KEY not set in .env file")
        return False
    
    print(f"[OK] API Key found: {api_key[:10]}...")
    
    # Initialize TTS with ElevenLabs
    try:
        tts = TextToSpeech(use_elevenlabs=True)
        
        if not tts.use_elevenlabs:
            print("[ERROR] ElevenLabs not enabled. Check your API key.")
            return False
        
        print("[OK] ElevenLabs TTS initialized successfully")
        print("\nSpeaking test message...")
        tts.speak("Hello! This is a test of the ElevenLabs text to speech system.")
        print("\n[OK] Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_elevenlabs()
    sys.exit(0 if success else 1)
