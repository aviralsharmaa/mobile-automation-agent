"""
Simple TTS test to verify audio output works
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from src.voice.tts import TextToSpeech

def test_pyttsx3():
    """Test pyttsx3 directly"""
    print("=" * 60)
    print("Testing pyttsx3 TTS")
    print("=" * 60)
    
    tts = TextToSpeech(use_elevenlabs=False)
    
    if not tts.engine:
        print("[ERROR] pyttsx3 engine not initialized!")
        return False
    
    print(f"[INFO] Engine: {tts.engine}")
    print(f"[INFO] Rate: {tts.engine.getProperty('rate')}")
    print(f"[INFO] Volume: {tts.engine.getProperty('volume')}")
    
    try:
        voices = tts.engine.getProperty('voices')
        if voices:
            print(f"[INFO] Current voice: {tts.engine.getProperty('voice')}")
            print(f"[INFO] Available voices: {len(voices)}")
            for i, voice in enumerate(voices[:3]):  # Show first 3
                print(f"  {i+1}. {voice.name}")
    except Exception as e:
        print(f"[WARNING] Could not get voices: {e}")
    
    print("\n[TEST] Speaking: 'Hello, this is a test'")
    print("[TEST] You should hear audio now...")
    
    try:
        tts.speak("Hello, this is a test. Can you hear me?")
        print("[SUCCESS] TTS completed without errors")
        return True
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_elevenlabs():
    """Test ElevenLabs"""
    print("\n" + "=" * 60)
    print("Testing ElevenLabs TTS")
    print("=" * 60)
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("[SKIP] ELEVENLABS_API_KEY not set, skipping ElevenLabs test")
        return False
    
    tts = TextToSpeech(use_elevenlabs=True)
    
    if not tts.use_elevenlabs:
        print("[SKIP] ElevenLabs not enabled")
        return False
    
    print("\n[TEST] Speaking: 'Hello, this is a test'")
    print("[TEST] You should hear audio now...")
    
    try:
        tts.speak("Hello, this is a test. Can you hear me?")
        print("[SUCCESS] TTS completed without errors")
        return True
    except Exception as e:
        print(f"[ERROR] TTS failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TTS Audio Test")
    print("Make sure your speakers/headphones are on and volume is up!")
    print("-" * 60)
    
    # Test pyttsx3 first (should always work)
    pyttsx3_ok = test_pyttsx3()
    
    # Test ElevenLabs if available
    elevenlabs_ok = test_elevenlabs()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  pyttsx3: {'OK' if pyttsx3_ok else 'FAILED'}")
    print(f"  ElevenLabs: {'OK' if elevenlabs_ok else 'SKIPPED/FAILED'}")
    print("=" * 60)
    
    if not pyttsx3_ok:
        print("\n[WARNING] pyttsx3 is not working. Check:")
        print("  1. Audio drivers are installed")
        print("  2. Speakers/headphones are connected")
        print("  3. Volume is not muted")
        print("  4. Windows audio service is running")
