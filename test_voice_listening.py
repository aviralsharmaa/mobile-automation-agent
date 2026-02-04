"""
Test voice listening with improved settings
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.voice.stt import SpeechToText

def test_listening():
    """Test improved voice listening"""
    load_dotenv()
    
    print("=" * 60)
    print("Voice Listening Test")
    print("=" * 60)
    print("\nThis will test if the voice recognition captures complete sentences.")
    print("Try saying: 'Open Gmail' or 'Open ChatGPT'")
    print("\nThe system will wait for 1.2 seconds of silence before processing.")
    print("-" * 60)
    
    # Prefer Whisper if OpenAI key is available
    use_whisper = os.getenv("OPENAI_API_KEY") is not None
    if use_whisper:
        print("[Using Whisper API for better accuracy]")
    else:
        print("[Using Google Speech Recognition]")
    
    stt = SpeechToText(use_whisper=use_whisper)
    
    if not stt.microphone:
        print("\n[ERROR] Microphone not available!")
        return
    
    print(f"\n[Settings]")
    print(f"  Pause threshold: {stt.recognizer.pause_threshold}s")
    print(f"  Energy threshold: {stt.recognizer.energy_threshold}")
    print(f"  Dynamic energy: {stt.recognizer.dynamic_energy_threshold}")
    print(f"  STT Backend: {'Whisper' if use_whisper else 'Google'}")
    print("\n[Ready] Starting test...")
    
    for i in range(3):
        print(f"\n--- Test {i+1}/3 ---")
        print("Speak your command now (pause 1-2 seconds when done)...")
        
        command = stt.listen(timeout=3.0, phrase_time_limit=25.0)
        
        if command:
            print(f"\n[SUCCESS] Heard: '{command}'")
            print(f"Length: {len(command)} characters")
        else:
            print("\n[FAILED] No command detected")
        
        if i < 2:
            input("\nPress Enter for next test...")

if __name__ == "__main__":
    test_listening()
