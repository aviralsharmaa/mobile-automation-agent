"""
Voice-Only Demo - Direct voice command interface
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.orchestrator import AgentOrchestrator
from src.utils.config import Config
from src.utils.logging import logger


def main():
    """Voice-only demo"""
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please set it in .env file or environment variables")
        return
    
    print("=" * 60)
    print("Mobile Automation Agent - Voice Command Mode")
    print("=" * 60)
    print("\n[Setup] Initializing agent...")
    
    try:
        config = Config()
        orchestrator = AgentOrchestrator(config)
        
        # Get STT instance and show which backend is being used
        stt = orchestrator.stt
        backend_name = "Whisper" if stt.use_whisper else "Google Speech Recognition"
        print(f"\n[STT Backend]: {backend_name}")
        if stt.use_whisper:
            print("[Note] Using Whisper API for better accuracy")
        else:
            print("[Note] Using Google Speech Recognition (free but less accurate)")
            print("[Tip] Set OPENAI_API_KEY in .env to use Whisper")
        
        # Check if microphone is available
        if not stt.microphone:
            print("\n[ERROR] Microphone not available!")
            print("Please check:")
            print("  1. Microphone is connected")
            print("  2. Microphone permissions are granted")
            print("  3. PyAudio is installed: pip install pyaudio")
            return
        
        # Welcome message
        orchestrator.tts.speak("Hello! I'm your mobile automation assistant. I can help you control your Android device. You can chat with me or give me commands like 'Open Gmail'.")
        print("\n[Ready] Listening for voice commands...")
        print("You can:")
        print("  - Chat with me: 'Hello', 'What can you do?', 'Help'")
        print("  - Give commands: 'Open Settings', 'Open Gmail', 'Open ChatGPT'")
        print("\nSay 'exit' or 'quit' to stop")
        print("-" * 60)
        
        while True:
            try:
                # Listen for voice command with longer timeout for complete sentences
                print("\n[Listening...] (speak your complete command)")
                print("(Pause 1-2 seconds after finishing your sentence)")
                # Use longer phrase_time_limit to capture full sentences
                # pause_threshold (1.2s) will wait for silence before processing
                command = stt.listen(timeout=2.0, phrase_time_limit=25.0)
                
                if not command:
                    print("[No command detected. Listening again...]")
                    continue
                
                command = command.strip()
                original_command = command  # Keep original for display
                command_lower = command.lower()
                
                print(f"[Heard]: {command}")
                
                # Show correction if applied (will be shown by STT module)
                
                # Check for exit commands
                if any(word in command_lower for word in ['exit', 'quit', 'stop', 'goodbye', 'shut down', 'bye']):
                    orchestrator.tts.speak("Goodbye! Have a great day!")
                    print("\n[Stopped]")
                    break
                
                if command:
                    # Process the command (handles both conversational and actual commands)
                    result = orchestrator.process_command(command)
                    
                    # Check if it was conversational
                    if result.get("conversational"):
                        # Already handled conversationally, just continue
                        print("-" * 60)
                    else:
                        # It was a command
                        print(f"[Processing]: {command}")
                        print(f"[Done]")
                        print("-" * 60)
            
            except KeyboardInterrupt:
                print("\n[Stopped by user]")
                orchestrator.tts.speak("Stopped")
                break
            except Exception as e:
                print(f"[Error]: {e}")
                logger.error(f"Error: {e}")
                orchestrator.tts.speak("I encountered an error. Please try again.")
    
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
