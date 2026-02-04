"""
Mobile Automation Agent - Production Entry Point
Main entry point for running the production-grade mobile automation agent
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
    """Main entry point for production agent"""
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("=" * 60)
        print("ERROR: OPENAI_API_KEY not set in environment")
        print("=" * 60)
        print("Please set it in .env file or environment variables")
        print("Example: OPENAI_API_KEY=your-key-here")
        return 1
    
    print("=" * 60)
    print("Mobile Automation Agent - Production Mode")
    print("=" * 60)
    print("\n[Initializing] Setting up agent...")
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(config)
        
        # Check microphone availability
        if not orchestrator.stt.microphone:
            print("\n" + "=" * 60)
            print("WARNING: Microphone not available!")
            print("=" * 60)
            print("The agent requires microphone input for voice commands.")
            print("\nPlease check:")
            print("  1. Microphone is connected")
            print("  2. Microphone permissions are granted")
            print("  3. PyAudio is installed: pip install pyaudio")
            print("\nContinuing anyway (you can use text input if available)...")
            print("=" * 60)
        
        # Show configuration
        print("\n[Configuration]")
        print(f"  STT Backend: {'Whisper' if orchestrator.stt.use_whisper else 'Google Speech Recognition'}")
        print(f"  TTS Backend: pyttsx3")
        print(f"  Wake Word: {config.get('agent.wake_word', 'hey assistant')}")
        print(f"  Device: {orchestrator.adb_client.get_device().serial if orchestrator.adb_client.get_device() else 'Not connected'}")
        
        print("\n" + "=" * 60)
        print("[Ready] Agent is ready to receive commands")
        print("=" * 60)
        print("\nYou can:")
        print("  - Chat: 'Hello', 'What can you do?', 'Help'")
        print("  - Commands: 'Open Gmail', 'Open Settings', 'Open ChatGPT'")
        print("  - Say 'exit' or 'quit' to stop")
        print("-" * 60)
        
        # Get STT instance for direct listening
        stt = orchestrator.stt
        
        # Welcome message
        orchestrator.tts.speak("Voice command mode activated. Say your command.")
        
        # Main loop - listen and process commands directly (no wake word required)
        while True:
            try:
                # Listen for voice command
                print("\n[Listening...] (speak your complete command)")
                print("(Pause 1-2 seconds after finishing your sentence)")
                command = stt.listen(timeout=2.0, phrase_time_limit=25.0)
                
                if not command:
                    print("[No command detected. Listening again...]")
                    continue
                
                command = command.strip()
                command_lower = command.lower()
                
                print(f"[Heard]: {command}")
                
                # Check for exit commands - require explicit exit phrases to avoid false positives
                # Only accept clear, unambiguous exit commands
                import re
                command_clean = command_lower.strip()
                
                # Explicit exit phrases (must match exactly or be the main intent)
                explicit_exit_phrases = [
                    r'^\s*(exit|quit|stop|shutdown|shut down)\s*$',  # Standalone exit words
                    r'goodbye',  # "goodbye" anywhere
                    r'close app',
                    r'end session',
                    r'see you',
                    r'farewell'
                ]
                
                # Check if command matches explicit exit phrases
                is_exit = False
                for pattern in explicit_exit_phrases:
                    if re.search(pattern, command_clean):
                        is_exit = True
                        break
                
                # Only accept "bye" if it's part of "goodbye" or "bye-bye" (with context)
                # Reject standalone "bye" or "bye bye" as they're too easily misrecognized
                if not is_exit and 'bye' in command_clean:
                    # Only match if it's clearly "goodbye" or "bye-bye" (with hyphen)
                    if re.search(r'\b(goodbye|bye-bye)\b', command_clean):
                        is_exit = True
                    # Reject standalone "bye" or "bye bye" - too ambiguous
                    elif re.search(r'^\s*bye\s*$', command_clean) or re.search(r'^\s*bye\s+bye\s*$', command_clean):
                        print("[Warning] Detected standalone 'bye' - ignoring to prevent false exit")
                        print("[Info] Say 'exit', 'quit', or 'goodbye' to stop the agent")
                        continue  # Don't exit, wait for next command
                
                if is_exit:
                    orchestrator.tts.speak("Goodbye! Have a great day!")
                    print("\n[Stopped]")
                    break
                
                if command:
                    # Check if we're in login flow and awaiting email/password
                    # This is a simple check - in production you'd track this in state
                    command_lower = command.lower()
                    
                    # Check for email pattern
                    import re
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    has_email = bool(re.search(email_pattern, command))
                    has_password_keyword = "password" in command_lower
                    
                    # Process the command (handles both conversational and actual commands)
                    # The orchestrator will handle issue context and login flow internally
                    result = orchestrator.process_command(command)
                    
                    # Check if it was login flow
                    if result.get("login_flow"):
                        if result.get("awaiting_password"):
                            print("[Login Flow] Email entered. Waiting for password...")
                            print("-" * 60)
                        elif result.get("login_complete"):
                            print("[Login Flow] Login completed successfully!")
                            print("-" * 60)
                    # Check if it was conversational
                    elif result.get("conversational"):
                        # Already handled conversationally, just continue
                        print("-" * 60)
                    else:
                        # It was a command
                        print(f"[Processing]: {command}")
                        print(f"[Done]")
                        print("-" * 60)
            
            except KeyboardInterrupt:
                print("\n\n[Shutdown] Agent stopped by user")
                orchestrator.tts.speak("Stopped")
                break
            except Exception as e:
                print(f"[Error]: {e}")
                logger.error(f"Error: {e}")
                orchestrator.tts.speak("I encountered an error. Please try again.")
        
    except KeyboardInterrupt:
        print("\n\n[Shutdown] Agent stopped by user")
        logger.info("Agent shutdown by user")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to start agent: {e}")
        logger.error(f"Failed to start agent: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
