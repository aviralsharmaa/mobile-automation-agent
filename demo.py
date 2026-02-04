"""
Demo Script - End-to-end demonstration
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.orchestrator import AgentOrchestrator
from src.utils.config import Config
from src.utils.logging import logger


def demo_basic_app_launch():
    """Demo: Basic app launch"""
    print("\n=== Demo: Basic App Launch ===")
    print("Command: 'Open Settings'")
    
    config = Config()
    orchestrator = AgentOrchestrator(config)
    
    result = orchestrator.process_command("open settings")
    print(f"Result: {result}")
    return result


def demo_with_search():
    """Demo: App launch with search"""
    print("\n=== Demo: App Launch with Search ===")
    print("Command: 'Open ChatGPT and ask what's the capital of France'")
    
    config = Config()
    orchestrator = AgentOrchestrator(config)
    
    result = orchestrator.process_command("open chatgpt and ask what's the capital of France")
    print(f"Result: {result}")
    return result


def demo_authentication():
    """Demo: Authentication flow"""
    print("\n=== Demo: Authentication Flow ===")
    print("Command: 'Open Gmail'")
    print("Note: This will prompt for credentials if login screen detected")
    
    config = Config()
    orchestrator = AgentOrchestrator(config)
    
    result = orchestrator.process_command("open gmail")
    print(f"Result: {result}")
    return result


def interactive_demo():
    """Interactive demo mode with voice input"""
    print("\n=== Interactive Demo Mode (Voice) ===")
    print("Speak your commands or say 'exit' to quit")
    print("Example: 'open settings' or 'open gmail'")
    print("\nListening for voice commands...")
    
    config = Config()
    orchestrator = AgentOrchestrator(config)
    
    # Get STT instance from orchestrator
    stt = orchestrator.stt
    
    # Check if microphone is available
    if not stt.microphone:
        print("\n[ERROR] Microphone not available. Please check:")
        print("1. Microphone is connected")
        print("2. Microphone permissions are granted")
        print("3. PyAudio is installed correctly")
        print("\nFalling back to text input mode...")
        return interactive_demo_text()
    
    print("\n[Ready] Say your command...")
    
    while True:
        try:
            # Listen for voice command
            print("\n[Listening...]")
            command = stt.listen(timeout=10.0, phrase_time_limit=10.0)
            
            if not command:
                print("[No command detected. Try again.]")
                continue
            
            command = command.strip().lower()
            print(f"\n[Heard]: {command}")
            
            if command in ['exit', 'quit', 'stop', 'goodbye']:
                orchestrator.tts.speak("Goodbye!")
                break
            
            if command:
                print(f"[Processing]: {command}")
                result = orchestrator.process_command(command)
                print(f"[Result]: Task completed")
        
        except KeyboardInterrupt:
            print("\n[Stopped]")
            orchestrator.tts.speak("Stopped by user")
            break
        except Exception as e:
            print(f"[Error]: {e}")
            logger.error(f"Error in interactive demo: {e}")


def interactive_demo_text():
    """Fallback: Interactive demo mode with text input"""
    print("\n=== Interactive Demo Mode (Text) ===")
    print("Type commands or 'exit' to quit")
    print("Example: 'open settings'")
    
    config = Config()
    orchestrator = AgentOrchestrator(config)
    
    while True:
        try:
            command = input("\nCommand: ").strip()
            if command.lower() in ['exit', 'quit', 'q']:
                break
            
            if command:
                result = orchestrator.process_command(command)
                print(f"Result: {result}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main demo function"""
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set in environment")
        print("Please set it in .env file or environment variables")
        return
    
    print("Mobile Automation Agent - Demo")
    print("=" * 50)
    
    # Run demos
    try:
        # Demo 1: Basic app launch
        demo_basic_app_launch()
        
        # Wait for user input
        input("\nPress Enter to continue to next demo...")
        
        # Demo 2: With search (if ChatGPT is installed)
        # demo_with_search()
        
        # Wait for user input
        input("\nPress Enter to continue to interactive voice mode...")
        
        # Interactive mode with voice input
        interactive_demo()
    
    except Exception as e:
        logger.error(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
