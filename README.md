# Mobile Automation Agent for Blind Users

An AI-powered mobile automation agent that assists blind users in operating Android devices through natural language voice commands.

## Features

- **Voice Interface**: Speech-to-text input and text-to-speech output
- **Vision Understanding**: GPT-4o vision for screen analysis and element detection
- **Device Control**: ADB integration for Android device interaction
- **Authentication Handling**: Secure credential handling for login flows
- **Error Recovery**: Robust handling of popups and UI changes

## Prerequisites

- Python 3.11+
- Conda (recommended) or pip
- Android Studio with emulator running
- ADB accessible (usually comes with Android Studio)
- OpenAI API key

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate mobile-automation-agent
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your `OPENAI_API_KEY`.

### 3. Verify ADB Connection

Ensure your Android emulator is running and connected:

```bash
adb devices
```

You should see your device listed (e.g., `emulator-5554`).

### 4. Configure App Mappings

Edit `config/config.yaml` to add app package names for apps you want to control.

## Usage

### Production Mode (Recommended)

Run the production-grade agent:

```bash
python main.py
```

The agent will:
1. Initialize all components
2. Listen for voice commands (wake word: "Hey Assistant" by default)
3. Process commands and execute actions on Android device
4. Provide audio feedback via TTS

### Demo Mode

For testing and development:

```bash
# Voice-only interactive demo
python demo_voice.py

# Full demo with examples
python demo.py
```

### Example Commands

**Conversational:**
- "Hello" - Greet the agent
- "What can you do?" - Ask about capabilities
- "Help" - Get help

**Device Control:**
- "Open Settings"
- "Open Gmail"
- "Open ChatGPT and ask what's the capital of France"
- "Open Gmail and check for new emails"

## Workflow Documentation

For detailed workflow documentation, see [WORKFLOW.md](WORKFLOW.md)

## Project Structure

```
mobile-automation-agent/
├── src/
│   ├── voice/          # STT and TTS modules
│   ├── vision/         # Screen analysis and element detection
│   ├── device/         # ADB device control
│   ├── agent/          # Agent orchestration
│   └── utils/          # Utilities and configuration
├── config/             # Configuration files
├── requirements.txt    # Python dependencies
├── environment.yml     # Conda environment
└── README.md          # This file
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines.

## Security

- Credentials are never logged or persisted
- All sensitive data is handled in memory only
- Environment variables are used for API keys

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
