# Quick Start Guide

## Prerequisites

1. **Conda Environment** (already created)
   ```bash
   conda activate mobile-automation-agent
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Android Emulator Running**
   - Ensure your Pixel 7A emulator is running
   - Verify connection: `adb devices`
   - Should show: `emulator-5554   device`

4. **OpenAI API Key**
   - Get your API key from https://platform.openai.com/api-keys
   - Create `.env` file from `.env.example`
   - Add your `OPENAI_API_KEY`

## Setup Steps

### 1. Activate Environment
```bash
conda activate mobile-automation-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 4. Verify ADB Connection
```bash
adb devices
# Should show: emulator-5554   device
```

## Running the Agent

### Option 1: Interactive Demo
```bash
python demo.py
```

### Option 2: Full Agent (Voice Mode)
```bash
python -m src.agent.orchestrator
```

The agent will:
1. Say "System active. What can I help you with?"
2. Listen for wake word: "Hey Assistant"
3. Process your command
4. Execute actions on the device
5. Provide audio feedback

## Example Commands

- "Hey Assistant, open Settings"
- "Hey Assistant, open ChatGPT and ask what's the capital of France"
- "Hey Assistant, open Gmail"

## Troubleshooting

### ADB Not Found
- Ensure Android Studio is installed
- Add Android SDK platform-tools to PATH, or
- Set `ADB_PATH` in `.env` file

### No Device Found
- Ensure emulator is running
- Run `adb devices` to verify connection
- Check that USB debugging is enabled in emulator settings

### OpenAI API Errors
- Verify `OPENAI_API_KEY` is set in `.env`
- Check API key is valid and has credits
- Ensure internet connection is working

### Microphone Issues
- Check microphone permissions
- On Windows: Settings > Privacy > Microphone
- Test microphone with other applications

## Testing

Run basic tests:
```bash
pytest tests/test_device.py -v
```

## Next Steps

1. Test basic app launch: `python demo.py`
2. Try voice commands with the full agent
3. Add more app mappings in `config/config.yaml`
4. Customize prompts for better screen understanding
