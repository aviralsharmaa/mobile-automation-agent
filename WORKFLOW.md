# Mobile Automation Agent - Workflow Documentation

## Overview

The Mobile Automation Agent is an AI-powered assistant that helps blind users control Android devices through natural language voice commands. It uses computer vision, speech recognition, and device automation to understand screens and execute actions.

## Architecture

```
┌─────────────────┐
│   Voice Input   │ (Microphone)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   STT (Whisper) │ Speech-to-Text conversion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Intent Parser  │ Extract action, app name, query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Screen Capture │ ADB screenshot
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vision Analysis │ GPT-4o Vision API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Action Execute │ ADB commands (tap, type, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   TTS Output    │ ElevenLabs/pyttsx3 audio feedback
└─────────────────┘
```

## Main Workflow (LangGraph State Machine)

The agent uses a state machine with the following states:

```
START
  │
  ▼
OBSERVE ──► ANALYZE ──► [Auth Needed?] ──► ACT ──► VERIFY ──► [Confirmation Needed?] ──► RESPOND ──► END
  │           │              │              │          │                    │
  │           │              │              │          │                    │
  │           │              │              │          │                    │
  │           │              │              │          │                    │
  │           │              │              │          │                    │
  │           │              ▼              │          │                    │
  │           │         AUTHENTICATE        │          │                    │
  │           │              │              │          │                    │
  │           │              └──────────────┘          │                    │
  │           │                                         │                    │
  │           │                                         │                    │
  │           │                                         ▼                    │
  │           │                                    CONFIRM_ACTION            │
  │           │                                         │                    │
  │           │                                         └────────────────────┘
  │           │
  │           └─────────────────────────────────────────────────────────────┘
  │                                    (on error)
  │
  └─────────────────────────────────────────────────────────────────────────┘
                              (retry loop, max 5 iterations)
```

## Detailed Step-by-Step Workflow

### 1. **Voice Input (STT - Speech-to-Text)**

**Location:** `src/voice/stt.py`

**Process:**
- User speaks into microphone
- Audio captured with pause detection (1.2s silence = end of speech)
- Sent to **Whisper API** (or Google Speech Recognition as fallback)
- Text correction applied (e.g., "chargerbt" → "chatgpt")
- Returns recognized text

**Example:**
```
User says: "Open ChatGPT"
Whisper hears: "open chargerbt"
Correction: "open chatgpt"
Output: "open chatgpt"
```

### 2. **Intent Parsing**

**Location:** `src/agent/orchestrator.py` → `_parse_intent()`

**Process:**
- Checks if input is conversational (greetings, questions) vs command
- If conversational → responds immediately, no device action
- If command → extracts:
  - **Action**: "open_app", "search", "query", "extract"
  - **App name**: "chatgpt", "gmail", "settings"
  - **Query**: "what is the capital of France"

**Example:**
```
Input: "Open ChatGPT and ask what is the capital of France"
Parsed:
  - action: "open_app"
  - app: "chatgpt"
  - query: "what is the capital of France"
```

### 3. **Screen Observation**

**Location:** `src/agent/orchestrator.py` → `_observe()`

**Process:**
- Captures screenshot via ADB: `device.screencap()`
- Sends screenshot to **GPT-4o Vision API**
- Analyzes screen content:
  - Main content description
  - Detected buttons, input fields, links
  - Login screen detection
  - Popup detection
  - Primary action button identification

**Output:**
```json
{
  "description": "Welcome to ChatGPT. There's a Continue button visible.",
  "is_login_screen": false,
  "has_popup": false,
  "primary_action": "Continue button",
  "elements": [
    {"description": "Continue button", "x": 540, "y": 1800, "type": "button"},
    {"description": "Welcome text", "x": 540, "y": 800, "type": "other"}
  ]
}
```

### 4. **Screen Analysis**

**Location:** `src/agent/orchestrator.py` → `_analyze()`

**Process:**
- Determines if authentication is needed
- Parses user intent into structured format
- Prepares action plan

**Decision Logic:**
- If `is_login_screen == true` → Go to AUTHENTICATE
- Otherwise → Go to ACT

### 5. **Action Execution**

**Location:** `src/agent/orchestrator.py` → `_act()`

**Actions Supported:**

#### A. Open App (`open_app`)
```python
1. Extract app name from intent
2. Use AppLauncher to find package name (with fuzzy matching)
3. Launch app: adb shell monkey -p <package> -c android.intent.category.LAUNCHER 1
4. Wait for screen to load (3 seconds)
```

#### B. Search/Query (`search`, `query`)
```python
1. Find search field using ElementDetector
2. Tap on search field
3. Type query text
4. Press Enter
```

#### C. Extract Information (`extract`)
```python
1. Get screen description
2. Return extracted information
```

### 6. **Verification & Screen Analysis**

**Location:** `src/agent/orchestrator.py` → `_verify()`

**Process:**
- After opening an app, captures screen again
- Analyzes what's displayed:
  - Welcome messages
  - Buttons (Continue, Get Started, Next, etc.)
  - Input fields
  - Primary action available

**Decision:**
- If important buttons detected → Set `needs_confirmation = True`
- Otherwise → Mark task as complete

### 7. **User Confirmation (NEW)**

**Location:** `src/agent/orchestrator.py` → `_confirm_action()`

**Process:**
1. **Describe Screen:**
   ```
   "I can see: Welcome to ChatGPT. There's a Continue button visible."
   ```

2. **Ask for Confirmation:**
   ```
   "Shall I proceed to continue?"
   ```

3. **Listen for Response:**
   - Waits for user voice input
   - Accepts: "yes", "okay", "continue", "proceed", "go ahead", "sure"

4. **Execute if Confirmed:**
   - Finds button coordinates
   - Taps button: `adb shell input tap <x> <y>`
   - Waits for screen change

**Example Flow:**
```
Agent: "I can see: Welcome to ChatGPT. There's a Continue button visible. Shall I proceed to continue?"
User: "Okay continue"
Agent: "Proceeding..."
[CLICKS CONTINUE BUTTON]
Agent: "Action completed successfully."
```

### 8. **Response**

**Location:** `src/agent/orchestrator.py` → `_respond()`

**Process:**
- Provides audio feedback via TTS
- Speaks results or errors
- Marks task as complete

**Responses:**
- Success: "Task completed successfully."
- Error: "I encountered an error: <error message>"
- Extracted info: "Here's what I found: <info>"

## Component Details

### Speech-to-Text (STT)

**Backend:** Whisper API (OpenAI) or Google Speech Recognition

**Features:**
- Pause detection (1.2s silence = end of speech)
- Text correction dictionary (fixes misrecognitions)
- Fuzzy matching for app names
- Context-aware corrections

**Correction Examples:**
- "chargerbt" → "chatgpt"
- "gmit" → "gmail"
- "charge pt" → "chatgpt"

### Text-to-Speech (TTS)

**Backend:** ElevenLabs API (high quality) or pyttsx3 (fallback)

**Features:**
- Natural voice synthesis
- Audio playback via pygame
- Fallback to system TTS if API fails

### Screen Analysis (Vision)

**Backend:** GPT-4o Vision API

**What it does:**
1. **Takes a screenshot** of the Android device screen via ADB
2. **Sends screenshot to GPT-4o Vision API** with a detailed prompt
3. **GPT-4o analyzes the image** and returns:
   - Text description of what's on screen
   - List of all UI elements (buttons, text fields, links)
   - Coordinates (x, y) for each element
   - Identification of primary action button
   - Login screen detection
   - Popup detection

**Capabilities:**
- Screen content description (e.g., "Welcome to ChatGPT with Continue button")
- UI element detection (buttons, text fields, links)
- Button coordinate extraction (x, y positions)
- Login screen detection and stage identification
- Primary action identification (most important button)
- Element type classification (button, text_field, link, etc.)

**Prompt Structure:**
```
Analyze this Android screen:
- Describe main content (welcome messages, text, etc.)
- Identify all clickable elements
- Provide coordinates (x, y) for 1080x2400 screen
- Identify primary action button
- Detect login screens and their stage
```

**Example Output:**
```json
{
  "description": "Welcome to ChatGPT. There's a Continue button visible.",
  "primary_action": "Continue button",
  "elements": [
    {"description": "Continue button", "x": 540, "y": 1800, "type": "button"},
    {"description": "Welcome text", "x": 540, "y": 800, "type": "other"}
  ]
}
```

**Why it's needed:**
- Blind users can't see the screen
- Agent needs to "see" what's displayed to make decisions
- Coordinates allow agent to tap buttons/fields at correct locations

### Device Control (ADB)

**Commands Used:**
- `screencap` - Capture screenshot
- `shell input tap <x> <y>` - Tap at coordinates
- `shell input text "<text>"` - Type text
- `shell input keyevent KEYCODE_ENTER` - Press Enter
- `shell monkey -p <package> -c android.intent.category.LAUNCHER 1` - Launch app

**How Button Clicking Works:**
1. Vision API provides button coordinates (x, y)
2. Coordinates validated (must be > 0, within screen bounds)
3. ADB command executed: `adb shell input tap <x> <y>`
4. Command result checked for errors
5. Screen waits 2 seconds for UI to update

**Common Issues:**
- Invalid coordinates (0, 0) → Button not found
- Coordinates out of bounds → Clamped to screen edges
- Button moved/changed → Re-analyze screen before clicking
- ADB command fails → Check device connection

## Example Complete Workflow

### Scenario: "Open ChatGPT"

```
1. USER: "Open ChatGPT"
   │
   ▼
2. STT: Recognizes "open chatgpt" (after correction)
   │
   ▼
3. INTENT PARSER: 
   - action: "open_app"
   - app: "chatgpt"
   │
   ▼
4. OBSERVE: Capture current screen
   │
   ▼
5. ANALYZE: Parse intent, check for auth needed
   │
   ▼
6. ACT: Launch ChatGPT app
   - AppLauncher.get_package_name("chatgpt") → "com.openai.chatgpt"
   - Execute: adb shell monkey -p com.openai.chatgpt ...
   │
   ▼
7. VERIFY: Capture new screen after app launch
   - Analyze: "Welcome to ChatGPT. Continue button visible."
   - Set needs_confirmation = True
   │
   ▼
8. CONFIRM_ACTION:
   - Agent: "I can see: Welcome to ChatGPT. There's a Continue button visible. Shall I proceed to continue?"
   - User: "Okay continue"
   - Agent: Finds Continue button coordinates (540, 1800)
   - Agent: Taps button
   │
   ▼
9. RESPOND:
   - Agent: "Action completed successfully."
   - Task complete
```

## Error Handling

### Retry Logic
- Maximum 1 retry attempt (configurable)
- No infinite loops (max 5 workflow iterations)
- Errors shown immediately

### Error Recovery
- Popup detection and dismissal
- Element not found → try scrolling
- Network errors → wait and retry
- Unknown errors → show error message

## Configuration

**File:** `config/config.yaml`

**Key Settings:**
- `voice.stt_backend`: "whisper" or "google"
- `voice.tts_backend`: "elevenlabs" or "pyttsx3"
- `agent.retry_attempts`: 1 (reduced for faster feedback)
- `agent.screenshot_delay`: 3.0 seconds (wait after app launch)

## State Management

**States Tracked:**
- Current task ID
- Screen analysis results
- Intent parts (action, app, query)
- Authentication state
- Error state
- Confirmation needed flag
- Important buttons detected

## Security

- Credentials never logged (redacted in logs)
- Credentials not persisted
- Secure credential prompting via voice
- No sensitive data in error messages

## Performance Optimizations

1. **Reduced Retries:** Only 1 retry attempt (was 3)
2. **Fast Fail:** Errors shown immediately, no long waits
3. **Smart Screen Analysis:** Only analyzes when needed
4. **Fuzzy Matching:** Handles misrecognitions quickly
5. **Direct Listening:** No wake word required (faster)

## Future Enhancements

- Multi-step task planning
- Context memory across sessions
- Better error recovery strategies
- Support for more complex actions
- Cloud device integration (after local development)
