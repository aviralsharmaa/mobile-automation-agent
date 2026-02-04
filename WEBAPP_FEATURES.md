# Web App Features

## 🎨 Modern React Web Interface

A beautiful, responsive web application for interacting with the Mobile Automation Agent.

### ✨ Key Features

1. **Dual Mode Interface**
   - 🎤 **Voice Mode**: Record and send voice commands
   - ⌨️ **Text Mode**: Type commands directly

2. **Real-time Status**
   - Agent initialization status
   - Processing indicators
   - Error handling and display

3. **User Experience**
   - Modern gradient design
   - Smooth animations
   - Responsive layout (mobile-friendly)
   - Command history (text mode)
   - Example commands for quick testing

4. **Voice Mode Features**
   - One-click recording
   - Visual feedback (pulsing animation when recording)
   - Audio processing indicator
   - Command transcription display
   - Response display

5. **Text Mode Features**
   - Multi-line text input
   - Enter key to submit
   - Command history (last 5 commands)
   - Example command buttons
   - Response display

## 🚀 Getting Started

### Prerequisites
- Node.js 16+ and npm
- Python 3.11+
- Flask and Flask-CORS installed

### Installation

1. **Install Frontend Dependencies**
   ```bash
   cd webapp
   npm install
   ```

2. **Install Backend Dependencies**
   ```bash
   pip install flask flask-cors
   ```

### Running

1. **Start Backend Server**
   ```bash
   python api_server.py
   ```
   Server runs on `http://localhost:5000`

2. **Start Frontend (Development)**
   ```bash
   cd webapp
   npm start
   ```
   App opens at `http://localhost:3000`

### Production Build

```bash
cd webapp
npm run build
```

The Flask server automatically serves the built React app.

## 📱 Usage

### Voice Mode
1. Click the microphone button
2. Speak your command clearly
3. Click again to stop recording
4. Wait for processing
5. View the response

### Text Mode
1. Type your command in the text area
2. Press Enter or click Send
3. View response and history
4. Use example buttons for quick commands

## 🎯 Example Commands

- "Open ChatGPT and ask what's the capital of France"
- "Open Gmail"
- "Open Settings"
- "Open ChatGPT and login"

## 🔧 Technical Details

### Frontend Stack
- React 18
- Axios for API calls
- React Icons
- CSS3 with animations

### Backend Stack
- FastAPI REST API
- Automatic API documentation (Swagger/ReDoc)
- CORS enabled
- Thread-safe agent access
- File upload handling with multipart support

### API Endpoints
- `GET /api/health` - Health check
- `POST /api/initialize` - Initialize agent
- `POST /api/command/text` - Text command
- `POST /api/command/voice` - Voice command (multipart/form-data)

### API Documentation
FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## 🎨 Design Highlights

- Gradient backgrounds
- Smooth transitions
- Pulse animations for recording
- Color-coded status badges
- Responsive grid layout
- Modern card-based UI

## 📝 Notes

- TTS is disabled in web mode (responses are text-only)
- Microphone permission required for voice mode
- Agent must be initialized before use
- Commands are processed sequentially (thread-safe)
