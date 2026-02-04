# Web App Setup Guide

## Quick Start

### 1. Install Backend Dependencies

```bash
pip install fastapi uvicorn python-multipart
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd webapp
npm install
```

### 3. Start the Backend Server

In the project root:
```bash
python api_server.py
```

The API server will start on `http://localhost:5000`

### 4. Start the Frontend (Development)

In a new terminal, from the `webapp` directory:
```bash
npm start
```

The React app will open at `http://localhost:3000`

## Production Build

### Build React App

```bash
cd webapp
npm run build
```

This creates a `build` folder with production-ready files.

### Run Production Server

The FastAPI server (`api_server.py`) is configured to serve the React build files automatically. Just run:

```bash
python api_server.py
```

Or using uvicorn directly:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 5000
```

And visit `http://localhost:5000` - the FastAPI server will serve both the API and the React app.

### API Documentation

FastAPI provides automatic interactive API documentation:
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## Features

### Voice Mode
- Click the microphone button to start recording
- Speak your command
- Click again to stop recording
- The audio is sent to the backend for processing

### Text Mode
- Type your command in the text area
- Press Enter or click Send
- View response and command history
- Use example commands for quick testing

## API Endpoints

- `GET /api/health` - Check if agent is initialized
- `POST /api/initialize` - Initialize the agent
- `POST /api/command/text` - Process text command
  ```json
  {
    "command": "Open ChatGPT and ask what's the capital of France"
  }
  ```
- `POST /api/command/voice` - Process voice command (multipart/form-data with 'audio' file)

## Troubleshooting

### Microphone Permission
If voice mode doesn't work, check browser permissions for microphone access.

### CORS Issues
The FastAPI server has CORS enabled via middleware. If you see CORS errors, make sure `fastapi` and CORS middleware are properly configured.

### Agent Not Initializing
Check that:
1. ADB device is connected (`adb devices`)
2. OpenAI API key is set in `.env`
3. All Python dependencies are installed

### Port Already in Use
Change the port in `api_server.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)
```
