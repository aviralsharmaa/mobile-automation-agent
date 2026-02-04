# Quick Start - Web App

## Step 1: Start the Backend Server

The FastAPI server is already running! ✅

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:5000
```

## Step 2: Build the React App (First Time Only)

Open a **new terminal** and run:

```bash
cd webapp
npm install
npm run build
```

This creates the `webapp/build` folder with the production React app.

## Step 3: Open in Browser

Open your web browser and go to:

**🌐 Main Web App:**
```
http://localhost:5000
```

**📚 API Documentation (Swagger):**
```
http://localhost:5000/docs
```

**📖 API Documentation (ReDoc):**
```
http://localhost:5000/redoc
```

## Alternative: Development Mode (Recommended for Development)

If you want to develop the React app with hot-reload:

### Terminal 1: Backend (already running)
```bash
python api_server.py
```

### Terminal 2: Frontend (new terminal)
```bash
cd webapp
npm install  # First time only
npm start
```

Then open: `http://localhost:3000` (React dev server)

The React dev server will proxy API requests to `http://localhost:5000` automatically.

## What You'll See

1. **Main Page** (`http://localhost:5000`):
   - Two mode buttons: Voice Mode and Text Mode
   - Status indicator showing if agent is ready
   - Interface to interact with the agent

2. **API Docs** (`http://localhost:5000/docs`):
   - Interactive Swagger UI
   - Test API endpoints directly
   - See request/response schemas

## Troubleshooting

### "React app not built" message?
Run: `cd webapp && npm run build`

### Port 5000 already in use?
Change port in `api_server.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=5001)
```

### Can't access localhost:5000?
Try: `http://127.0.0.1:5000`
