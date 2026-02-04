"""
API Server for Mobile Automation Agent Web Interface
Provides REST API endpoints for voice and text-based interactions using FastAPI
"""
import os
import json
import base64
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from src.agent.orchestrator import AgentOrchestrator
from src.utils.config import Config
from src.utils.logging import logger
import threading
import queue
import tempfile

# Initialize FastAPI app
app = FastAPI(title="Mobile Automation Agent API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Webapp removed - API-only server

# Global agent instance
agent = None
agent_lock = threading.Lock()
response_queue = queue.Queue()

# Pydantic models for request/response
class TextCommandRequest(BaseModel):
    command: str

class InitializeResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class CommandResponse(BaseModel):
    success: bool
    command: Optional[str] = None
    result: Optional[str] = None
    response: Optional[str] = None  # TTS response text (what would be spoken)
    extracted_info: Optional[str] = None  # Extracted information (e.g., ChatGPT answer)
    extracted_response: Optional[str] = None  # Response from query execution
    conversational: Optional[bool] = False
    login_flow: Optional[bool] = False
    awaiting_email: Optional[bool] = False
    awaiting_password: Optional[bool] = False
    login_complete: Optional[bool] = False
    session_active: Optional[bool] = False
    session_ended: Optional[bool] = False
    action_success: Optional[bool] = None
    task_complete: Optional[bool] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    initialized: bool
    status: str
    current_task: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    agent_initialized: bool

def initialize_agent():
    """Initialize the agent orchestrator"""
    global agent
    if agent is None:
        try:
            config = Config()
            agent = AgentOrchestrator(config)
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    return agent

@app.get("/", include_in_schema=True)
async def root():
    """API root endpoint"""
    return {
        "message": "Mobile Automation Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "initialize": "/api/initialize",
            "command": "/api/command",
            "status": "/api/status"
        }
    }

@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent_initialized=agent is not None
    )

@app.post("/api/initialize", response_model=InitializeResponse)
async def initialize():
    """Initialize the agent"""
    try:
        initialize_agent()
        return InitializeResponse(
            success=True,
            message="Agent initialized successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/api/command/text", response_model=CommandResponse)
async def process_text_command(request: TextCommandRequest):
    """Process text-based command"""
    try:
        command = request.command.strip()
        
        if not command:
            raise HTTPException(
                status_code=400,
                detail="No command provided"
            )
        
        if agent is None:
            initialize_agent()
        
        logger.info(f"Processing text command: {command}")
        
        # Process command (exactly like main.py does)
        # TTS audio won't play in API mode, but response text is captured in result["response"]
        with agent_lock:
            result = agent.process_command(command)
        
        # Extract all response fields (matching main.py behavior exactly)
        # The "response" field contains the TTS text that would be spoken (set in _respond node)
        # Determine success: no error AND (action succeeded OR conversational OR session ended)
        success = result.get("error") is None and (
            result.get("action_success", False) or 
            result.get("conversational", False) or
            result.get("session_ended", False) or
            result.get("login_complete", False)
        )
        
        # Format response with all fields from process_command (matching main.py return structure)
        return CommandResponse(
            success=success,
            command=command,
            result=result.get("extracted_info", ""),  # Legacy field for compatibility
            response=result.get("response", ""),  # TTS response text (what would be spoken)
            extracted_info=result.get("extracted_info", ""),  # Extracted information
            extracted_response=result.get("extracted_response", ""),  # Response from query execution
            conversational=result.get("conversational", False),
            login_flow=result.get("login_flow", False),
            awaiting_email=result.get("awaiting_email", False),
            awaiting_password=result.get("awaiting_password", False),
            login_complete=result.get("login_complete", False),
            session_active=result.get("session_active", False),
            session_ended=result.get("session_ended", False),
            action_success=result.get("action_success"),
            task_complete=result.get("task_complete"),
            error=result.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing text command: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/api/command/voice", response_model=CommandResponse)
async def process_voice_command(audio: UploadFile = File(...)):
    """Process voice command from audio file"""
    temp_path = None
    try:
        if agent is None:
            initialize_agent()
        
        # Save audio temporarily
        suffix = os.path.splitext(audio.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            temp_path = tmp_file.name
            content = await audio.read()
            tmp_file.write(content)
        
        # Convert audio to text using agent's STT
        logger.info("Converting audio to text...")
        text_command = agent.stt.listen_from_file(temp_path)
        
        if not text_command:
            raise HTTPException(
                status_code=400,
                detail="Could not transcribe audio"
            )
        
        logger.info(f"Transcribed command: {text_command}")
        
        # Process command (exactly like main.py does)
        # TTS audio won't play in API mode, but response text is captured in result["response"]
        with agent_lock:
            result = agent.process_command(text_command)
        
        # Extract all response fields (matching main.py behavior exactly)
        # Determine success: no error AND (action succeeded OR conversational OR session ended)
        success = result.get("error") is None and (
            result.get("action_success", False) or 
            result.get("conversational", False) or
            result.get("session_ended", False) or
            result.get("login_complete", False)
        )
        
        # Format response with all fields from process_command (matching main.py return structure)
        return CommandResponse(
            success=success,
            command=text_command,
            result=result.get("extracted_info", ""),  # Legacy field for compatibility
            response=result.get("response", ""),  # TTS response text (what would be spoken)
            extracted_info=result.get("extracted_info", ""),  # Extracted information
            extracted_response=result.get("extracted_response", ""),  # Response from query execution
            conversational=result.get("conversational", False),
            login_flow=result.get("login_flow", False),
            awaiting_email=result.get("awaiting_email", False),
            awaiting_password=result.get("awaiting_password", False),
            login_complete=result.get("login_complete", False),
            session_active=result.get("session_active", False),
            session_ended=result.get("session_ended", False),
            action_success=result.get("action_success"),
            task_complete=result.get("task_complete"),
            error=result.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing voice command: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {e}")

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get current agent status"""
    if agent is None:
        return StatusResponse(
            initialized=False,
            status="not_initialized"
        )
    
    state = agent.state_manager.get_current_state()
    return StatusResponse(
        initialized=True,
        status=state.name if state else "idle",
        current_task=agent.state_manager.get_current_task()
    )

# API-only server - no webapp routes

if __name__ == '__main__':
    import uvicorn
    # Get port from environment (Cloud Run sets PORT env var)
    port = int(os.getenv("PORT", 5000))
    
    # Initialize agent on startup
    try:
        initialize_agent()
        logger.info(f"FastAPI Server starting on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
