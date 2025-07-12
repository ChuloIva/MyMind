#!/usr/bin/env python3
"""
MyMind Therapeutic AI System Startup Script

This script sets up and runs the complete therapeutic AI system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_dependencies():
    """Install required Python packages"""
    logger.info("Installing dependencies...")
    
    # Core packages that can be installed without additional system packages
    core_packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "pydantic==2.4.2",
        "pydantic-settings==2.0.3",
        "python-dotenv==1.0.0",
        "requests==2.31.0",
        "httpx==0.25.2",
        "typing-extensions==4.8.0"
    ]
    
    for package in core_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {package}: {e}")
    
    logger.info("Core dependencies installed")

def setup_directories():
    """Create required directories"""
    directories = [
        "./data",
        "./data/raw_audio",
        "./data/transcripts",
        "./data/processed_data",
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    
    if not env_file.exists():
        logger.info("Creating .env file...")
        
        env_content = """# MyMind Therapeutic AI Configuration

# Database Configuration
DATABASE_URL=sqlite:///./data/mymind.db

# OpenAI Configuration (Required for full functionality)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Configuration (Required for speaker diarization)
HF_TOKEN=your_huggingface_token_here

# Audio Processing
AUDIO_UPLOAD_PATH=./data/raw_audio
TRANSCRIPT_PATH=./data/transcripts
PROCESSED_DATA_PATH=./data/processed_data

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Security
SECRET_KEY=your_secret_key_here_change_in_production
ALGORITHM=HS256

# Model Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
PYANNOTE_MODEL=pyannote/speaker-diarization-3.1

# Processing Configuration
CHUNK_SIZE=3
BATCH_SIZE=4
MAX_SPEAKERS=5
"""
        
        with open(env_file, "w") as f:
            f.write(env_content)
        
        logger.info("Created .env file. Please update with your API keys!")
    else:
        logger.info(".env file already exists")

def create_minimal_app():
    """Create a minimal FastAPI app that works without all dependencies"""
    
    app_content = '''"""
MyMind Therapeutic AI System - Minimal FastAPI Application

This is a simplified version that runs without heavy ML dependencies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import uuid
from datetime import datetime
import os

app = FastAPI(
    title="MyMind Therapeutic AI",
    description="AI-powered therapeutic support system",
    version="1.0.0"
)

# In-memory storage for demo (replace with database in production)
sessions = {}
clients = {}

class SessionCreate(BaseModel):
    client_id: Optional[str] = None
    title: str = "Therapy Session"
    notes: str = ""

class SessionResponse(BaseModel):
    session_id: str
    client_id: str
    title: str
    status: str
    created_at: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Welcome page with system overview"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MyMind Therapeutic AI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .feature { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .api-endpoint { background: #e8f6f3; padding: 10px; margin: 5px 0; border-radius: 3px; font-family: monospace; }
            .status { text-align: center; padding: 20px; }
            .warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 MyMind Therapeutic AI System</h1>
            
            <div class="status">
                <h2>✅ System Status: Running</h2>
                <p>API Server is active and ready to receive requests</p>
            </div>
            
            <div class="warning">
                <h3>⚠️ Setup Required</h3>
                <p>To enable full functionality, please:</p>
                <ul>
                    <li>Add your OpenAI API key to the .env file</li>
                    <li>Add your HuggingFace token to the .env file</li>
                    <li>Install ML dependencies: <code>pip install -r requirements.txt</code></li>
                </ul>
            </div>
            
            <h2>🚀 Features</h2>
            
            <div class="feature">
                <h3>Audio Processing</h3>
                <p>Upload therapy session recordings for automatic transcription and speaker identification</p>
            </div>
            
            <div class="feature">
                <h3>AI Analysis</h3>
                <p>Extract therapeutic insights, identify cognitive patterns, and assess emotional states</p>
            </div>
            
            <div class="feature">
                <h3>Visualization</h3>
                <p>Interactive concept maps and progress tracking for therapeutic sessions</p>
            </div>
            
            <div class="feature">
                <h3>Reporting</h3>
                <p>Generate comprehensive therapeutic reports and recommendations</p>
            </div>
            
            <h2>📡 API Endpoints</h2>
            
            <div class="api-endpoint">GET /api/health - Health check</div>
            <div class="api-endpoint">POST /api/sessions - Create new session</div>
            <div class="api-endpoint">GET /api/sessions/{id} - Get session details</div>
            <div class="api-endpoint">GET /api/docs - Interactive API documentation</div>
            
            <h2>🔗 Quick Links</h2>
            <ul>
                <li><a href="/docs">📖 API Documentation</a></li>
                <li><a href="/redoc">📋 API Reference</a></li>
                <li><a href="/api/health">🔍 Health Check</a></li>
            </ul>
            
            <p style="text-align: center; margin-top: 40px; color: #7f8c8d;">
                MyMind Therapeutic AI System v1.0.0
            </p>
        </div>
    </body>
    </html>
    """
    
    return html_content

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "features": {
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "hf_configured": bool(os.getenv("HF_TOKEN")),
            "database_ready": True
        }
    }

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(session_data: SessionCreate):
    """Create a new therapy session"""
    
    session_id = str(uuid.uuid4())
    client_id = session_data.client_id or str(uuid.uuid4())
    
    session = {
        "session_id": session_id,
        "client_id": client_id,
        "title": session_data.title,
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "notes": session_data.notes
    }
    
    sessions[session_id] = session
    
    return SessionResponse(**session)

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@app.get("/api/sessions")
async def list_sessions():
    """List all sessions"""
    
    return {
        "sessions": list(sessions.values()),
        "total": len(sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open("minimal_app.py", "w") as f:
        f.write(app_content)
    
    logger.info("Created minimal_app.py")

def main():
    """Main startup function"""
    
    logger.info("🧠 Starting MyMind Therapeutic AI System...")
    
    # Check Python version
    check_python_version()
    
    # Set up directories
    setup_directories()
    
    # Create environment file
    create_env_file()
    
    # Install core dependencies
    install_dependencies()
    
    # Create minimal app
    create_minimal_app()
    
    logger.info("✅ Setup complete!")
    logger.info("")
    logger.info("🚀 To start the system:")
    logger.info("   python minimal_app.py")
    logger.info("")
    logger.info("🌐 Then visit: http://localhost:8000")
    logger.info("")
    logger.info("📖 For full functionality, install ML dependencies:")
    logger.info("   pip install -r requirements.txt")
    logger.info("")
    logger.info("🔑 Don't forget to add your API keys to .env file!")

if __name__ == "__main__":
    main()