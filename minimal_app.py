"""
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
