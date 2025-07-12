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

class Client(BaseModel):
    client_id: str
    created_at: str

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
    with open("index.html") as f:
        return f.read()

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
    
    if session_data.client_id:
        if session_data.client_id not in clients:
            raise HTTPException(status_code=404, detail="Client not found")
        client_id = session_data.client_id
    else:
        client_id = str(uuid.uuid4())
        clients[client_id] = Client(client_id=client_id, created_at=datetime.now().isoformat())

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
