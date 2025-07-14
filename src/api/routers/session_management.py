from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from uuid import UUID
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

# Database imports
from ...database.database import get_session
from ...database.models import Session as SessionModel, SessionStatus, Client

router = APIRouter(prefix="/api/sessions", tags=["session_management"])

class SessionUpdate(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None

@router.get("/", response_model=List[dict])
async def list_sessions(
    client_id: Optional[UUID] = None,
    db: Session = Depends(get_session)
):
    """List all sessions, optionally filtered by client_id"""
    query = select(SessionModel)
    
    if client_id:
        query = query.where(SessionModel.client_id == client_id)
    
    sessions = db.exec(query.order_by(SessionModel.created_at.desc())).all()
    
    return [
        {
            "id": str(session.id),
            "client_id": str(session.client_id),
            "title": session.title,
            "status": session.status,
            "notes": session.notes,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "client_name": session.client.name if session.client else None
        }
        for session in sessions
    ]

@router.get("/{session_id}", response_model=dict)
async def get_session(
    session_id: UUID,
    db: Session = Depends(get_session)
):
    """Get a specific session"""
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "id": str(session.id),
        "client_id": str(session.client_id),
        "title": session.title,
        "status": session.status,
        "notes": session.notes,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "client_name": session.client.name if session.client else None
    }

@router.put("/{session_id}", response_model=dict)
async def update_session(
    session_id: UUID,
    session_data: SessionUpdate,
    db: Session = Depends(get_session)
):
    """Update a session"""
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_data.title is not None:
        session.title = session_data.title
    if session_data.notes is not None:
        session.notes = session_data.notes
    if session_data.status is not None:
        session.status = session_data.status
    
    session.updated_at = datetime.utcnow()
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return {
        "id": str(session.id),
        "client_id": str(session.client_id),
        "title": session.title,
        "status": session.status,
        "notes": session.notes,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat()
    }

@router.delete("/{session_id}")
async def delete_session(
    session_id: UUID,
    db: Session = Depends(get_session)
):
    """Delete a session"""
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.delete(session)
    db.commit()
    
    return {"message": "Session deleted successfully"}

@router.get("/{session_id}/sentences", response_model=List[dict])
async def get_session_sentences(
    session_id: UUID,
    db: Session = Depends(get_session)
):
    """Get all sentences for a session"""
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return [
        {
            "id": str(sentence.id),
            "sentence_index": sentence.sentence_index,
            "start_ms": sentence.start_ms,
            "end_ms": sentence.end_ms,
            "speaker": sentence.speaker,
            "text": sentence.text,
            "confidence": sentence.confidence,
            "created_at": sentence.created_at.isoformat()
        }
        for sentence in session.sentences
    ]