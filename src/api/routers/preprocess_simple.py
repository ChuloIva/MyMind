from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlmodel import Session
from uuid import UUID, uuid4
from pathlib import Path
import shutil
import logging
import os
from typing import Optional
from datetime import datetime
import re

# Local imports
from ...database.database import get_session
from ...database.models import Session as SessionModel, SessionStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/preprocess", tags=["preprocessing"])

def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent security risks."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

@router.post("/upload-audio")
async def upload_audio_file(
    file: UploadFile = File(...),
    client_id: Optional[UUID] = None,
    num_speakers: Optional[int] = None,
    db: Session = Depends(get_session)
):
    """Upload and process audio file for therapy session"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported audio format. Please use WAV, MP3, M4A, FLAC, or OGG."
        )
    
    try:
        # Create session record
        session_id = uuid4()
        sanitized_filename = sanitize_filename(file.filename)
        session = SessionModel(
            id=session_id,
            client_id=client_id or uuid4(),
            title=sanitized_filename,
            status=SessionStatus.PROCESSING.value,
            created_at=datetime.utcnow()
        )
        
        # Save audio file
        audio_dir = Path("./uploads")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        file_extension = Path(sanitized_filename).suffix
        audio_path = audio_dir / f"{session_id}{file_extension}"
        
        # Save uploaded file
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        session.audio_file_path = str(audio_path)
        
        # Save session to database
        db.add(session)
        db.commit()
        db.refresh(session)
        
        logger.info(f"Audio file uploaded for session {session_id}")
        
        return {
            "session_id": str(session_id),
            "message": "Audio file uploaded successfully",
            "status": "uploaded",
            "file_path": str(audio_path)
        }
        
    except Exception as e:
        logger.error(f"Audio upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/status/{session_id}")
async def get_processing_status(
    session_id: UUID,
    db: Session = Depends(get_session)
):
    """Get processing status for a session"""
    
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": str(session_id),
        "status": session.status,
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }