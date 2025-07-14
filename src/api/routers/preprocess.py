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
from ...database.models import Session as SessionModel, SessionSentence, SessionStatus, ClientNeedProfile, NeedCategory, LifeSegment
from ...input_processing.speech_to_text.transcribe import transcribe_with_speakers
from ...preprocessing.llm_processing.keyword_extraction import extract_session_keywords
from ...preprocessing.llm_processing.needs_extraction import NeedsExtractor
from ...common.config import settings

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
        status=SessionStatus.PROCESSING,
        created_at=datetime.utcnow()
        )
        
        # Save audio file
        audio_dir = Path(settings.audio_upload_path)
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

@router.post("/transcribe/{session_id}")
async def transcribe_session(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    num_speakers: Optional[int] = None,
    db: Session = Depends(get_session)
):
    """Start transcription and speaker diarization for a session"""
    
    # Get session from database
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.audio_file_path or not os.path.exists(session.audio_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Update session status
    session.status = SessionStatus.PROCESSING
    db.add(session)
    db.commit()
    
    # Start background transcription
    background_tasks.add_task(
        process_transcription_background,
        session_id=session_id,
        audio_path=session.audio_file_path,
        num_speakers=num_speakers
    )
    
    return {
        "session_id": str(session_id),
        "message": "Transcription started",
        "status": "processing"
    }

@router.post("/keywords/{session_id}")
async def extract_keywords(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    chunk_size: int = 3,
    db: Session = Depends(get_session)
):
    """Extract keywords and sentiment from session transcription"""
    
    # Get session
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if transcription exists
    sentences = db.query(SessionSentence).filter(
        SessionSentence.session_id == session_id
    ).all()
    
    if not sentences:
        raise HTTPException(
        status_code=404, 
        detail="No transcription found. Please transcribe the session first."
        )
    
    # Start background keyword extraction
    background_tasks.add_task(
        process_keywords_background,
        session_id=session_id,
        chunk_size=chunk_size
    )
    
    return {
        "session_id": str(session_id),
        "message": "Keyword extraction started",
        "status": "processing",
        "segments_count": len(sentences)
    }

@router.post("/needs/{session_id}")
async def extract_needs_from_session(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session)
):
    """Extract needs and life segments from session transcription."""
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if transcription exists
    sentences = db.query(SessionSentence).filter(
        SessionSentence.session_id == session_id
    ).all()
    
    if not sentences:
        raise HTTPException(
        status_code=404, 
        detail="No transcription found. Please transcribe the session first."
        )

    background_tasks.add_task(process_needs_background, session_id=session_id)
    
    return {
        "session_id": str(session_id),
        "message": "Needs extraction started",
        "status": "processing"
    }

@router.get("/status/{session_id}")
async def get_processing_status(
    session_id: UUID,
    db: Session = Depends(get_session)
):
    """Get processing status for a session"""
    
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Count processed segments
    total_sentences = db.query(SessionSentence).filter(
        SessionSentence.session_id == session_id
    ).count()
    
    processed_sentences = db.query(SessionSentence).filter(
        SessionSentence.session_id == session_id,
        SessionSentence.keywords.isnot(None)
    ).count()
    
    return {
        "session_id": str(session_id),
        "status": session.status,
        "progress": {
        "total_segments": total_sentences,
        "processed_segments": processed_sentences,
        "completion_percentage": (processed_sentences / total_sentences * 100) if total_sentences > 0 else 0
        },
        "created_at": session.created_at,
        "updated_at": session.updated_at
    }

# Background task functions
async def process_transcription_background(
    session_id: UUID, 
    audio_path: str, 
    num_speakers: Optional[int] = None
):
    """Background task for transcription and speaker diarization"""
    
    try:
        db = next(get_session())
        # Get session
        session = db.get(SessionModel, session_id)
        if not session:
        logger.error(f"Session {session_id} not found")
        return
        
        # Perform transcription with speaker diarization
        result = transcribe_with_speakers(
            audio_path=Path(audio_path),
            num_speakers=num_speakers
        )
        
        # Save transcription segments to database
        combined_segments = result.get('combined_segments', [])
        
        for i, segment in enumerate(combined_segments):
            sentence = SessionSentence(
                session_id=session_id,
                sentence_index=i,
                start_ms=int(segment.get('start', 0) * 1000),
                end_ms=int(segment.get('end', 0) * 1000),
                speaker=segment.get('speaker', 'UNKNOWN'),
                text=segment.get('text', ''),
                confidence=segment.get('avg_logprob', 0)
            )
            db.add(sentence)
        
        # Update session
        session.status = SessionStatus.COMPLETED
        session.duration_seconds = max(
            (seg.get('end', 0) for seg in combined_segments), 
            default=0
        )
        db.add(session)
        db.commit()
        
        logger.info(f"Transcription completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Transcription background task failed: {e}")
        db = next(get_session())
        # Update session status to failed
        try:
            session = db.get(SessionModel, session_id)
            if session:
                session.status = SessionStatus.FAILED
                db.add(session)
                db.commit()
        except Exception as db_e:
            logger.error(f"Failed to update session status to FAILED: {db_e}")

async def process_keywords_background(
    session_id: UUID, 
    chunk_size: int = 3
):
    """Background task for keyword extraction"""
    
    try:
        db = next(get_session())
        # Get session sentences
        sentences = db.query(SessionSentence).filter(
            SessionSentence.session_id == session_id
        ).order_by(SessionSentence.sentence_index).all()
        
        if not sentences:
            logger.error(f"No sentences found for session {session_id}")
            return
        
        # Convert to format expected by keyword extractor
        segments = []
        for sentence in sentences:
            segments.append({
                'text': sentence.text,
                'start': sentence.start_ms / 1000,
                'end': sentence.end_ms / 1000,
                'speaker': sentence.speaker
            })
        
        # Extract keywords
        from ...preprocessing.llm_processing.keyword_extraction import KeywordExtractor
        extractor = KeywordExtractor()
        processed_segments = extractor.extract_keywords_and_sentiment(segments, chunk_size)
        
        # Update database with keywords
        for i, processed_segment in enumerate(processed_segments):
            if i < len(sentences):
                sentence = sentences[i]
                sentence.keywords = processed_segment.get('keywords', [])
                sentence.sentiment_scores = processed_segment.get('sentiment_scores', {})
                db.add(sentence)
        
        db.commit()
        logger.info(f"Keyword extraction completed for session {session_id}")
        
    except Exception as e:
        logger.error(f"Keyword extraction background task failed: {e}")

async def process_needs_background(session_id: UUID):
    """Background task for needs extraction."""
    try:
        db = next(get_session())
        sentences = db.query(SessionSentence).filter(SessionSentence.session_id == session_id).order_by(SessionSentence.sentence_index).all()
        if not sentences:
        logger.error(f"No sentences found for needs extraction in session {session_id}")
        return

        segments_to_process = [{'text': s.text, 'start': s.start_ms / 1000} for s in sentences]
        
        extractor = NeedsExtractor(api_key=settings.openai_api_key)
        extracted_data = extractor.extract_needs_and_segments(segments_to_process)

        # Store results in ClientNeedProfile table
        # Get session to find client_id
        session = db.get(SessionModel, session_id)
        if not session:
        logger.error(f"Session {session_id} not found")
        return
        
        # Process extracted data and save to database
        extractions = extracted_data.get('extractions', [])
        for extraction in extractions:
        for need_data in extraction.get('extractions', []):
            # Find matching need and life segment IDs
            need_category = db.query(NeedCategory).filter(NeedCategory.need == need_data.get('need')).first()
            life_segment = db.query(LifeSegment).filter(LifeSegment.life_area == need_data.get('life_segment')).first()
            
            if need_category and life_segment:
                need_profile = ClientNeedProfile(
                    client_id=session.client_id,
                    session_id=session_id,
                    need_category_id=need_category.id,
                    life_segment_id=life_segment.id,
                    content=need_data.get('content', ''),
                    content_type=need_data.get('content_type', 'unknown'),
                    sentiment_score=need_data.get('sentiment_score', 0.0),
                    need_fulfillment_score=need_data.get('need_fulfillment_score', 0.5),
                    intensity=need_data.get('intensity', 0.5),
                    timestamp_ms=int(extraction.get('start', 0) * 1000),
                    therapeutic_relevance=need_data.get('therapeutic_relevance', 0.5)
                )
                db.add(need_profile)
        
        db.commit()
        logger.info(f"Needs extraction completed for session {session_id}")

    except Exception as e:
        logger.error(f"Needs extraction background task failed: {e}")

@router.post("/process-complete/{session_id}")
async def process_complete_session(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    num_speakers: Optional[int] = None,
    chunk_size: int = 3,
    db: Session = Depends(get_session)
):
    """Complete end-to-end processing: transcription + keywords"""
    
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.audio_file_path or not os.path.exists(session.audio_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Start complete processing
    background_tasks.add_task(
        process_complete_background,
        session_id=session_id,
        audio_path=session.audio_file_path,
        num_speakers=num_speakers,
        chunk_size=chunk_size
    )
    
    return {
        "session_id": str(session_id),
        "message": "Complete processing started",
        "status": "processing"
    }

async def process_complete_background(
    session_id: UUID,
    audio_path: str,
    num_speakers: Optional[int] = None,
    chunk_size: int = 3
):
    """Complete background processing pipeline"""
    
    try:
        # First do transcription
        await process_transcription_background(session_id, audio_path, num_speakers)
        
        # Then do keyword extraction
        await process_keywords_background(session_id, chunk_size)
        
        logger.info(f"Complete processing finished for session {session_id}")
        
    except Exception as e:
        logger.error(f"Complete processing failed for session {session_id}: {e}")
