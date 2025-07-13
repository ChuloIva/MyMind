# /src/api/routers/mvp.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlmodel import Session
import logging
from uuid import uuid4

# Your existing analysis imports
from src.preprocessing.llm_processing.keyword_extraction import KeywordExtractor
from src.analysis.therapeutic_methods.distortions import CognitiveDistortionAnalyzer

# New database imports
from src.database.database import get_session
from src.database.models import Session as SessionModel, SessionAnalysis, Client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["MVP Analysis"])

@router.post("/analyze_text_session")
async def analyze_text_and_create_session(
    file: UploadFile = File(...),
    db: Session = Depends(get_session)  # <-- Add database dependency
):
    """
    Accepts a .txt file, creates a new session record, analyzes the text,
    stores the results, and returns the session ID.
    """
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    # --- Step 1: Read the file ---
    contents = await file.read()
    text = contents.decode('utf-8')
    if not text.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        
    logger.info("Received text file for analysis.")

    try:
        # --- Step 2: Create Client and Session records in the DB ---
        # For simplicity, we create a new client every time.
        # In a real app, you'd look up an existing client.
        new_client = Client(id=uuid4())
        db.add(new_client)
        db.commit()
        db.refresh(new_client)

        new_session = SessionModel(
            id=uuid4(), 
            client_id=new_client.id, 
            title=file.filename,
            status="completed" # Since we do it all at once
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        logger.info(f"Created session {new_session.id} for client {new_client.id}")

        # --- Step 3: Run the existing AI analysis ---
        keyword_extractor = KeywordExtractor()
        keywords_analysis = keyword_extractor.extract_keywords_and_sentiment(
            [{'text': text, 'start': 0, 'end': 0}]
        )

        distortion_analyzer = CognitiveDistortionAnalyzer()
        therapeutic_analysis = distortion_analyzer.analyze_session(
            {'processed_segments': [{'text': text, 'speaker': 'client', 'start': 0}]}
        )
        logger.info("AI analysis complete.")

        # --- Step 4: Store the analysis results in the DB ---
        analysis_record = SessionAnalysis(
            session_id=new_session.id,
            cognitive_distortions=therapeutic_analysis.get("cognitive_distortions"),
            key_themes=keywords_analysis # We can store the whole keyword result here
        )
        db.add(analysis_record)
        db.commit()
        logger.info(f"Stored analysis for session {new_session.id}")
        
        # --- Step 5: Return the ID of the created session ---
        return {
            "message": "Analysis complete and stored.",
            "session_id": new_session.id,
            "client_id": new_client.id
        }

    except Exception as e:
        logger.error(f"Full analysis pipeline failed: {e}", exc_info=True)
        # Here you might want to roll back the session creation, but for now, we'll just error out.
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    from uuid import UUID # Add this to your imports at the top

@router.get("/analysis/{session_id}", response_model=SessionAnalysis)
async def get_analysis_for_session(
    session_id: UUID,
    db: Session = Depends(get_session)
):
    """
    Retrieves the stored analysis results for a given session ID.
    """
    analysis = db.query(SessionAnalysis).filter(SessionAnalysis.session_id == session_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis for this session not found.")
    
    return analysis