from fastapi import APIRouter
from uuid import UUID

router = APIRouter()

@router.post("/preprocess/{session_id}")
async def preprocess_session(session_id: UUID):
    # This would call transcription, diarisation, and keyword extraction
    return {"message": f"Preprocessing session {session_id}"}
