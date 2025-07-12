from fastapi import APIRouter
from uuid import UUID

router = APIRouter()

@router.post("/analyse/{session_id}")
async def analyse_session(session_id: UUID):
    # This would call graph building and distortion analysis
    return {"message": f"Analysing session {session_id}"}
