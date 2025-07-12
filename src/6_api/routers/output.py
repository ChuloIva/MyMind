from fastapi import APIRouter
from uuid import UUID
from src.5_output.generate_report import stream

router = APIRouter()

@router.get("/output/{session_id}")
async def get_output(session_id: UUID):
    return stream(session_id)
