from fastapi.responses import StreamingResponse
from openai import OpenAI
from .templates import build_prompt  # build from DB
from uuid import UUID
from ..common.openai_utils import to_event_stream
from ..common.config import settings

def stream(session_id: UUID):
    client = OpenAI(api_key=settings.openai_api_key)
    stream = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14", stream=True,
        messages=[{"role":"user","content": build_prompt(session_id)}])
    return StreamingResponse(to_event_stream(stream), media_type="text/event-stream")
