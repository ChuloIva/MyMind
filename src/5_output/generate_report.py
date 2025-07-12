from fastapi.responses import StreamingResponse
from openai import OpenAI
from .templates import build_prompt  # build from DB
from uuid import UUID
from src.common.openai_utils import to_event_stream

client = OpenAI()

def stream(session_id: UUID):
    stream = client.chat.completions.create(
        model="gpt-4o-large", stream=True,
        messages=[{"role":"user","content": build_prompt(session_id)}])
    return StreamingResponse(to_event_stream(stream), media_type="text/event-stream")
