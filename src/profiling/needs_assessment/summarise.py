from openai import OpenAI
import json
from uuid import UUID
from ....common.config import settings

def compute(client_id: UUID, transcript: str):
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = "Summarise stress_index etc:" + transcript
    res = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14", response_format={"type":"json_object"},
        messages=[{"role":"user", "content": prompt}])
    return json.loads(res.choices[0].message.content)
