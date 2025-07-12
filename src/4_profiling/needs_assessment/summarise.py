from openai import OpenAI
import json
from uuid import UUID

client = OpenAI()

def compute(client_id: UUID, transcript: str):
    prompt = "Summarise stress_index etc:" + transcript
    res = client.chat.completions.create(
        model="gpt-4o-mini", response_format={"type":"json_object"},
        messages=[{"role":"user", "content": prompt}])
    return json.loads(res.choices[0].message.content)
