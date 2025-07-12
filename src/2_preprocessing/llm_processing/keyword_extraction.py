from openai import OpenAI
import json

client = OpenAI()

PROMPT = "Return JSON: [{sentence_id, keywords:[{term,sentiment,start_ms,end_ms}]}]"

def extract(text: str):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT + text}]
    )
    return json.loads(res.choices[0].message.content)