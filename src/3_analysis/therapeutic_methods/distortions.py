from openai import OpenAI
import json

client = OpenAI()
TEMPLATE = "Identify cognitive distortions… Return JSON: {distortions:[…]}"

def analyse(transcript: str):
    r = client.chat.completions.create(
        model="gpt-4o-large", temperature=0,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":TEMPLATE+transcript}]
    )
    return json.loads(r.choices[0].message.content)
