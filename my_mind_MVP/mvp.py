# mymind_mvp/mvp_app.py

import os
import json
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI Client
# The client automatically reads the OPENAI_API_KEY from the environment
try:
    client = OpenAI()
except Exception as e:
    logging.error(f"Failed to initialize OpenAI client. Ensure OPENAI_API_KEY is set in your .env file. Error: {e}")
    client = None

# --- FastAPI App ---
app = FastAPI(
    title="MyMind MVP",
    description="Minimal app to analyze text for keywords, distortions, and schemas."
)

# --- AI Analysis Prompts (extracted and simplified from your project) ---

KEYWORD_PROMPT = """
Analyze the following text from a therapy session. Extract key terms, their sentiment, and the context in which they appeared.
Return the result as a JSON object with a single key "keywords".
The "keywords" value should be a list of objects, where each object has:
- "term": The extracted keyword or concept.
- "sentiment": A score from -1.0 (very negative) to 1.0 (very positive).
- "context": The sentence or phrase where the term was found.

Here is the text:
---
{text}
---
"""

THERAPEUTIC_ANALYSIS_PROMPT = """
Analyze the following text from a therapy session for signs of cognitive distortions (based on CBT) and schema patterns (based on Schema Therapy).
Return the result as a single JSON object with two keys: "cognitive_distortions" and "schema_patterns".

1.  For "cognitive_distortions", provide a list of objects, each with:
    - "type": The name of the distortion (e.g., "Catastrophizing", "Mind Reading").
    - "evidence": The exact quote from the text that demonstrates the distortion.
    - "explanation": A brief explanation of why this is a distortion.

2.  For "schema_patterns", provide a list of objects, each with:
    - "schema": The name of the likely schema pattern (e.g., "Abandonment/Instability", "Defectiveness/Shame").
    - "evidence": The exact quote from the text that suggests this schema.
    - "explanation": A brief explanation of how the evidence points to this schema.

If no evidence is found for a category, return an empty list.

Here is the text:
---
{text}
---
"""

# --- Core Analysis Functions ---

async def analyze_text(text: str) -> dict:
    """
    Performs all required AI analyses on the input text.
    """
    if not client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check API key.")

    try:
        # 1. Keyword Extraction
        keyword_response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": KEYWORD_PROMPT.format(text=text)}]
        )
        keywords_result = json.loads(keyword_response.choices[0].message.content)

        # 2. Therapeutic Analysis (Distortions & Schemas)
        therapeutic_response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",  # Using the more powerful model for nuanced analysis
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": THERAPEUTIC_ANALYSIS_PROMPT.format(text=text)}]
        )
        therapeutic_result = json.loads(therapeutic_response.choices[0].message.content)

        # 3. Combine results
        final_result = {
            "keywords": keywords_result.get("keywords", []),
            "cognitive_distortions": therapeutic_result.get("cognitive_distortions", []),
            "schema_patterns": therapeutic_result.get("schema_patterns", [])
        }
        return final_result

    except Exception as e:
        logging.error(f"An error occurred during OpenAI API call: {e}")
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the main HTML page."""
    with open("index.html", "r") as f:
        return f.read()

@app.post("/analyze_text")
async def handle_text_analysis(file: UploadFile = File(...)):
    """
    Accepts a .txt file, analyzes it, and returns a JSON with all insights.
    """
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .txt file.")

    contents = await file.read()
    text = contents.decode('utf-8')

    if not text.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    analysis_results = await analyze_text(text)
    return analysis_results

# --- Main execution (for uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)