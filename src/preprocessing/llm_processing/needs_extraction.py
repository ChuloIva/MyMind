# src/2_preprocessing/llm_processing/needs_extraction.py

from openai import OpenAI
from typing import List, Dict, Any
import json

class NeedsExtractor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.needs_categories = self._load_needs_categories()
        self.life_segments = self._load_life_segments()

    def extract_needs_and_segments(self, transcript_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract needs and life segments from transcript segments"""

        prompt = f"""
        Analyze these therapy transcript segments and extract:
        1. Which universal needs or SDT needs are being expressed
        2. Which life segments/areas are being discussed
        3. The specific content (events, feelings, thoughts, behaviors)
        4. Sentiment and need fulfillment level

        Available need categories:
        {json.dumps([n['need'] for n in self.needs_categories], indent=2)}

        Available life segments:
        {json.dumps(self.life_segments, indent=2)}

        For each relevant segment, return:
        {{
            "segment_index": 0,
            "text": "original text",
            "extractions": [
                {{
                    "need": "autonomy",  // must match available needs
                    "life_segment": "work",  // must match available segments
                    "content": "feeling micromanaged by boss",
                    "content_type": "feeling",  // event|feeling|thought|behavior|relationship
                    "sentiment_score": -0.7,  // -1 to 1
                    "need_fulfillment_score": 0.2,  // 0 to 1 (low = unmet need)
                    "intensity": 0.8,  // 0 to 1
                    "therapeutic_relevance": 0.9
                }}
            ]
        }}

        Transcript segments:
        {json.dumps(transcript_segments, indent=2)}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.choices[0].message.content)
