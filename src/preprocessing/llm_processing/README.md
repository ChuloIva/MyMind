# LLM Processing with GPT-4.1-nano

This module provides advanced text processing capabilities using OpenAI's GPT-4.1-nano models for keyword extraction, sentiment analysis, and therapeutic insight generation.

## Overview

The LLM processing pipeline transforms raw therapy transcripts into structured, analyzable data by extracting semantically meaningful keywords with temporal and emotional context.

## Core Implementation

### `keyword_extraction.py`

```python
from openai import OpenAI
import json

client = OpenAI()

PROMPT = "Return JSON: [{sentence_id, keywords:[{term,sentiment,start_ms,end_ms}]}]"

def extract(text: str):
    res = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT + text}]
    )
    return json.loads(res.choices[0].message.content)
```

## Advanced Usage Examples

### Basic Keyword Extraction
```python
from keyword_extraction import extract

# Sample therapy session text
text = """
Client: I've been feeling really overwhelmed with work lately. 
The anxiety is getting worse and I can't seem to focus on anything.
Therapist: Can you tell me more about what's making you feel overwhelmed?
"""

# Extract keywords with sentiment
keywords = extract(text)
print(json.dumps(keywords, indent=2))
```

### Batch Processing
```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def extract_batch(texts: list[str]) -> list[dict]:
    """Process multiple text segments concurrently"""
    tasks = [
        async_client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT + text}]
        ) for text in texts
    ]
    
    results = await asyncio.gather(*tasks)
    return [json.loads(result.choices[0].message.content) for result in results]

# Usage
texts = ["Text segment 1...", "Text segment 2..."]
keywords_batch = asyncio.run(extract_batch(texts))
```

### Custom Therapeutic Prompts
```python
def extract_therapeutic_insights(text: str) -> dict:
    """Extract therapeutic insights with specialized prompting"""
    
    therapeutic_prompt = f"""
    Analyze this therapy session text for therapeutic insights:
    
    Text: {text}
    
    Extract and return JSON with the following structure:
    {{
      "emotional_states": [
        {{
          "emotion": "anxiety",
          "intensity": 0.8,
          "start_ms": 1200,
          "end_ms": 1800,
          "evidence": "feeling overwhelmed, can't focus"
        }}
      ],
      "cognitive_patterns": [
        {{
          "pattern": "catastrophizing",
          "confidence": 0.75,
          "example": "everything is falling apart",
          "timestamp": 2400
        }}
      ],
      "coping_mechanisms": [
        {{
          "mechanism": "avoidance",
          "adaptive": false,
          "description": "avoiding work tasks"
        }}
      ],
      "therapeutic_targets": [
        {{
          "target": "anxiety_management",
          "priority": "high",
          "interventions": ["CBT", "mindfulness"]
        }}
      ]
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",  # Use nano model for complex analysis
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": therapeutic_prompt}]
    )
    
    return json.loads(response.choices[0].message.content)
```

## Prompt Engineering

### Standard Keyword Extraction
```python
KEYWORD_PROMPT = """
Analyze the following therapy session text and extract keywords with their emotional context.

For each sentence, identify:
1. Key terms (emotions, topics, people, events)
2. Sentiment score (-1.0 to +1.0)
3. Temporal position (start/end milliseconds)
4. Therapeutic relevance (high/medium/low)

Return JSON format:
[{
  "sentence_id": 0,
  "keywords": [
    {
      "term": "anxiety",
      "sentiment": -0.7,
      "start_ms": 1200,
      "end_ms": 1800,
      "confidence": 0.92,
      "category": "emotion",
      "therapeutic_relevance": "high"
    }
  ]
}]

Text: {text}
"""
```

### Emotion-Specific Analysis
```python
EMOTION_PROMPT = """
Analyze this therapy text for emotional patterns and intensity.

Focus on:
- Primary emotions (anxiety, depression, anger, joy, etc.)
- Emotional intensity (0.0 to 1.0)
- Emotional transitions
- Underlying emotional themes

Return detailed emotional analysis in JSON format.

Text: {text}
"""
```

### Cognitive Distortion Detection
```python
CBT_PROMPT = """
Analyze this therapy text for cognitive distortions and thinking patterns.

Identify:
- All-or-nothing thinking
- Catastrophizing
- Mind reading
- Emotional reasoning
- Personalization
- Should statements

For each distortion found, provide:
- Type of distortion
- Evidence in text
- Confidence score
- Suggested reframe

Text: {text}
"""
```

## Model Selection Guidelines

### GPT-4o Mini (Default)
- **Use for**: Routine keyword extraction
- **Advantages**: Fast, cost-effective, good accuracy
- **Limitations**: Less nuanced understanding
- **Cost**: ~$0.15 per 1M tokens

### GPT-4o (Premium)
- **Use for**: Complex therapeutic analysis
- **Advantages**: Superior understanding, nuanced insights
- **Limitations**: Higher cost, slower processing
- **Cost**: ~$5.00 per 1M tokens

### Model Selection Logic
```python
def get_model_for_task(task_type: str, priority: str) -> str:
    """Select appropriate model based on task requirements"""
    
    if task_type == "keyword_extraction" and priority == "standard":
        return "gpt-4o-mini"
    elif task_type == "therapeutic_analysis" or priority == "high":
        return "gpt-4o"
    elif task_type == "batch_processing":
        return "gpt-4o-mini"
    else:
        return "gpt-4o-mini"  # Default fallback
```

## Error Handling & Retry Logic

### Robust Processing Function
```python
import time
import logging
from typing import Optional

def extract_with_retry(text: str, max_retries: int = 3) -> Optional[dict]:
    """Extract keywords with exponential backoff retry"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": PROMPT + text}]
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate response structure
            if validate_response(result):
                return result
            else:
                logging.warning(f"Invalid response structure on attempt {attempt + 1}")
                
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error on attempt {attempt + 1}: {e}")
        except Exception as e:
            logging.error(f"API error on attempt {attempt + 1}: {e}")
            
        # Exponential backoff
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    
    logging.error(f"Failed to extract keywords after {max_retries} attempts")
    return None

def validate_response(response: dict) -> bool:
    """Validate response structure"""
    if not isinstance(response, list):
        return False
    
    for item in response:
        if not all(key in item for key in ["sentence_id", "keywords"]):
            return False
        
        for keyword in item["keywords"]:
            required_fields = ["term", "sentiment", "start_ms", "end_ms"]
            if not all(field in keyword for field in required_fields):
                return False
    
    return True
```

## Performance Optimization

### Token Management
```python
def optimize_prompt_tokens(text: str, max_tokens: int = 3000) -> str:
    """Optimize text length for token limits"""
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(text) // 4
    
    if estimated_tokens > max_tokens:
        # Truncate text while preserving sentence boundaries
        sentences = text.split('. ')
        truncated_text = ""
        
        for sentence in sentences:
            if len(truncated_text + sentence) // 4 < max_tokens:
                truncated_text += sentence + ". "
            else:
                break
        
        return truncated_text.strip()
    
    return text
```

### Batch Processing with Rate Limiting
```python
import asyncio
from asyncio import Semaphore

async def process_with_rate_limit(texts: list[str], rate_limit: int = 5) -> list[dict]:
    """Process texts with rate limiting"""
    
    semaphore = Semaphore(rate_limit)
    
    async def process_single(text: str) -> dict:
        async with semaphore:
            response = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": PROMPT + text}]
            )
            return json.loads(response.choices[0].message.content)
    
    tasks = [process_single(text) for text in texts]
    return await asyncio.gather(*tasks)
```

## Quality Assurance

### Confidence Scoring
```python
def calculate_confidence(keywords: list[dict]) -> float:
    """Calculate overall confidence for extracted keywords"""
    
    if not keywords:
        return 0.0
    
    confidences = []
    for keyword_set in keywords:
        for keyword in keyword_set.get("keywords", []):
            if "confidence" in keyword:
                confidences.append(keyword["confidence"])
    
    return sum(confidences) / len(confidences) if confidences else 0.0
```

### Validation Pipeline
```python
def validate_extraction_quality(text: str, keywords: dict) -> dict:
    """Validate quality of keyword extraction"""
    
    quality_metrics = {
        "completeness": 0.0,
        "accuracy": 0.0,
        "relevance": 0.0,
        "temporal_alignment": 0.0
    }
    
    # Check completeness (are key terms covered?)
    important_terms = extract_important_terms(text)
    extracted_terms = [kw["term"] for kw_set in keywords for kw in kw_set["keywords"]]
    
    quality_metrics["completeness"] = len(set(extracted_terms) & set(important_terms)) / len(important_terms)
    
    # Additional quality checks...
    
    return quality_metrics
```

## Integration with Database

### Saving Results
```python
from src.database.models import SessionSentence
from sqlalchemy.orm import Session

def save_extracted_keywords(session_id: str, keywords: list[dict], db: Session):
    """Save extracted keywords to database"""
    
    for keyword_set in keywords:
        sentence_id = keyword_set["sentence_id"]
        
        # Update existing sentence record
        sentence = db.query(SessionSentence).filter(
            SessionSentence.session_id == session_id,
            SessionSentence.id == sentence_id
        ).first()
        
        if sentence:
            sentence.keywords = keyword_set["keywords"]
            db.commit()
```

This comprehensive implementation provides the foundation for sophisticated therapeutic text analysis, enabling downstream processing for insights and interventions.
