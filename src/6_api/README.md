# 6. API Gateway

This module provides a high-performance FastAPI gateway that exposes all system functionality through RESTful endpoints, enabling seamless integration with frontend applications and external systems.

## Architecture

```
6_api/
├── main.py              # FastAPI application entry point
└── routers/
    ├── preprocess.py    # POST /preprocess/{session_id}
    ├── analyse.py       # POST /analyse/{session_id}
    ├── rag.py           # POST /qa/{session_id}
    └── output.py        # GET  /output/{session_id}
```

## Core Implementation

### `main.py`

Current FastAPI application setup:

```python
from fastapi import FastAPI
from .routers import preprocess, analyse, rag, output
app = FastAPI()
for r in (preprocess, analyse, rag, output):
    app.include_router(r.router)
```

## API Endpoints

### Session Processing
```python
# Process audio file for session
POST /api/sessions/{session_id}/process-audio
Content-Type: multipart/form-data

# Get session transcript
GET /api/sessions/{session_id}/transcript

# Process transcript for keywords
POST /api/sessions/{session_id}/preprocess

# Analyze session for therapeutic insights
POST /api/sessions/{session_id}/analyse
```

### Analysis & Insights
```python
# Get session visualization data
GET /api/sessions/{session_id}/visualization

# Query session using RAG
POST /api/sessions/{session_id}/query
Content-Type: application/json
{
    "query": "What are the main themes in this session?"
}

# Get therapeutic analysis
GET /api/sessions/{session_id}/therapeutic-analysis

# Stream report generation
GET /api/sessions/{session_id}/stream-report
```

### Client Management
```python
# Get client profile
GET /api/clients/{client_id}/profile

# Get client trajectory
GET /api/clients/{client_id}/trajectory

# Get progress report
GET /api/clients/{client_id}/progress-report?session_count=10
```

## Enhanced API Implementation

### Session Processing Router
```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import UUID
import logging

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

@router.post("/{session_id}/process-audio")
async def process_audio_file(session_id: UUID, audio_file: UploadFile = File(...)):
    """Process uploaded audio file for therapy session"""
    
    try:
        # Validate file type
        if not audio_file.filename.endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Save audio file
        file_path = save_audio_file(session_id, audio_file)
        
        # Process with Whisper
        from src.input_processing.speech_to_text.transcribe import transcribe
        segments = transcribe(file_path)
        
        # Save to database
        save_transcription(session_id, segments)
        
        return {
            "success": True,
            "session_id": str(session_id),
            "segments_count": len(segments),
            "duration": calculate_duration(segments)
        }
        
    except Exception as e:
        logging.error(f"Audio processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{session_id}/transcript")
async def get_session_transcript(session_id: UUID):
    """Get session transcript with speaker separation"""
    
    try:
        transcript = get_session_transcript(session_id)
        
        return {
            "session_id": str(session_id),
            "transcript": transcript,
            "speaker_count": len(transcript.get("speakers", [])),
            "duration": transcript.get("duration", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail="Session not found")
```

### Analysis Router
```python
from fastapi import APIRouter, HTTPException
from uuid import UUID

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

@router.post("/sessions/{session_id}/therapeutic-analysis")
async def analyze_session(session_id: UUID):
    """Perform comprehensive therapeutic analysis"""
    
    try:
        # Get session data
        transcript = get_session_transcript(session_id)
        
        # Perform analysis
        from src.analysis.therapeutic_methods.distortions import analyse
        analysis = analyse(transcript)
        
        # Generate insights
        insights = generate_therapeutic_insights(analysis)
        
        # Validate results
        validation = validate_analysis_quality(analysis)
        
        return {
            "session_id": str(session_id),
            "analysis": analysis,
            "insights": insights,
            "validation": validation,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/visualization")
async def get_session_visualization(session_id: UUID):
    """Get session concept visualization data"""
    
    try:
        # Generate visualization
        from src.analysis.nlp.graph_construction.graph_builder import build_session_graph
        visualization = build_session_graph(session_id)
        
        return {
            "session_id": str(session_id),
            "visualization": visualization,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### RAG Router
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import UUID

router = APIRouter(prefix="/api/rag", tags=["rag"])

class QueryRequest(BaseModel):
    query: str
    context_limit: int = 5

@router.post("/sessions/{session_id}/query")
async def query_session(session_id: UUID, request: QueryRequest):
    """Query session data using RAG"""
    
    try:
        # Get QA chain
        from src.analysis.rag.rag import get_qa_chain
        qa_chain = get_qa_chain(session_id)
        
        # Execute query
        result = qa_chain({"query": request.query})
        
        # Validate answer
        validation = validate_answer_quality(request.query, result)
        
        return {
            "session_id": str(session_id),
            "query": request.query,
            "answer": result["result"],
            "sources": [doc.metadata for doc in result.get("source_documents", [])],
            "validation": validation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/insights")
async def get_session_insights(session_id: UUID):
    """Get comprehensive session insights"""
    
    try:
        # Generate insights using RAG
        from src.analysis.rag.rag import generate_session_insights
        insights = generate_session_insights(session_id)
        
        return {
            "session_id": str(session_id),
            "insights": insights,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Output Router
```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from uuid import UUID

router = APIRouter(prefix="/api/output", tags=["output"])

@router.get("/sessions/{session_id}/report")
async def generate_session_report(session_id: UUID, report_type: str = "summary"):
    """Generate session report"""
    
    try:
        if report_type == "summary":
            report = generate_session_summary(session_id)
        elif report_type == "comprehensive":
            report = generate_comprehensive_report(session_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")
        
        return {
            "session_id": str(session_id),
            "report": report,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/stream-report")
async def stream_session_report(session_id: UUID):
    """Stream session report generation"""
    
    try:
        from src.output.generate_report import stream
        return stream(session_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Security & Authentication

### JWT Authentication
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Apply to protected routes
@router.get("/sessions/{session_id}/protected-data")
async def get_protected_data(session_id: UUID, user_id: str = Depends(verify_token)):
    """Protected endpoint example"""
    return {"data": "protected", "user_id": user_id}
```

### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@router.post("/sessions/{session_id}/analyze")
@limiter.limit("5/minute")
async def analyze_with_rate_limit(request: Request, session_id: UUID):
    """Rate-limited analysis endpoint"""
    return await analyze_session(session_id)
```

## Error Handling

### Global Exception Handler
```python
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    
    error_id = generate_error_id()
    
    # Log error
    logging.error(f"Error {error_id}: {exc}", exc_info=True)
    
    # Return user-friendly error
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_id": error_id,
            "detail": str(exc) if DEBUG else "Please contact support"
        }
    )
```

## Performance Optimization

### Caching
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

# Initialize cache
@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8")
    FastAPICache.init(RedisBackend(redis), prefix="mymind-cache")

# Cached endpoint
@router.get("/sessions/{session_id}/cached-analysis")
@cache(expire=3600)  # Cache for 1 hour
async def get_cached_analysis(session_id: UUID):
    """Cached session analysis"""
    return await analyze_session(session_id)
```

### Background Tasks
```python
from fastapi import BackgroundTasks

@router.post("/sessions/{session_id}/process-async")
async def process_session_async(session_id: UUID, background_tasks: BackgroundTasks):
    """Process session asynchronously"""
    
    # Add background task
    background_tasks.add_task(process_session_background, session_id)
    
    return {
        "message": "Processing started",
        "session_id": str(session_id),
        "status": "queued"
    }

async def process_session_background(session_id: UUID):
    """Background processing task"""
    try:
        # Perform heavy processing
        await analyze_session(session_id)
        await generate_insights(session_id)
        
        # Update status
        update_session_status(session_id, "completed")
        
    except Exception as e:
        logging.error(f"Background processing failed: {e}")
        update_session_status(session_id, "failed")
```

## API Documentation

### OpenAPI Customization
```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="MyMind Therapeutic AI API",
        version="1.0.0",
        description="AI-powered therapeutic analysis system",
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "https://mymind.ai/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

This API gateway provides comprehensive access to all system functionality, enabling scalable and secure integration with frontend applications and external therapeutic systems.
