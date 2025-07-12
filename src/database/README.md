# 7. Database Layer

This module provides comprehensive data management using SQLModel with PostgreSQL, featuring optimized storage for session data, transcripts, and therapeutic insights with advanced indexing and querying capabilities.

## Architecture

```
database/
├── models.py            # SQLModel database schemas
├── migrations/          # Alembic database migrations
└── README.md           # This file
```

## Core Implementation

### `models.py`

Current SQLModel schema implementation:

```python
from sqlmodel import Field, SQLModel, Index
from uuid import UUID, uuid4
from typing import Optional, Dict, Any

class Session(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID
    created_at: str # Using string for simplicity, should be datetime

class SessionSentence(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    keywords: Optional[Dict[str, Any]] = Field(default=None)   # jsonb
    chunks:   Optional[Dict[str, Any]] = Field(default=None)   # jsonb
    __table_args__ = (Index("idx_keywords", "keywords", postgresql_using="gin"),)
```

## Database Schema

### Core Tables

#### Sessions
- **Purpose**: Track therapy sessions and metadata
- **Key Fields**: session_id, client_id, date, duration, status
- **Relationships**: One-to-many with SessionSentence

#### SessionSentence
- **Purpose**: Store transcribed segments with temporal data
- **Key Fields**: text, speaker, start_ms, end_ms, keywords (JSONB)
- **Indexing**: GIN index on keywords for efficient searching

#### Clients
- **Purpose**: Store client demographic and profile data
- **Key Fields**: client_id, demographics, preferences, history
- **Relationships**: One-to-many with Sessions

## Enhanced Schema Implementation

### Complete Database Models
```python
from sqlmodel import SQLModel, Field, Relationship, Index
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4

class Client(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Demographics
    age: Optional[int] = None
    gender: Optional[str] = None
    demographics: Optional[Dict[str, Any]] = Field(default=None)
    
    # Clinical information
    clinical_notes: Optional[str] = None
    risk_factors: Optional[Dict[str, Any]] = Field(default=None)
    treatment_history: Optional[Dict[str, Any]] = Field(default=None)
    
    # Relationships
    sessions: List["Session"] = Relationship(back_populates="client")

class Session(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Session metadata
    date: datetime
    duration_minutes: Optional[int] = None
    session_type: str = "individual"  # individual, group, family
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled
    
    # Session data
    audio_file_path: Optional[str] = None
    transcript_status: str = "pending"  # pending, processing, completed, failed
    analysis_status: str = "pending"   # pending, processing, completed, failed
    
    # Relationships
    client: Client = Relationship(back_populates="sessions")
    sentences: List["SessionSentence"] = Relationship(back_populates="session")
    insights: List["SessionInsight"] = Relationship(back_populates="session")

class SessionSentence(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Temporal data
    start_ms: int
    end_ms: int
    sequence_number: int
    
    # Content
    text: str
    speaker: str  # "therapist", "client", "speaker_1", etc.
    confidence: Optional[float] = None
    
    # Analysis results
    keywords: Optional[Dict[str, Any]] = Field(default=None)
    sentiment: Optional[float] = None
    emotion_scores: Optional[Dict[str, float]] = Field(default=None)
    
    # Relationships
    session: Session = Relationship(back_populates="sentences")
    
    # Indexes
    __table_args__ = (
        Index("idx_session_time", "session_id", "start_ms"),
        Index("idx_keywords", "keywords", postgresql_using="gin"),
        Index("idx_text_search", "text", postgresql_using="gin"),
    )

class SessionInsight(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Insight metadata
    insight_type: str  # "cognitive_distortion", "schema", "progress", etc.
    confidence: float
    severity: Optional[str] = None
    
    # Insight data
    title: str
    description: str
    evidence: Optional[str] = None
    recommendations: Optional[Dict[str, Any]] = Field(default=None)
    
    # Therapeutic context
    therapeutic_method: str  # "CBT", "Schema", "DBT", etc.
    intervention_priority: str = "medium"  # low, medium, high, urgent
    
    # Relationships
    session: Session = Relationship(back_populates="insights")
    
    # Indexes
    __table_args__ = (
        Index("idx_session_insight", "session_id", "insight_type"),
        Index("idx_therapeutic_method", "therapeutic_method"),
    )

class ClientProfile(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Profile data
    needs_assessment: Dict[str, Any] = Field(default_factory=dict)
    trajectory_analysis: Dict[str, Any] = Field(default_factory=dict)
    progress_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Risk assessment
    risk_factors: Dict[str, Any] = Field(default_factory=dict)
    protective_factors: Dict[str, Any] = Field(default_factory=dict)
    
    # Treatment planning
    treatment_goals: Dict[str, Any] = Field(default_factory=dict)
    intervention_history: Dict[str, Any] = Field(default_factory=dict)
    
    # Indexes
    __table_args__ = (
        Index("idx_client_profile", "client_id"),
        Index("idx_needs_assessment", "needs_assessment", postgresql_using="gin"),
    )
```

## Database Operations

### Session Management
```python
from sqlmodel import Session, select
from typing import List, Optional

class SessionManager:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_session(self, client_id: UUID, date: datetime) -> Session:
        """Create new therapy session"""
        session = Session(
            client_id=client_id,
            date=date,
            status="scheduled"
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session
    
    def get_client_sessions(self, client_id: UUID) -> List[Session]:
        """Get all sessions for a client"""
        statement = select(Session).where(Session.client_id == client_id)
        return self.db.exec(statement).all()
    
    def update_session_status(self, session_id: UUID, status: str):
        """Update session status"""
        statement = select(Session).where(Session.id == session_id)
        session = self.db.exec(statement).first()
        if session:
            session.status = status
            session.updated_at = datetime.now()
            self.db.add(session)
            self.db.commit()
```

### Transcript Storage
```python
class TranscriptManager:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def save_transcript(self, session_id: UUID, segments: List[dict]):
        """Save transcript segments to database"""
        for i, segment in enumerate(segments):
            sentence = SessionSentence(
                session_id=session_id,
                start_ms=int(segment['start'] * 1000),
                end_ms=int(segment['end'] * 1000),
                sequence_number=i,
                text=segment['text'],
                speaker=segment.get('speaker', 'unknown'),
                confidence=segment.get('confidence', 0.0)
            )
            self.db.add(sentence)
        
        self.db.commit()
    
    def get_session_transcript(self, session_id: UUID) -> List[SessionSentence]:
        """Get complete transcript for session"""
        statement = select(SessionSentence).where(
            SessionSentence.session_id == session_id
        ).order_by(SessionSentence.start_ms)
        return self.db.exec(statement).all()
    
    def update_keywords(self, sentence_id: UUID, keywords: Dict[str, Any]):
        """Update keywords for a sentence"""
        statement = select(SessionSentence).where(SessionSentence.id == sentence_id)
        sentence = self.db.exec(statement).first()
        if sentence:
            sentence.keywords = keywords
            self.db.add(sentence)
            self.db.commit()
```

### Advanced Querying
```python
class AnalyticsManager:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def search_keywords(self, client_id: UUID, keyword: str) -> List[SessionSentence]:
        """Search for keyword across client's sessions"""
        statement = select(SessionSentence).join(Session).where(
            Session.client_id == client_id,
            SessionSentence.keywords.op('@>')({'term': keyword})
        )
        return self.db.exec(statement).all()
    
    def get_emotion_trends(self, client_id: UUID) -> List[dict]:
        """Get emotion trends over time"""
        statement = select(SessionSentence, Session.date).join(Session).where(
            Session.client_id == client_id,
            SessionSentence.emotion_scores.is_not(None)
        ).order_by(Session.date)
        
        results = self.db.exec(statement).all()
        return [
            {
                "date": session_date,
                "emotions": sentence.emotion_scores,
                "sentiment": sentence.sentiment
            }
            for sentence, session_date in results
        ]
    
    def get_progress_metrics(self, client_id: UUID) -> List[dict]:
        """Get progress metrics across sessions"""
        statement = select(SessionInsight, Session.date).join(Session).where(
            Session.client_id == client_id,
            SessionInsight.insight_type == "progress"
        ).order_by(Session.date)
        
        results = self.db.exec(statement).all()
        return [
            {
                "date": session_date,
                "insight": insight.description,
                "confidence": insight.confidence,
                "recommendations": insight.recommendations
            }
            for insight, session_date in results
        ]
```

## Performance Optimization

### Indexing Strategy
```python
# Additional indexes for common queries
class OptimizedIndexes:
    """Define optimized indexes for therapeutic queries"""
    
    indexes = [
        # Full-text search on transcripts
        Index("idx_fulltext_search", "text", postgresql_using="gin"),
        
        # Temporal queries
        Index("idx_session_timeline", "session_id", "start_ms", "end_ms"),
        
        # Client analysis
        Index("idx_client_sessions", "client_id", "created_at"),
        
        # Keyword analysis
        Index("idx_keyword_sentiment", "keywords", "sentiment"),
        
        # Therapeutic insights
        Index("idx_therapeutic_insights", "therapeutic_method", "confidence"),
        
        # Risk assessment
        Index("idx_risk_assessment", "insight_type", "severity"),
    ]
```

### Query Optimization
```python
class OptimizedQueries:
    """Optimized queries for common therapeutic analysis patterns"""
    
    @staticmethod
    def get_client_summary(db: Session, client_id: UUID) -> dict:
        """Get comprehensive client summary with optimized queries"""
        
        # Single query for session overview
        session_stats = db.exec(
            select(
                func.count(Session.id).label("total_sessions"),
                func.avg(Session.duration_minutes).label("avg_duration"),
                func.max(Session.date).label("last_session")
            ).where(Session.client_id == client_id)
        ).first()
        
        # Keyword frequency analysis
        keyword_stats = db.exec(
            select(
                func.jsonb_object_keys(SessionSentence.keywords).label("keyword"),
                func.count().label("frequency")
            ).join(Session).where(
                Session.client_id == client_id,
                SessionSentence.keywords.is_not(None)
            ).group_by(func.jsonb_object_keys(SessionSentence.keywords))
        ).all()
        
        return {
            "session_stats": session_stats._asdict(),
            "keyword_frequency": {k.keyword: k.frequency for k in keyword_stats}
        }
```

## Database Migrations

### Alembic Configuration
```python
# alembic/env.py
from alembic import context
from sqlmodel import SQLModel
from src.database.models import *

target_metadata = SQLModel.metadata

def run_migrations():
    """Run database migrations"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True
    )
    
    with context.begin_transaction():
        context.run_migrations()
```

### Migration Examples
```bash
# Create new migration
alembic revision --autogenerate -m "Add therapeutic insights table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## Integration Examples

### FastAPI Integration
```python
from fastapi import Depends
from sqlmodel import Session, create_engine

DATABASE_URL = "postgresql://user:password@localhost/mymind"
engine = create_engine(DATABASE_URL)

def get_db():
    """Database dependency for FastAPI"""
    with Session(engine) as session:
        yield session

@app.get("/api/clients/{client_id}/sessions")
def get_client_sessions(client_id: UUID, db: Session = Depends(get_db)):
    """Get all sessions for a client"""
    manager = SessionManager(db)
    sessions = manager.get_client_sessions(client_id)
    return {"sessions": sessions}
```

### Background Processing
```python
from celery import Celery

app = Celery('mymind')

@app.task
def process_session_analysis(session_id: UUID):
    """Background task for session analysis"""
    with Session(engine) as db:
        # Perform analysis
        analysis = analyze_session(session_id)
        
        # Save insights
        insight_manager = InsightManager(db)
        insight_manager.save_insights(session_id, analysis)
```

This database layer provides robust, scalable data management for the therapeutic AI system, with optimized performance for complex therapeutic analysis queries and comprehensive audit trails for clinical compliance.
