from sqlmodel import Field, SQLModel, Index, Relationship
from uuid import UUID, uuid4
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class SessionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Client(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    sessions: List["Session"] = Relationship(back_populates="client")

class Session(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id")
    title: Optional[str] = None
    status: SessionStatus = Field(default=SessionStatus.PENDING)
    audio_file_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    client: Client = Relationship(back_populates="sessions")
    sentences: List["SessionSentence"] = Relationship(back_populates="session")
    analysis: Optional["SessionAnalysis"] = Relationship(back_populates="session")

class SessionSentence(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    sentence_index: int
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    confidence: Optional[float] = None
    keywords: Optional[Dict[str, Any]] = Field(default=None)   # jsonb
    sentiment_scores: Optional[Dict[str, Any]] = Field(default=None)   # jsonb
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    session: Session = Relationship(back_populates="sentences")
    
    __table_args__ = (
        Index("idx_session_sentences", "session_id"),
        Index("idx_keywords", "keywords", postgresql_using="gin"),
        Index("idx_sentiment", "sentiment_scores", postgresql_using="gin"),
    )

class SessionAnalysis(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id", unique=True)
    
    # Analysis results
    key_themes: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    mood_assessment: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    cognitive_distortions: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    therapeutic_insights: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    progress_indicators: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    risk_factors: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    
    # Embeddings and visualization
    embedding_coordinates: Optional[Dict[str, Any]] = Field(default=None)  # jsonb for UMAP coordinates
    concept_graph: Optional[Dict[str, Any]] = Field(default=None)  # jsonb for graph data
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    session: Session = Relationship(back_populates="analysis")

class ClientProfile(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id", unique=True)
    
    # Profile data
    demographics: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    therapy_goals: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    clinical_notes: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    risk_assessment: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    treatment_plan: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    
    # Progress tracking
    baseline_metrics: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    current_metrics: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    trajectory_summary: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TherapeuticReport(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    report_type: str  # "summary", "comprehensive", "progress"
    title: str
    content: str  # markdown content
    metadata: Optional[Dict[str, Any]] = Field(default=None)  # jsonb
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_report_session", "session_id"),
        Index("idx_report_type", "report_type"),
    )

class QAInteraction(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    question: str
    answer: str
    sources: Optional[Dict[str, Any]] = Field(default=None)  # jsonb for source references
    confidence_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_qa_session", "session_id"),
    )
