# /Users/ivanculo/Desktop/Projects/MyMind/src/database/models.py
# ----- START OF REPLACEMENT CODE -----

from sqlmodel import Field, SQLModel, Index, Relationship
from sqlalchemy import JSON
from uuid import UUID, uuid4
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class SessionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Client(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Use string forward reference to avoid NameError
    sessions: List["Session"] = Relationship(back_populates="client")

class Session(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id")
    title: Optional[str] = None
    status: str = Field(default=SessionStatus.PENDING.value)
    audio_file_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relationships
    client: "Client" = Relationship(back_populates="sessions")
    sentences: List["SessionSentence"] = Relationship(back_populates="session")
    
    # --- THIS IS THE FIX ---
    # Remove the Optional[] wrapper from the relationship type hint.
    # The relationship will be None if no analysis record exists.
    analysis: "SessionAnalysis" = Relationship(back_populates="session") 

class SessionSentence(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    sentence_index: int
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    confidence: Optional[float] = None
    keywords: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    sentiment_scores: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Use string forward reference for relationship
    session: "Session" = Relationship(back_populates="sentences")
    
    __table_args__ = (
        Index("idx_session_sentences", "session_id"),
        Index("idx_keywords", "keywords", postgresql_using="gin"),
        Index("idx_sentiment", "sentiment_scores", postgresql_using="gin"),
    )

class SessionAnalysis(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id", unique=True)
    
    key_themes: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    mood_assessment: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    cognitive_distortions: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    therapeutic_insights: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    progress_indicators: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    risk_factors: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    embedding_coordinates: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    concept_graph: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Use string forward reference for relationship
    session: "Session" = Relationship(back_populates="analysis")

class ClientProfile(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id", unique=True)
    
    demographics: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    therapy_goals: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    clinical_notes: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    risk_assessment: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    treatment_plan: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    baseline_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    current_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    trajectory_summary: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TherapeuticReport(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    report_type: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
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
    sources: Optional[Dict[str, Any]] = Field(default=None, sa_column=JSON)
    confidence_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_qa_session", "session_id"),
    )

Client.model_rebuild()
Session.model_rebuild()
SessionSentence.model_rebuild()
SessionAnalysis.model_rebuild()
ClientProfile.model_rebuild()
TherapeuticReport.model_rebuild()
QAInteraction.model_rebuild()
# ----- END OF REPLACEMENT CODE -----