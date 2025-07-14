from sqlmodel import Field, SQLModel, Relationship
from uuid import UUID, uuid4
from typing import Optional, List, Dict, Any
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
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    sessions: List["Session"] = Relationship(back_populates="client")

class Session(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id")
    title: Optional[str] = None
    status: str = Field(default=SessionStatus.PENDING.value)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    client: Optional[Client] = Relationship(back_populates="sessions")
    sentences: List["SessionSentence"] = Relationship(back_populates="session")

class SessionSentence(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    sentence_index: int
    start_ms: int
    end_ms: int
    speaker: str
    text: str
    confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    session: "Session" = Relationship(back_populates="sentences")

class SessionAnalysis(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id", unique=True)
    summary: Optional[str] = None
    insights: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# src/7_database/models.py - Add these new models

class NeedCategory(SQLModel, table=True):
    """Universal needs from universal_needs.csv"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    need: str = Field(unique=True, index=True)
    origin_or_core_issue: str
    solution_or_resolution: str
    category_type: str = "universal"  # universal, sdt, custom

class LifeSegment(SQLModel, table=True):
    """Life areas and segments from life_segments.csv"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    life_area: str = Field(index=True)
    segment: str = Field(index=True)
    description: str
    what_belongs_here: str

class ClientNeedProfile(SQLModel, table=True):
    """Maps client transcript content to needs and life segments"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id")
    session_id: UUID = Field(foreign_key="session.id")
    sentence_id: Optional[UUID] = Field(foreign_key="sessionsentence.id")

    # Need mapping
    need_category_id: UUID = Field(foreign_key="needcategory.id")

    # Life segment mapping
    life_segment_id: UUID = Field(foreign_key="lifesegment.id")

    # Extracted content
    content: str  # The actual text/event/situation
    content_type: str  # event, feeling, thought, behavior, relationship

    # Sentiment and metrics
    sentiment_score: float  # -1 to 1
    need_fulfillment_score: float  # 0 to 1 (how well this need is being met)
    intensity: float  # 0 to 1 (how strongly this is expressed)

    # Temporal data
    timestamp_ms: int
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    therapeutic_relevance: float = 0.5  # 0 to 1

class ClientNeedSummary(SQLModel, table=True):
    """Aggregated view of client's needs profile"""
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id", unique=True)

    # Simplified fields - JSON serialization can be added later
    summary_data: Optional[str] = None  # JSON string for complex data

    last_updated: datetime = Field(default_factory=datetime.utcnow)