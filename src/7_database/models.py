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
