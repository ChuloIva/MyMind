from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.engine import Engine
import os
from typing import Generator

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/mymind_db"
)

# Create engine
engine: Engine = create_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
    pool_pre_ping=True,
    pool_recycle=300
)

def create_db_and_tables():
    """Create database tables"""
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    """Get database session"""
    with Session(engine) as session:
        yield session

# Dependency for FastAPI
def get_db_session():
    """FastAPI dependency for database session"""
    return get_session()