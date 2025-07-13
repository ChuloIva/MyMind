from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.engine import Engine
import os
from typing import Generator
from ..common.config import settings

def get_engine():
    """Get database engine with fallback to SQLite."""
    try:
        database_url = settings.database_url
        if database_url.startswith("postgresql://"):
            # Try PostgreSQL first
            engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=300
            )
            # Test connection
            with engine.connect() as conn:
                pass
            print(f"✅ Connected to PostgreSQL")
            return engine
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
    
    # Fallback to SQLite
    print("Using SQLite database...")
    database_url = "sqlite:///./mymind.db"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False}
    )
    print(f"✅ Connected to SQLite: {database_url}")
    return engine

# Create engine instance
engine = get_engine()

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
