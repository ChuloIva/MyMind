from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
import os
from typing import Generator
from ..common.config import settings

# Create engine
engine: Engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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