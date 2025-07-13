#!/usr/bin/env python3
"""Database initialization script for MyMind project."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlmodel import SQLModel
from src.database.database import engine, create_db_and_tables

# Import all models to ensure they're registered
from src.database.models import Client, Session, SessionAnalysis

def main():
    """Initialize the database and create all tables."""
    print("Initializing MyMind database...")
    
    try:
        # Create all tables
        create_db_and_tables()
        print("✅ Database tables created successfully!")
        
        # Test basic connection
        with engine.connect() as conn:
            print("✅ Database connection test passed!")
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()