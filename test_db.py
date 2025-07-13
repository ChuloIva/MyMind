#!/usr/bin/env python3
"""Simple database test script for MyMind project."""

import sys
import os
from sqlmodel import SQLModel, create_engine, Session, Field
from uuid import UUID, uuid4
from typing import Optional
from datetime import datetime

# Simple test models
class TestClient(SQLModel, table=True):
    __tablename__ = "test_clients"
    
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

def main():
    """Test database connectivity and basic operations."""
    print("🧪 Testing database connectivity...")
    
    # Use SQLite for simple testing
    database_url = "sqlite:///./test_mymind.db"
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    
    try:
        # Create tables
        print("Creating test tables...")
        SQLModel.metadata.create_all(engine)
        print("✅ Tables created successfully!")
        
        # Test basic operations
        with Session(engine) as session:
            # Create a test client
            client = TestClient(name="Test Client")
            session.add(client)
            session.commit()
            session.refresh(client)
            print(f"✅ Created client: {client.name} (ID: {client.id})")
            
            # Query the client
            found_client = session.get(TestClient, client.id)
            print(f"✅ Retrieved client: {found_client.name}")
            
            # Count clients
            count = session.query(TestClient).count()
            print(f"✅ Total clients: {count}")
        
        print("🎉 Database test passed! Your database is working correctly.")
        
        # Clean up test file
        if os.path.exists("test_mymind.db"):
            os.remove("test_mymind.db")
            print("🧹 Cleaned up test database file.")
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()