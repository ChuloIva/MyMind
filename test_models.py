#!/usr/bin/env python3
"""Test script to verify database models work correctly."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlmodel import Session, select
from src.database.database import engine
from src.database.models import Client, Session as SessionModel, SessionAnalysis, SessionStatus

def test_basic_operations():
    """Test basic CRUD operations with the models."""
    print("🧪 Testing basic database operations...")
    
    with Session(engine) as session:
        # Create a client
        client = Client(name="John Doe", email="john@example.com")
        session.add(client)
        session.commit()
        session.refresh(client)
        print(f"✅ Created client: {client.name} (ID: {client.id})")
        
        # Create a session for the client
        therapy_session = SessionModel(
            client_id=client.id,
            title="Initial Consultation",
            status=SessionStatus.PENDING.value,
            notes="First session with the client"
        )
        session.add(therapy_session)
        session.commit()
        session.refresh(therapy_session)
        print(f"✅ Created session: {therapy_session.title} (ID: {therapy_session.id})")
        
        # Create analysis for the session
        analysis = SessionAnalysis(
            session_id=therapy_session.id,
            summary="Initial assessment completed",
            insights="Client shows good engagement"
        )
        session.add(analysis)
        session.commit()
        session.refresh(analysis)
        print(f"✅ Created analysis for session (ID: {analysis.id})")
        
        # Test queries
        # Find all clients
        clients = session.exec(select(Client)).all()
        print(f"✅ Found {len(clients)} clients")
        
        # Find sessions for a client
        client_sessions = session.exec(
            select(SessionModel).where(SessionModel.client_id == client.id)
        ).all()
        print(f"✅ Found {len(client_sessions)} sessions for client")
        
        # Find analysis for a session
        session_analysis = session.exec(
            select(SessionAnalysis).where(SessionAnalysis.session_id == therapy_session.id)
        ).first()
        print(f"✅ Found analysis: {session_analysis.summary}")
        
        print("🎉 All database operations completed successfully!")

def main():
    """Run the test."""
    try:
        test_basic_operations()
        print("\n✅ Database models are working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()