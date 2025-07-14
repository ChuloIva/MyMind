#!/usr/bin/env python3

from sqlmodel import Session, select
from src.database.database import get_engine
from src.database.models import Client, Session as SessionModel

def test_database():
    """Test database operations"""
    engine = get_engine()
    
    with Session(engine) as session:
        # Test client creation
        client = Client(name="Test Client", email="test@example.com")
        session.add(client)
        session.commit()
        session.refresh(client)
        
        print(f"✅ Created client: {client.id} - {client.name}")
        
        # Test session creation
        test_session = SessionModel(
            client_id=client.id,
            title="Test Session",
            notes="This is a test session"
        )
        session.add(test_session)
        session.commit()
        session.refresh(test_session)
        
        print(f"✅ Created session: {test_session.id} - {test_session.title}")
        
        # Test querying clients
        clients = session.exec(select(Client)).all()
        print(f"✅ Found {len(clients)} clients in database")
        
        # Test querying sessions
        sessions = session.exec(select(SessionModel)).all()
        print(f"✅ Found {len(sessions)} sessions in database")
        
        print("\n✅ Database test completed successfully!")

if __name__ == "__main__":
    test_database()