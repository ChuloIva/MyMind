#!/usr/bin/env python3
"""
View Client and Session IDs in MyMind Database
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database.database import get_session
from src.database.models import Client, Session as SessionModel
from sqlmodel import select

def view_all_ids():
    """Display all client and session IDs"""
    print("🔍 MyMind Database - Client and Session IDs")
    print("=" * 60)
    
    with next(get_session()) as db:
        # Get all clients
        clients = db.exec(select(Client)).all()
        print(f"\n👥 CLIENTS ({len(clients)} total):")
        print("-" * 40)
        for client in clients:
            print(f"Client ID: {client.id}")
            print(f"  Created: {client.created_at}")
            print(f"  Name: {client.name or 'Not set'}")
            print(f"  Email: {client.email or 'Not set'}")
            print()
        
        # Get all sessions
        sessions = db.exec(select(SessionModel)).all()
        print(f"\n📋 SESSIONS ({len(sessions)} total):")
        print("-" * 40)
        for session in sessions:
            print(f"Session ID: {session.id}")
            print(f"  Client ID: {session.client_id}")
            print(f"  Title: {session.title or 'No title'}")
            print(f"  Status: {session.status}")
            print(f"  Created: {session.created_at}")
            print()
        
        # Show relationships
        print(f"\n🔗 RELATIONSHIPS:")
        print("-" * 40)
        for client in clients:
            client_sessions = db.exec(
                select(SessionModel).where(SessionModel.client_id == client.id)
            ).all()
            print(f"Client {client.id} has {len(client_sessions)} sessions")
            for session in client_sessions:
                print(f"  - Session {session.id}: {session.title}")

if __name__ == "__main__":
    view_all_ids()

