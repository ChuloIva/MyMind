#!/usr/bin/env python3

import requests
import json

# Test API endpoints
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_create_client():
    """Test client creation"""
    try:
        client_data = {
            "name": "John Doe",
            "email": "john.doe@example.com"
        }
        response = requests.post(f"{BASE_URL}/api/clients/", json=client_data)
        print(f"Create client: {response.status_code} - {response.json()}")
        return response.status_code == 200, response.json()
    except Exception as e:
        print(f"Create client failed: {e}")
        return False, None

def test_list_clients():
    """Test listing clients"""
    try:
        response = requests.get(f"{BASE_URL}/api/clients/")
        print(f"List clients: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"List clients failed: {e}")
        return False

def test_create_session(client_id):
    """Test session creation"""
    try:
        session_data = {
            "title": "Initial Assessment",
            "notes": "First session with the client"
        }
        response = requests.post(f"{BASE_URL}/api/clients/{client_id}/sessions", json=session_data)
        print(f"Create session: {response.status_code} - {response.json()}")
        return response.status_code == 200, response.json()
    except Exception as e:
        print(f"Create session failed: {e}")
        return False, None

def test_list_sessions():
    """Test listing sessions"""
    try:
        response = requests.get(f"{BASE_URL}/api/sessions/")
        print(f"List sessions: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"List sessions failed: {e}")
        return False

def main():
    print("Testing MyMind Therapy Admin API...")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("❌ Health check failed!")
        return
    
    # Test client creation
    success, client = test_create_client()
    if not success:
        print("❌ Client creation failed!")
        return
    
    client_id = client["id"]
    print(f"✅ Created client with ID: {client_id}")
    
    # Test listing clients
    if not test_list_clients():
        print("❌ List clients failed!")
        return
    
    # Test session creation
    success, session = test_create_session(client_id)
    if not success:
        print("❌ Session creation failed!")
        return
    
    session_id = session["id"]
    print(f"✅ Created session with ID: {session_id}")
    
    # Test listing sessions
    if not test_list_sessions():
        print("❌ List sessions failed!")
        return
    
    print("\n✅ All API tests passed!")
    print("🚀 Therapy Admin system is working correctly!")

if __name__ == "__main__":
    main()