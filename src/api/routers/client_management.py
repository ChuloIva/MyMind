from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from uuid import UUID
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

# Database imports
from ...database.database import get_session
from ...database.models import Client, Session as SessionModel, SessionStatus

router = APIRouter(prefix="/api/clients", tags=["client_management"])

class ClientCreate(BaseModel):
    name: str
    email: Optional[str] = None

class ClientUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None

class SessionCreate(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None

@router.post("/", response_model=dict)
async def create_client(
    client_data: ClientCreate,
    db: Session = Depends(get_session)
):
    """Create a new client"""
    client = Client(
        name=client_data.name,
        email=client_data.email,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(client)
    db.commit()
    db.refresh(client)
    
    return {
        "id": str(client.id),
        "name": client.name,
        "email": client.email,
        "created_at": client.created_at.isoformat(),
        "updated_at": client.updated_at.isoformat()
    }

@router.get("/", response_model=List[dict])
async def list_clients(db: Session = Depends(get_session)):
    """List all clients"""
    clients = db.exec(select(Client)).all()
    return [
        {
            "id": str(client.id),
            "name": client.name,
            "email": client.email,
            "created_at": client.created_at.isoformat(),
            "updated_at": client.updated_at.isoformat(),
            "session_count": len(client.sessions) if client.sessions else 0
        }
        for client in clients
    ]

@router.get("/{client_id}", response_model=dict)
async def get_client(
    client_id: UUID,
    db: Session = Depends(get_session)
):
    """Get a specific client"""
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    return {
        "id": str(client.id),
        "name": client.name,
        "email": client.email,
        "created_at": client.created_at.isoformat(),
        "updated_at": client.updated_at.isoformat(),
        "session_count": len(client.sessions) if client.sessions else 0
    }

@router.put("/{client_id}", response_model=dict)
async def update_client(
    client_id: UUID,
    client_data: ClientUpdate,
    db: Session = Depends(get_session)
):
    """Update a client"""
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    if client_data.name is not None:
        client.name = client_data.name
    if client_data.email is not None:
        client.email = client_data.email
    
    client.updated_at = datetime.utcnow()
    
    db.add(client)
    db.commit()
    db.refresh(client)
    
    return {
        "id": str(client.id),
        "name": client.name,
        "email": client.email,
        "created_at": client.created_at.isoformat(),
        "updated_at": client.updated_at.isoformat()
    }

@router.delete("/{client_id}")
async def delete_client(
    client_id: UUID,
    db: Session = Depends(get_session)
):
    """Delete a client"""
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    db.delete(client)
    db.commit()
    
    return {"message": "Client deleted successfully"}

@router.post("/{client_id}/sessions", response_model=dict)
async def create_session(
    client_id: UUID,
    session_data: SessionCreate,
    db: Session = Depends(get_session)
):
    """Create a new session for a client"""
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    session = SessionModel(
        client_id=client_id,
        title=session_data.title,
        notes=session_data.notes,
        status=SessionStatus.PENDING.value,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    return {
        "id": str(session.id),
        "client_id": str(session.client_id),
        "title": session.title,
        "status": session.status,
        "notes": session.notes,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat()
    }

@router.get("/{client_id}/sessions", response_model=List[dict])
async def list_client_sessions(
    client_id: UUID,
    db: Session = Depends(get_session)
):
    """List all sessions for a specific client"""
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    
    sessions = db.exec(
        select(SessionModel)
        .where(SessionModel.client_id == client_id)
        .order_by(SessionModel.created_at.desc())
    ).all()
    
    return [
        {
            "id": str(session.id),
            "client_id": str(session.client_id),
            "title": session.title,
            "status": session.status,
            "notes": session.notes,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }
        for session in sessions
    ]