# src/6_api/routers/profiling.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from uuid import UUID
from typing import List, Dict, Any
import json

# Database imports
from ...database.database import get_session
from ...database.models import ClientNeedSummary, Session as SessionModel, Client

# Profiling imports
from ...profiling.needs_assessment.needs_profiler import NeedsProfiler

# Router setup
router = APIRouter(prefix="/api/profiling", tags=["profiling"])

# Helper functions
def get_recent_sessions(client_id: UUID, session_count: int, db: Session) -> List[UUID]:
    """Get recent session IDs for a client"""
    sessions = db.exec(
        select(SessionModel.id)
        .where(SessionModel.client_id == client_id)
        .order_by(SessionModel.created_at.desc())
        .limit(session_count)
    ).all()
    return [session for session in sessions]

def generate_visual_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate visual profile data for client"""
    # Create chart data structures for frontend
    life_segment_scores = profile_data.get("life_segment_scores", {})
    need_fulfillment_scores = profile_data.get("need_fulfillment_scores", {})
    
    # Radar chart data for life segments
    radar_data = {
        "labels": list(life_segment_scores.keys()),
        "datasets": [{
            "label": "Sentiment",
            "data": [scores.get("sentiment", 0) for scores in life_segment_scores.values()],
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "borderColor": "rgba(75, 192, 192, 1)"
        }, {
            "label": "Fulfillment",
            "data": [scores.get("fulfillment", 0) for scores in life_segment_scores.values()],
            "backgroundColor": "rgba(255, 99, 132, 0.2)",
            "borderColor": "rgba(255, 99, 132, 1)"
        }]
    }
    
    # Bar chart data for needs
    bar_data = {
        "labels": list(need_fulfillment_scores.keys()),
        "datasets": [{
            "label": "Need Fulfillment",
            "data": list(need_fulfillment_scores.values()),
            "backgroundColor": "rgba(54, 162, 235, 0.2)",
            "borderColor": "rgba(54, 162, 235, 1)"
        }]
    }
    
    return {
        "radar_chart": radar_data,
        "bar_chart": bar_data
    }

def generate_segment_insights(life_segment_scores: Dict[str, Dict[str, float]]) -> List[str]:
    """Generate insights from life segment scores"""
    insights = []
    for segment, scores in life_segment_scores.items():
        sentiment = scores.get('sentiment', 0)
        fulfillment = scores.get('fulfillment', 0)
        
        if sentiment < -0.3:
            insights.append(f"Challenges identified in {segment} area")
        elif sentiment > 0.3:
            insights.append(f"Positive experiences in {segment} area")
            
        if fulfillment < 0.4:
            insights.append(f"Unmet needs in {segment} require attention")
    
    return insights if insights else ["No significant patterns identified"]

def generate_therapeutic_recommendations(profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate therapeutic recommendations based on profile"""
    recommendations = []
    
    unmet_needs = profile_data.get("unmet_needs", [])
    life_segment_scores = profile_data.get("life_segment_scores", {})
    
    # Recommendations based on unmet needs
    for need_data in unmet_needs:
        need = need_data.get('need', 'Unknown')
        recommendations.append({
            'type': 'needs_focus',
            'intervention': f'Address {need.title()} needs',
            'priority': 'high',
            'description': f'Focus on interventions to meet {need} needs'
        })
    
    # Recommendations based on concerning life segments
    for segment, scores in life_segment_scores.items():
        if scores.get('sentiment', 0) < -0.2:
            recommendations.append({
                'type': 'life_segment',
                'intervention': f'Explore {segment.title()} challenges',
                'priority': 'medium',
                'description': f'Develop coping strategies for {segment} area'
            })
    
    return recommendations if recommendations else [{
        'type': 'general',
        'intervention': 'Continue current approach',
        'priority': 'low',
        'description': 'Maintain current therapeutic direction'
    }]

async def run_profiling_background(client_id: UUID, session_ids: List[UUID]):
    """Background task to run the needs profiler."""
    try:
        db = next(get_session())
        profiler = NeedsProfiler(db_session=db)
        result = profiler.build_client_profile(client_id, session_ids)
        print(f"✅ Profile generated for client {client_id}: {result}")
    except Exception as e:
        print(f"❌ Error generating profile for client {client_id}: {e}")
        raise
        
@router.post("/clients/{client_id}/analyze-needs")
async def analyze_client_needs(
    client_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session),
    session_count: int = 10
):
    """Trigger a background task to build/update a client's needs profile."""
    # Get recent sessions
    session_ids = get_recent_sessions(client_id, session_count, db)

    if not session_ids:
        raise HTTPException(status_code=404, detail="No sessions found for this client.")

    background_tasks.add_task(run_profiling_background, client_id, session_ids)
    
    return {"message": "Needs profile analysis has been triggered.", "client_id": client_id}

@router.get("/clients/{client_id}/needs-dashboard")
async def get_needs_dashboard(
    client_id: UUID, 
    db: Session = Depends(get_session)
):
    """Get dashboard data for needs visualization."""
    # Get the client's needs summary
    profile = db.exec(
        select(ClientNeedSummary).where(ClientNeedSummary.client_id == client_id)
    ).first()

    if not profile or not profile.summary_data:
        raise HTTPException(status_code=404, detail="Needs profile not generated yet. Please trigger analysis first.")

    # Parse the JSON data
    profile_data = json.loads(profile.summary_data)
    life_segment_scores = profile_data.get("life_segment_scores", {})

    return {
        "life_segments": {
            "data": life_segment_scores,
            "insights": generate_segment_insights(life_segment_scores)
        },
        "needs": {
            "fulfillment_scores": profile_data.get("need_fulfillment_scores", {}),
            "unmet": profile_data.get("unmet_needs", []),
            "fulfilled": profile_data.get("fulfilled_needs", [])
        },
        "recommendations": generate_therapeutic_recommendations(profile_data),
        "visualization_data": generate_visual_profile(profile_data)
    }

@router.get("/clients")
async def list_clients(db: Session = Depends(get_session)):
    """List all clients with their IDs for easy selection."""
    clients = db.exec(select(Client)).all()
    return [{
        "id": str(client.id),
        "name": client.name or "Unnamed Client",
        "email": client.email,
        "created_at": client.created_at.isoformat(),
        "session_count": len(client.sessions) if client.sessions else 0
    } for client in clients]

@router.get("/clients/{client_id}/sessions")
async def list_client_sessions(
    client_id: UUID,
    db: Session = Depends(get_session)
):
    """List all sessions for a specific client with their IDs."""
    sessions = db.exec(
        select(SessionModel)
        .where(SessionModel.client_id == client_id)
        .order_by(SessionModel.created_at.desc())
    ).all()
    
    if not sessions:
        raise HTTPException(status_code=404, detail="No sessions found for this client.")
    
    return [{
        "id": str(session.id),
        "title": session.title or f"Session {session.created_at.strftime('%Y-%m-%d %H:%M')}",
        "status": session.status,
        "created_at": session.created_at.isoformat(),
        "notes": session.notes
    } for session in sessions]
