# src/6_api/routers/profiling.py

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from uuid import UUID
from typing import List, Dict, Any

# Database imports
from src.database.database import get_session
from src.database.models import ClientNeedSummary

# Profiling imports
from src.profiling.needs_assessment.needs_profiler import NeedsProfiler

# Output imports
from src.output.needs_report import get_client_needs_profile, NeedsProfileReport

# Analysis imports
from src.analysis.therapeutic_methods.distortions import CognitiveDistortionAnalyzer

# Router setup
router = APIRouter(prefix="/api/profiling", tags=["profiling"])

# Helper functions that need to be implemented or imported
def get_recent_sessions(client_id: UUID, session_count: int) -> List[UUID]:
    """Get recent session IDs for a client"""
    # TODO: Implement database query to get recent sessions
    # This should query the Session table for the given client_id
    # For now, return empty list as placeholder
    return []

def generate_visual_profile(client_id: UUID) -> Dict[str, Any]:
    """Generate visual profile data for client"""
    report_generator = NeedsProfileReport()
    return report_generator.generate_visual_profile(client_id)

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

def generate_therapeutic_recommendations(profile: ClientNeedSummary) -> List[Dict[str, Any]]:
    """Generate therapeutic recommendations based on profile"""
    recommendations = []
    
    # Recommendations based on unmet needs
    for need_data in profile.unmet_needs:
        need = need_data.get('need', 'Unknown')
        recommendations.append({
            'type': 'needs_focus',
            'intervention': f'Address {need.title()} needs',
            'priority': 'high',
            'description': f'Focus on interventions to meet {need} needs'
        })
    
    # Recommendations based on concerning life segments
    for segment, scores in profile.life_segment_scores.items():
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

@router.post("/clients/{client_id}/analyze-needs")
async def analyze_client_needs(
    client_id: UUID,
    session_count: int = 10,
    db: Session = Depends(get_session)
):
    """Analyze client's needs profile"""

    profiler = NeedsProfiler()
    session_ids = get_recent_sessions(client_id, session_count)

    profile = profiler.build_client_profile(client_id, session_ids)

    return {
        "client_id": str(client_id),
        "profile": profile,
        "visualization_data": generate_visual_profile(client_id)
    }

@router.get("/clients/{client_id}/needs-dashboard")
async def get_needs_dashboard(client_id: UUID):
    """Get dashboard data for needs visualization"""

    profile = get_client_needs_profile(client_id)

    return {
        "life_segments": {
            "data": profile.life_segment_scores,
            "insights": generate_segment_insights(profile.life_segment_scores)
        },
        "needs": {
            "fulfillment_scores": profile.need_fulfillment_scores,
            "unmet": profile.unmet_needs,
            "fulfilled": profile.fulfilled_needs
        },
        "recommendations": generate_therapeutic_recommendations(profile)
    }
