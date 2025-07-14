# 4. Profiling System

This module creates comprehensive client profiles through needs assessment and trajectory analysis, tracking therapeutic progress and providing personalized insights across multiple sessions.

## Architecture

```
profiling/
├── needs_assessment/
│   ├── summarise.py           # Trajectory analysis and client metrics
│   └── README.md             # Needs assessment documentation
├── finetuning/
│   └── README.md             # Model fine-tuning for personalization
└── README.md                 # This file
```

## Core Components

### Needs Assessment
- **Universal Assessment**: Comprehensive evaluation across life domains
- **Trajectory Analysis**: Session-to-session progress tracking
- **Client Metrics**: Standardized measurement of therapeutic outcomes
- **Personalization**: Individual client pattern recognition

### Profile Generation
- **Comprehensive Profiles**: Multi-dimensional client characterization
- **Progress Tracking**: Longitudinal therapeutic journey mapping
- **Predictive Insights**: Early warning systems and trend analysis
- **Intervention Recommendations**: Personalized therapeutic strategies

## Implementation

### `needs_assessment/summarise.py`

Current implementation uses GPT-4o for trajectory summarization:

```python
from openai import OpenAI; client = OpenAI()

def compute(client_id: UUID, transcript: str):
    prompt = "Summarise stress_index etc:" + transcript
    res = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14", response_format={"type":"json_object"},
        messages=[{"role":"user", "content": prompt}])
    return json.loads(res.choices[0].message.content)
```

## Advanced Features

### Comprehensive Needs Assessment
```python
def comprehensive_needs_assessment(client_id: UUID, session_ids: list[str]) -> dict:
    """Complete needs assessment across multiple domains"""
    
    assessment_prompt = """
    Conduct a comprehensive needs assessment based on therapy sessions:
    
    Analyze across these domains:
    1. Emotional Well-being (mood, anxiety, depression)
    2. Relationships (family, romantic, social)
    3. Work/Career (stress, satisfaction, goals)
    4. Health (physical, mental, sleep)
    5. Life Satisfaction (purpose, meaning, fulfillment)
    6. Coping Resources (strengths, skills, support)
    
    Return JSON with domain scores, needs identified, and recommendations.
    """
    
    # Aggregate session data
    all_transcripts = []
    for session_id in session_ids:
        transcript = get_session_transcript(session_id)
        all_transcripts.append(transcript)
    
    combined_text = "\n\n".join(all_transcripts)
    
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": assessment_prompt + combined_text}]
    )
    
    return json.loads(response.choices[0].message.content)
```

### Trajectory Analysis
```python
def analyze_client_trajectory(client_id: UUID, session_ids: list[str]) -> dict:
    """Analyze client's therapeutic journey over time"""
    
    trajectory_data = {
        "client_id": str(client_id),
        "session_count": len(session_ids),
        "timeline": [],
        "progress_metrics": {},
        "trend_analysis": {},
        "predictions": {}
    }
    
    for i, session_id in enumerate(session_ids):
        transcript = get_session_transcript(session_id)
        
        # Compute session metrics
        session_metrics = compute(client_id, transcript)
        
        session_data = {
            "session_number": i + 1,
            "session_id": session_id,
            "date": get_session_date(session_id),
            "metrics": session_metrics,
            "mood_score": session_metrics.get("mood_score", 0),
            "anxiety_level": session_metrics.get("anxiety_level", 0),
            "coping_effectiveness": session_metrics.get("coping_effectiveness", 0),
            "therapeutic_engagement": session_metrics.get("engagement", 0)
        }
        
        trajectory_data["timeline"].append(session_data)
    
    # Analyze trends
    trajectory_data["trend_analysis"] = analyze_trends(trajectory_data["timeline"])
    trajectory_data["progress_metrics"] = calculate_progress_metrics(trajectory_data["timeline"])
    trajectory_data["predictions"] = predict_future_outcomes(trajectory_data)
    
    return trajectory_data
```

### Progress Metrics Calculation
```python
def calculate_progress_metrics(timeline: list[dict]) -> dict:
    """Calculate various progress indicators"""
    
    metrics = {
        "overall_progress": 0.0,
        "mood_trend": 0.0,
        "anxiety_trend": 0.0,
        "coping_improvement": 0.0,
        "engagement_stability": 0.0,
        "therapeutic_gains": [],
        "areas_of_concern": []
    }
    
    if len(timeline) < 2:
        return metrics
    
    # Extract metric series
    mood_scores = [session["mood_score"] for session in timeline]
    anxiety_levels = [session["anxiety_level"] for session in timeline]
    coping_scores = [session["coping_effectiveness"] for session in timeline]
    engagement_scores = [session["therapeutic_engagement"] for session in timeline]
    
    # Calculate trends
    metrics["mood_trend"] = calculate_trend(mood_scores)
    metrics["anxiety_trend"] = calculate_trend(anxiety_levels)
    metrics["coping_improvement"] = calculate_trend(coping_scores)
    metrics["engagement_stability"] = calculate_stability(engagement_scores)
    
    # Overall progress (weighted composite)
    metrics["overall_progress"] = (
        metrics["mood_trend"] * 0.3 +
        abs(metrics["anxiety_trend"]) * 0.3 +  # Improvement = reduction in anxiety
        metrics["coping_improvement"] * 0.2 +
        metrics["engagement_stability"] * 0.2
    )
    
    # Identify gains and concerns
    metrics["therapeutic_gains"] = identify_therapeutic_gains(timeline)
    metrics["areas_of_concern"] = identify_areas_of_concern(timeline)
    
    return metrics
```

## Client Profiling

### Multi-Dimensional Profile
```python
def generate_client_profile(client_id: UUID) -> dict:
    """Generate comprehensive client profile"""
    
    # Get all sessions for client
    session_ids = get_client_sessions(client_id)
    
    profile = {
        "client_id": str(client_id),
        "profile_generated": datetime.now().isoformat(),
        "session_count": len(session_ids),
        "demographic_info": get_client_demographics(client_id),
        "clinical_presentation": {},
        "therapeutic_history": {},
        "progress_summary": {},
        "risk_factors": {},
        "strengths": {},
        "treatment_recommendations": {}
    }
    
    # Needs assessment
    needs_assessment = comprehensive_needs_assessment(client_id, session_ids)
    profile["needs_assessment"] = needs_assessment
    
    # Trajectory analysis
    trajectory = analyze_client_trajectory(client_id, session_ids)
    profile["trajectory_analysis"] = trajectory
    
    # Clinical presentation
    profile["clinical_presentation"] = extract_clinical_presentation(session_ids)
    
    # Therapeutic history
    profile["therapeutic_history"] = compile_therapeutic_history(session_ids)
    
    # Progress summary
    profile["progress_summary"] = trajectory["progress_metrics"]
    
    # Risk assessment
    profile["risk_factors"] = assess_risk_factors(trajectory, needs_assessment)
    
    # Strengths identification
    profile["strengths"] = identify_client_strengths(trajectory, needs_assessment)
    
    # Treatment recommendations
    profile["treatment_recommendations"] = generate_treatment_recommendations(profile)
    
    return profile
```

### Predictive Analytics
```python
def predict_therapeutic_outcomes(client_profile: dict) -> dict:
    """Predict future therapeutic outcomes based on current trajectory"""
    
    predictions = {
        "short_term_prognosis": {},  # Next 4 sessions
        "medium_term_outlook": {},   # Next 3 months
        "long_term_trajectory": {},  # Next 6-12 months
        "intervention_recommendations": [],
        "risk_alerts": []
    }
    
    trajectory = client_profile["trajectory_analysis"]
    progress_metrics = client_profile["progress_summary"]
    
    # Short-term predictions
    predictions["short_term_prognosis"] = {
        "mood_prediction": predict_mood_trajectory(trajectory, weeks=4),
        "anxiety_prediction": predict_anxiety_trajectory(trajectory, weeks=4),
        "engagement_prediction": predict_engagement_levels(trajectory, weeks=4),
        "therapeutic_readiness": assess_therapeutic_readiness(client_profile)
    }
    
    # Medium-term outlook
    predictions["medium_term_outlook"] = {
        "progress_likelihood": calculate_progress_likelihood(progress_metrics),
        "intervention_response": predict_intervention_response(client_profile),
        "relapse_risk": assess_relapse_risk(trajectory),
        "therapeutic_milestones": identify_upcoming_milestones(client_profile)
    }
    
    # Long-term trajectory
    predictions["long_term_trajectory"] = {
        "recovery_timeline": estimate_recovery_timeline(client_profile),
        "maintenance_needs": assess_maintenance_needs(client_profile),
        "long_term_prognosis": calculate_long_term_prognosis(client_profile)
    }
    
    # Risk alerts
    predictions["risk_alerts"] = identify_risk_alerts(client_profile)
    
    return predictions
```

## Quality Assurance

### Profile Validation
```python
def validate_client_profile(profile: dict) -> dict:
    """Validate client profile accuracy and completeness"""
    
    validation_metrics = {
        "completeness": 0.0,
        "consistency": 0.0,
        "clinical_accuracy": 0.0,
        "temporal_coherence": 0.0
    }
    
    # Check completeness
    required_fields = [
        "needs_assessment", "trajectory_analysis", "clinical_presentation",
        "progress_summary", "risk_factors", "strengths"
    ]
    
    present_fields = sum(1 for field in required_fields if field in profile)
    validation_metrics["completeness"] = present_fields / len(required_fields)
    
    # Check consistency
    validation_metrics["consistency"] = validate_internal_consistency(profile)
    
    # Clinical accuracy
    validation_metrics["clinical_accuracy"] = validate_clinical_accuracy(profile)
    
    # Temporal coherence
    validation_metrics["temporal_coherence"] = validate_temporal_coherence(profile)
    
    return validation_metrics
```

## Integration Examples

### API Integration
```python
@app.get("/api/clients/{client_id}/profile")
def get_client_profile(client_id: UUID):
    """Get comprehensive client profile"""
    
    try:
        # Generate profile
        profile = generate_client_profile(client_id)
        
        # Validate profile
        validation = validate_client_profile(profile)
        
        # Generate predictions
        predictions = predict_therapeutic_outcomes(profile)
        
        return {
            "profile": profile,
            "validation_metrics": validation,
            "predictions": predictions,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/clients/{client_id}/trajectory")
def get_client_trajectory(client_id: UUID):
    """Get client's therapeutic trajectory"""
    
    session_ids = get_client_sessions(client_id)
    trajectory = analyze_client_trajectory(client_id, session_ids)
    
    return {
        "trajectory": trajectory,
        "summary": generate_trajectory_summary(trajectory),
        "insights": extract_trajectory_insights(trajectory)
    }
```

### Dashboard Integration
```python
def generate_profiling_dashboard(client_id: UUID) -> dict:
    """Generate profiling dashboard data"""
    
    profile = generate_client_profile(client_id)
    
    dashboard_data = {
        "client_overview": {
            "session_count": profile["session_count"],
            "overall_progress": profile["progress_summary"]["overall_progress"],
            "current_phase": determine_therapeutic_phase(profile),
            "next_milestone": identify_next_milestone(profile)
        },
        "progress_charts": {
            "mood_trend": extract_mood_trend_data(profile),
            "anxiety_trend": extract_anxiety_trend_data(profile),
            "coping_improvement": extract_coping_trend_data(profile)
        },
        "risk_indicators": profile["risk_factors"],
        "strengths_summary": profile["strengths"],
        "recommendations": profile["treatment_recommendations"][:3]
    }
    
    return dashboard_data
```

This profiling system provides comprehensive client assessment capabilities, enabling clinicians to track progress, identify patterns, and make data-driven therapeutic decisions based on longitudinal analysis of client data.
