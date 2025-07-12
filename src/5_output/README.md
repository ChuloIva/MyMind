# 5. Output & Report Generation

This module generates comprehensive therapeutic reports and insights using streaming AI technology, providing real-time analysis and professional documentation for therapy sessions.

## Core Implementation

### `generate_report.py`

Current implementation uses GPT-4o with streaming capabilities:

```python
from fastapi.responses import StreamingResponse
from openai import OpenAI
from .templates import build_prompt  # build from DB
from uuid import UUID
from src.common.openai_utils import to_event_stream

client = OpenAI()

def stream(session_id: UUID):
    stream = client.chat.completions.create(
        model="gpt-4o-large", stream=True,
        messages=[{"role":"user","content": build_prompt(session_id)}])
    return StreamingResponse(to_event_stream(stream), media_type="text/event-stream")
```

## Key Features

### Streaming Report Generation
- **Real-time Insights**: Live generation of therapeutic analysis
- **Progressive Display**: Results appear as they're generated
- **Professional Format**: Clinical-grade documentation
- **Customizable Templates**: Flexible report structures

### Report Types
- **Session Summaries**: Comprehensive session analysis
- **Progress Reports**: Multi-session trajectory analysis
- **Clinical Assessments**: Diagnostic and therapeutic evaluations
- **Intervention Recommendations**: Evidence-based treatment suggestions

## Advanced Features

### Comprehensive Report Generation
```python
def generate_comprehensive_report(session_id: UUID) -> dict:
    """Generate complete therapeutic report for session"""
    
    report_prompt = f"""
    Generate a comprehensive therapeutic report for session {session_id}.
    
    Include:
    1. Session Overview
    2. Key Themes and Patterns
    3. Cognitive and Emotional Assessment
    4. Therapeutic Progress
    5. Clinical Observations
    6. Intervention Recommendations
    7. Next Steps and Goals
    
    Format as professional clinical documentation.
    """
    
    # Build context from session data
    context = build_session_context(session_id)
    full_prompt = report_prompt + "\n\nSession Context:\n" + context
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    return {
        "session_id": str(session_id),
        "report": response.choices[0].message.content,
        "generated_at": datetime.now().isoformat(),
        "report_type": "comprehensive"
    }
```

### Progress Report Generation
```python
def generate_progress_report(client_id: UUID, session_ids: list[str]) -> dict:
    """Generate multi-session progress report"""
    
    progress_prompt = f"""
    Generate a therapeutic progress report for client {client_id} 
    covering {len(session_ids)} sessions.
    
    Analyze:
    1. Overall Progress Trajectory
    2. Therapeutic Gains and Improvements
    3. Persistent Challenges
    4. Intervention Effectiveness
    5. Future Recommendations
    6. Risk Factors and Protective Factors
    
    Provide data-driven insights and clinical recommendations.
    """
    
    # Aggregate session data
    combined_context = aggregate_session_contexts(session_ids)
    full_prompt = progress_prompt + "\n\nSession Data:\n" + combined_context
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    return {
        "client_id": str(client_id),
        "session_count": len(session_ids),
        "report": response.choices[0].message.content,
        "generated_at": datetime.now().isoformat(),
        "report_type": "progress"
    }
```

### Real-Time Streaming
```python
async def stream_real_time_analysis(session_id: UUID):
    """Stream real-time analysis during therapy session"""
    
    async def generate_insights():
        # Get live session data
        live_data = await get_live_session_data(session_id)
        
        # Stream analysis
        stream = await client.chat.completions.create(
            model="gpt-4o",
            stream=True,
            messages=[{
                "role": "user",
                "content": f"Analyze this ongoing therapy session: {live_data}"
            }]
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield f"data: {chunk.choices[0].delta.content}\n\n"
    
    return StreamingResponse(generate_insights(), media_type="text/event-stream")
```

## Report Templates

### Session Summary Template
```python
def build_session_summary_template(session_id: UUID) -> str:
    """Build template for session summary report"""
    
    session_data = get_session_data(session_id)
    keywords = get_session_keywords(session_id)
    insights = get_session_insights(session_id)
    
    template = f"""
    SESSION SUMMARY REPORT
    
    Session ID: {session_id}
    Date: {session_data['date']}
    Duration: {session_data['duration']} minutes
    
    CLIENT PRESENTATION:
    {format_client_presentation(session_data)}
    
    KEY THEMES:
    {format_key_themes(keywords)}
    
    THERAPEUTIC OBSERVATIONS:
    {format_therapeutic_observations(insights)}
    
    PROGRESS INDICATORS:
    {format_progress_indicators(session_data)}
    
    RECOMMENDATIONS:
    {format_recommendations(insights)}
    
    NEXT STEPS:
    {format_next_steps(session_data)}
    """
    
    return template
```

### Clinical Assessment Template
```python
def build_clinical_assessment_template(client_id: UUID) -> str:
    """Build template for clinical assessment report"""
    
    client_profile = get_client_profile(client_id)
    recent_sessions = get_recent_sessions(client_id, count=5)
    
    template = f"""
    CLINICAL ASSESSMENT REPORT
    
    Client ID: {client_id}
    Assessment Date: {datetime.now().strftime('%Y-%m-%d')}
    
    CLINICAL PRESENTATION:
    {format_clinical_presentation(client_profile)}
    
    DIAGNOSTIC IMPRESSIONS:
    {format_diagnostic_impressions(client_profile)}
    
    THERAPEUTIC PROGRESS:
    {format_therapeutic_progress(recent_sessions)}
    
    RISK ASSESSMENT:
    {format_risk_assessment(client_profile)}
    
    TREATMENT RECOMMENDATIONS:
    {format_treatment_recommendations(client_profile)}
    
    PROGNOSIS:
    {format_prognosis(client_profile)}
    """
    
    return template
```

## Integration Examples

### API Endpoints
```python
@app.get("/api/sessions/{session_id}/report")
def get_session_report(session_id: UUID, report_type: str = "summary"):
    """Generate session report"""
    
    try:
        if report_type == "summary":
            report = generate_session_summary(session_id)
        elif report_type == "comprehensive":
            report = generate_comprehensive_report(session_id)
        else:
            return {"error": "Invalid report type"}
        
        return {
            "success": True,
            "report": report,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/sessions/{session_id}/stream-report")
def stream_session_report(session_id: UUID):
    """Stream session report generation"""
    
    return stream(session_id)

@app.get("/api/clients/{client_id}/progress-report")
def get_progress_report(client_id: UUID, session_count: int = 10):
    """Generate client progress report"""
    
    try:
        session_ids = get_recent_sessions(client_id, session_count)
        report = generate_progress_report(client_id, session_ids)
        
        return {
            "success": True,
            "report": report,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}
```

### Dashboard Integration
```python
def generate_dashboard_summary(session_id: UUID) -> dict:
    """Generate summary for dashboard display"""
    
    # Get session analysis
    session_analysis = analyze_session(session_id)
    
    # Generate concise insights
    summary = {
        "session_id": str(session_id),
        "key_insights": extract_key_insights(session_analysis),
        "mood_assessment": assess_session_mood(session_analysis),
        "progress_indicators": extract_progress_indicators(session_analysis),
        "recommendations": extract_top_recommendations(session_analysis),
        "risk_factors": identify_risk_factors(session_analysis),
        "next_steps": determine_next_steps(session_analysis)
    }
    
    return summary
```

## Quality Assurance

### Report Validation
```python
def validate_report_quality(report: dict) -> dict:
    """Validate generated report quality"""
    
    validation_metrics = {
        "completeness": 0.0,
        "clinical_accuracy": 0.0,
        "professional_formatting": 0.0,
        "actionable_insights": 0.0
    }
    
    # Check completeness
    required_sections = [
        "session_overview", "key_themes", "therapeutic_observations",
        "progress_indicators", "recommendations", "next_steps"
    ]
    
    present_sections = sum(1 for section in required_sections 
                          if section in report["report"].lower())
    validation_metrics["completeness"] = present_sections / len(required_sections)
    
    # Validate clinical accuracy
    validation_metrics["clinical_accuracy"] = validate_clinical_content(report)
    
    # Check professional formatting
    validation_metrics["professional_formatting"] = assess_formatting_quality(report)
    
    # Assess actionable insights
    validation_metrics["actionable_insights"] = count_actionable_recommendations(report)
    
    return validation_metrics
```

This output module provides comprehensive reporting capabilities, enabling clinicians to generate professional therapeutic documentation with real-time insights and evidence-based recommendations.
