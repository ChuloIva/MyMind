# 3.2. Therapeutic Methods Analysis

This module analyzes therapy session transcripts using evidence-based therapeutic approaches, focusing on cognitive distortion detection, schema identification, and therapeutic intervention recommendations.

## Core Components

### Cognitive Behavioral Therapy (CBT) Analysis
- **Distortion Detection**: Identifies 15+ common cognitive distortions
- **Pattern Recognition**: Automated identification of maladaptive thinking patterns
- **Evidence Extraction**: Specific text examples supporting each identified distortion
- **Intervention Suggestions**: CBT-based therapeutic recommendations

### Schema Therapy Integration
- **Maladaptive Schemas**: Detects underlying dysfunctional patterns
- **Schema Modes**: Identifies active emotional and behavioral states
- **Therapeutic Targets**: Prioritizes schema-focused interventions

### Therapeutic Assessment
- **Bias Detection**: Recognizes cognitive biases affecting judgment
- **Coping Analysis**: Evaluates adaptive vs. maladaptive coping strategies
- **Progress Tracking**: Monitors therapeutic progress across sessions

## Implementation

### `distortions.py`

Current implementation uses GPT-4o for sophisticated pattern recognition:

```python
from openai import OpenAI, AsyncOpenAI
client = OpenAI()
TEMPLATE = "Identify cognitive distortions… Return JSON: {distortions:[…]}"

def analyse(transcript: str):
    r = client.chat.completions.create(
        model="gpt-4o-large", temperature=0,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":TEMPLATE+transcript}]
    )
    return json.loads(r.choices[0].message.content)
```

## Advanced Usage Examples

### Comprehensive CBT Analysis
```python
def comprehensive_cbt_analysis(transcript: str) -> dict:
    """Comprehensive CBT analysis of therapy session"""
    
    cbt_prompt = f"""
    Analyze this therapy session transcript for CBT-relevant patterns:
    
    {transcript}
    
    Return JSON with the following structure:
    {{
      "cognitive_distortions": [
        {{
          "type": "catastrophizing",
          "description": "Imagining worst-case scenarios",
          "evidence": "specific quote from transcript",
          "confidence": 0.85,
          "severity": "moderate",
          "timestamp": "approximate position in session"
        }}
      ],
      "thinking_patterns": [
        {{
          "pattern": "black_and_white_thinking",
          "examples": ["specific examples"],
          "frequency": "how often it occurs",
          "impact": "effect on client's wellbeing"
        }}
      ],
      "cognitive_strengths": [
        {{
          "strength": "reality_testing",
          "evidence": "examples of accurate thinking"
        }}
      ],
      "therapeutic_recommendations": [
        {{
          "intervention": "cognitive_restructuring",
          "rationale": "why this intervention is recommended",
          "priority": "high",
          "techniques": ["specific CBT techniques"]
        }}
      ]
    }}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": cbt_prompt}]
    )
    
    return json.loads(response.choices[0].message.content)
```

### Schema Therapy Analysis
```python
def schema_therapy_analysis(transcript: str) -> dict:
    """Schema therapy assessment of underlying patterns"""
    
    schema_prompt = f"""
    Analyze this therapy session for schema therapy patterns:
    
    {transcript}
    
    Identify:
    1. Early Maladaptive Schemas (18 core schemas)
    2. Schema Modes (Child, Parent, Coping modes)
    3. Schema Triggers and Activation patterns
    4. Therapeutic interventions needed
    
    Return JSON format with schemas, modes, triggers, and interventions.
    
    Text: {transcript}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": schema_prompt}]
    )
    
    return json.loads(response.choices[0].message.content)
```

## Reference Data

### Cognitive Distortions (`cognitive_biases.csv`)
The system recognizes 15+ common cognitive distortions:

| Distortion | Description | CBT Intervention |
|------------|-------------|------------------|
| All-or-Nothing | Black/white thinking | Identify gray areas |
| Catastrophizing | Worst-case scenarios | Probability estimation |
| Mind Reading | Assuming others' thoughts | Evidence gathering |
| Fortune Telling | Predicting negative futures | Examine evidence |
| Emotional Reasoning | Feelings as facts | Separate emotions from facts |
| Should Statements | Rigid expectations | Flexible thinking |
| Personalization | Taking excessive responsibility | Responsibility pie |
| Mental Filter | Focus on negatives | Balanced thinking |
| Discounting Positives | Minimizing achievements | Positive data log |
| Labeling | Global self-judgments | Specific behavior focus |

### Schema Categories (`schemas.csv`)
18 Early Maladaptive Schemas across 5 domains:

**Disconnection & Rejection:**
- Abandonment/Instability
- Mistrust/Abuse
- Emotional Deprivation
- Defectiveness/Shame
- Social Isolation

**Impaired Autonomy:**
- Dependence/Incompetence
- Vulnerability to Harm
- Enmeshment/Undeveloped Self
- Failure

**Impaired Limits:**
- Entitlement/Grandiosity
- Insufficient Self-Control

**Other-Directedness:**
- Subjugation
- Self-Sacrifice
- Approval-Seeking

**Overvigilance & Inhibition:**
- Negativity/Pessimism
- Emotional Inhibition
- Unrelenting Standards
- Punitiveness

## Advanced Features

### Multi-Session Pattern Analysis
```python
def analyze_therapeutic_progress(session_ids: list[str]) -> dict:
    """Track therapeutic patterns across multiple sessions"""
    
    progress_data = {
        "session_count": len(session_ids),
        "distortion_trends": {},
        "schema_evolution": {},
        "therapeutic_gains": [],
        "persistent_patterns": []
    }
    
    for session_id in session_ids:
        transcript = get_session_transcript(session_id)
        
        # Analyze CBT patterns
        cbt_analysis = comprehensive_cbt_analysis(transcript)
        
        # Track distortion frequency
        for distortion in cbt_analysis["cognitive_distortions"]:
            dist_type = distortion["type"]
            if dist_type not in progress_data["distortion_trends"]:
                progress_data["distortion_trends"][dist_type] = []
            progress_data["distortion_trends"][dist_type].append({
                "session_id": session_id,
                "severity": distortion["severity"],
                "confidence": distortion["confidence"]
            })
        
        # Analyze schema patterns
        schema_analysis = schema_therapy_analysis(transcript)
        progress_data["schema_evolution"][session_id] = schema_analysis
    
    # Calculate progress indicators
    progress_data["therapeutic_gains"] = calculate_therapeutic_gains(progress_data)
    progress_data["persistent_patterns"] = identify_persistent_patterns(progress_data)
    
    return progress_data
```

### Intervention Prioritization
```python
def prioritize_interventions(analysis_results: dict) -> list[dict]:
    """Prioritize therapeutic interventions based on analysis"""
    
    interventions = []
    
    # CBT interventions
    for distortion in analysis_results.get("cognitive_distortions", []):
        intervention = {
            "type": "CBT",
            "target": distortion["type"],
            "priority": calculate_priority(distortion),
            "techniques": get_cbt_techniques(distortion["type"]),
            "rationale": f"Address {distortion['type']} pattern",
            "evidence": distortion["evidence"]
        }
        interventions.append(intervention)
    
    # Schema interventions
    for schema in analysis_results.get("schemas", []):
        intervention = {
            "type": "Schema_Therapy",
            "target": schema["name"],
            "priority": calculate_schema_priority(schema),
            "techniques": get_schema_techniques(schema["name"]),
            "rationale": f"Address {schema['name']} schema",
            "mode_work": schema.get("mode_interventions", [])
        }
        interventions.append(intervention)
    
    # Sort by priority
    interventions.sort(key=lambda x: x["priority"], reverse=True)
    
    return interventions
```

## Quality Assurance

### Clinical Validation
```python
def validate_therapeutic_analysis(analysis: dict) -> dict:
    """Validate therapeutic analysis against clinical standards"""
    
    validation_metrics = {
        "clinical_accuracy": 0.0,
        "diagnostic_consistency": 0.0,
        "intervention_appropriateness": 0.0,
        "evidence_quality": 0.0
    }
    
    # Validate distortion identification
    if "cognitive_distortions" in analysis:
        clinical_accuracy = validate_distortion_accuracy(analysis["cognitive_distortions"])
        validation_metrics["clinical_accuracy"] = clinical_accuracy
    
    # Validate schema identification
    if "schemas" in analysis:
        diagnostic_consistency = validate_schema_consistency(analysis["schemas"])
        validation_metrics["diagnostic_consistency"] = diagnostic_consistency
    
    # Validate intervention recommendations
    if "therapeutic_recommendations" in analysis:
        intervention_quality = validate_intervention_appropriateness(analysis["therapeutic_recommendations"])
        validation_metrics["intervention_appropriateness"] = intervention_quality
    
    # Validate evidence quality
    evidence_quality = assess_evidence_quality(analysis)
    validation_metrics["evidence_quality"] = evidence_quality
    
    return validation_metrics
```

### Confidence Scoring
```python
def calculate_confidence_scores(analysis: dict) -> dict:
    """Calculate confidence scores for therapeutic analysis"""
    
    confidence_metrics = {
        "overall_confidence": 0.0,
        "distortion_confidence": 0.0,
        "schema_confidence": 0.0,
        "intervention_confidence": 0.0
    }
    
    # Calculate distortion confidence
    if "cognitive_distortions" in analysis:
        distortion_confidences = [d["confidence"] for d in analysis["cognitive_distortions"]]
        confidence_metrics["distortion_confidence"] = np.mean(distortion_confidences)
    
    # Calculate schema confidence
    if "schemas" in analysis:
        schema_confidences = [s.get("confidence", 0.5) for s in analysis["schemas"]]
        confidence_metrics["schema_confidence"] = np.mean(schema_confidences)
    
    # Calculate intervention confidence
    if "therapeutic_recommendations" in analysis:
        intervention_confidences = [i.get("confidence", 0.5) for i in analysis["therapeutic_recommendations"]]
        confidence_metrics["intervention_confidence"] = np.mean(intervention_confidences)
    
    # Overall confidence
    confidence_metrics["overall_confidence"] = np.mean([
        confidence_metrics["distortion_confidence"],
        confidence_metrics["schema_confidence"],
        confidence_metrics["intervention_confidence"]
    ])
    
    return confidence_metrics
```

## Integration Examples

### API Integration
```python
@app.post("/api/sessions/{session_id}/therapeutic-analysis")
def analyze_therapeutic_patterns(session_id: str):
    """Comprehensive therapeutic analysis endpoint"""
    
    try:
        # Get session transcript
        transcript = get_session_transcript(session_id)
        
        # Perform comprehensive analysis
        cbt_analysis = comprehensive_cbt_analysis(transcript)
        schema_analysis = schema_therapy_analysis(transcript)
        
        # Combine results
        comprehensive_analysis = {
            "session_id": session_id,
            "cbt_analysis": cbt_analysis,
            "schema_analysis": schema_analysis,
            "prioritized_interventions": prioritize_interventions({
                **cbt_analysis,
                **schema_analysis
            })
        }
        
        # Validate results
        validation = validate_therapeutic_analysis(comprehensive_analysis)
        confidence = calculate_confidence_scores(comprehensive_analysis)
        
        return {
            "analysis": comprehensive_analysis,
            "validation_metrics": validation,
            "confidence_scores": confidence,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": str(e)}
```

### Dashboard Integration
```python
def generate_therapeutic_dashboard(session_id: str) -> dict:
    """Generate therapeutic insights for dashboard display"""
    
    transcript = get_session_transcript(session_id)
    analysis = analyse(transcript)
    
    dashboard_data = {
        "session_id": session_id,
        "distortion_summary": {
            "count": len(analysis.get("distortions", [])),
            "most_common": get_most_common_distortion(analysis),
            "severity_distribution": calculate_severity_distribution(analysis)
        },
        "therapeutic_priorities": prioritize_interventions(analysis)[:3],
        "progress_indicators": calculate_session_progress(session_id),
        "recommended_actions": generate_session_recommendations(analysis)
    }
    
    return dashboard_data
```

This therapeutic methods module provides evidence-based analysis capabilities, enabling clinicians to identify cognitive patterns, understand underlying schemas, and develop targeted therapeutic interventions based on established therapeutic frameworks.
