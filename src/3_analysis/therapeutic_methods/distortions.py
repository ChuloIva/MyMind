from openai import OpenAI
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CognitiveDistortionAnalyzer:
    def __init__(self, api_key: str = "", model: str = "gpt-4o"):
        """Initialize cognitive distortion analyzer"""
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        
        # Common cognitive distortions from CBT
        self.distortion_types = [
            "all_or_nothing_thinking",
            "overgeneralization", 
            "mental_filter",
            "disqualifying_positive",
            "jumping_to_conclusions",
            "magnification_catastrophizing",
            "minimization",
            "emotional_reasoning",
            "should_statements",
            "labeling",
            "personalization",
            "fortune_telling",
            "mind_reading"
        ]
        
        # Schema therapy modes and schemas
        self.schema_modes = [
            "vulnerable_child",
            "angry_child",
            "punitive_parent",
            "demanding_parent", 
            "detached_protector",
            "avoidant_protector",
            "overcompensatory_mode",
            "healthy_adult"
        ]
        
    def analyze_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze therapy session for cognitive distortions and schema patterns
        
        Args:
            session_data: Session data with transcription and processing
            
        Returns:
            Comprehensive therapeutic analysis
        """
        try:
            # Extract text segments
            segments = self._extract_segments(session_data)
            
            if not segments:
                logger.warning("No segments found for analysis")
                return self._empty_analysis()
            
            # Analyze for cognitive distortions
            distortion_analysis = self._analyze_cognitive_distortions(segments)
            
            # Analyze for schema patterns
            schema_analysis = self._analyze_schema_patterns(segments)
            
            # Generate therapeutic insights
            insights = self._generate_therapeutic_insights(
                segments, distortion_analysis, schema_analysis
            )
            
            # Calculate risk factors
            risk_assessment = self._assess_risk_factors(
                distortion_analysis, schema_analysis, segments
            )
            
            return {
                'cognitive_distortions': distortion_analysis,
                'schema_analysis': schema_analysis,
                'therapeutic_insights': insights,
                'risk_assessment': risk_assessment,
                'recommendations': self._generate_recommendations(
                    distortion_analysis, schema_analysis
                ),
                'analysis_metadata': {
                    'analyzed_at': datetime.utcnow().isoformat(),
                    'segments_analyzed': len(segments),
                    'model_used': self.model
                }
            }
            
        except Exception as e:
            logger.error(f"Session analysis failed: {e}")
            return self._empty_analysis()
    
    def _extract_segments(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relevant text segments from session data"""
        segments = []
        
        # Get processed segments first
        processed_segments = session_data.get('processed_segments', [])
        if processed_segments:
            segments = processed_segments
        else:
            # Fallback to combined segments
            combined_segments = session_data.get('combined_segments', [])
            if combined_segments:
                segments = combined_segments
            else:
                # Last resort - raw transcription
                transcription = session_data.get('transcription', [])
                segments = transcription
        
        # Filter client segments (non-therapist)
        client_segments = []
        for segment in segments:
            speaker = segment.get('speaker', '').upper()
            # Assume therapist speakers are numbered higher or contain 'THERAPIST'
            if 'THERAPIST' not in speaker and not (speaker.endswith('_00') or speaker.endswith('_01')):
                client_segments.append(segment)
        
        return client_segments if client_segments else segments
    
    def _analyze_cognitive_distortions(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze segments for cognitive distortions"""
        
        # Combine text for analysis
        combined_text = "\n".join([
            f"[{seg.get('start', 0):.1f}s] {seg.get('text', '')}"
            for seg in segments
        ])
        
        prompt = f"""
        Analyze the following therapy session transcript for cognitive distortions from CBT.
        
        Transcript:
        {combined_text}
        
        Identify instances of these cognitive distortions:
        1. All-or-nothing thinking (black and white thinking)
        2. Overgeneralization (broad conclusions from single events)
        3. Mental filter (focusing only on negatives)
        4. Disqualifying the positive (rejecting positive experiences)
        5. Jumping to conclusions (mind reading, fortune telling)
        6. Magnification/Catastrophizing (blowing things out of proportion)
        7. Minimization (inappropriately shrinking importance)
        8. Emotional reasoning (feelings as facts)
        9. Should statements (rigid rules and expectations)
        10. Labeling (global negative labels)
        11. Personalization (taking responsibility for everything)
        12. Fortune telling (predicting negative outcomes)
        13. Mind reading (assuming others' thoughts)
        
        Return JSON format:
        {{
            "distortions_found": [
                {{
                    "type": "all_or_nothing_thinking",
                    "evidence": "specific quote from transcript",
                    "explanation": "why this is a cognitive distortion",
                    "severity": 0.8,
                    "timestamp": 15.2,
                    "alternative_thought": "more balanced perspective"
                }}
            ],
            "distortion_summary": {{
                "total_distortions": 5,
                "most_common": "catastrophizing",
                "severity_average": 0.6,
                "patterns": ["tendency to catastrophize", "black and white thinking"]
            }},
            "therapeutic_focus_areas": [
                "thought challenging", "cognitive restructuring"
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Identified {len(result.get('distortions_found', []))} cognitive distortions")
            return result
            
        except Exception as e:
            logger.error(f"Cognitive distortion analysis failed: {e}")
            return {
                'distortions_found': [],
                'distortion_summary': {
                    'total_distortions': 0,
                    'most_common': None,
                    'severity_average': 0,
                    'patterns': []
                },
                'therapeutic_focus_areas': []
            }
    
    def _analyze_schema_patterns(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze segments for schema therapy patterns"""
        
        combined_text = "\n".join([
            f"[{seg.get('start', 0):.1f}s] {seg.get('text', '')}"
            for seg in segments
        ])
        
        prompt = f"""
        Analyze the following therapy session transcript for schema therapy patterns and modes.
        
        Transcript:
        {combined_text}
        
        Identify evidence of these schema modes:
        1. Vulnerable Child Mode (helpless, abandoned, abused feelings)
        2. Angry Child Mode (rage, frustration, tantrum behaviors)
        3. Punitive Parent Mode (harsh self-criticism, punishment)
        4. Demanding Parent Mode (entitlement, controlling behaviors)
        5. Detached Protector Mode (emotional disconnection, avoidance)
        6. Avoidant Protector Mode (withdrawal, isolation)
        7. Overcompensatory Mode (aggression, grandiosity to compensate)
        8. Healthy Adult Mode (balanced, rational, nurturing)
        
        Also identify early maladaptive schemas:
        - Abandonment/Instability
        - Mistrust/Abuse
        - Emotional Deprivation
        - Defectiveness/Shame
        - Social Isolation/Alienation
        - Dependence/Incompetence
        - Vulnerability to Harm
        - Enmeshment/Undeveloped Self
        - Failure
        - Entitlement/Grandiosity
        - Insufficient Self-Control
        - Subjugation
        - Self-Sacrifice
        - Approval-Seeking
        - Negativity/Pessimism
        - Emotional Inhibition
        - Unrelenting Standards
        - Punitiveness
        
        Return JSON format:
        {{
            "active_modes": [
                {{
                    "mode": "vulnerable_child",
                    "evidence": "specific quotes",
                    "intensity": 0.7,
                    "triggers": ["criticism", "rejection"],
                    "timestamp": 25.5
                }}
            ],
            "schemas_identified": [
                {{
                    "schema": "abandonment_instability", 
                    "evidence": "specific examples",
                    "strength": 0.8,
                    "manifestations": ["fear of being left alone"]
                }}
            ],
            "mode_summary": {{
                "dominant_mode": "vulnerable_child",
                "healthy_adult_present": false,
                "mode_switches": 3
            }},
            "schema_domains": {{
                "disconnection_rejection": 0.8,
                "impaired_autonomy": 0.3,
                "impaired_limits": 0.1,
                "other_directedness": 0.6,
                "overvigilance_inhibition": 0.4
            }}
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Identified {len(result.get('active_modes', []))} schema modes")
            return result
            
        except Exception as e:
            logger.error(f"Schema analysis failed: {e}")
            return {
                'active_modes': [],
                'schemas_identified': [],
                'mode_summary': {
                    'dominant_mode': None,
                    'healthy_adult_present': False,
                    'mode_switches': 0
                },
                'schema_domains': {}
            }
    
    def _generate_therapeutic_insights(
        self, 
        segments: List[Dict[str, Any]], 
        distortions: Dict[str, Any], 
        schemas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate therapeutic insights from analysis"""
        
        total_distortions = distortions.get('distortion_summary', {}).get('total_distortions', 0)
        active_modes = len(schemas.get('active_modes', []))
        
        return {
            'overall_assessment': {
                'cognitive_complexity': min(1.0, total_distortions / 10),
                'emotional_dysregulation': min(1.0, active_modes / 5),
                'therapeutic_readiness': self._assess_readiness(segments),
                'intervention_urgency': self._assess_urgency(distortions, schemas)
            },
            'key_themes': self._identify_key_themes(distortions, schemas),
            'progress_indicators': self._identify_progress_indicators(segments),
            'therapeutic_relationship': self._assess_relationship(segments)
        }
    
    def _assess_risk_factors(
        self, 
        distortions: Dict[str, Any], 
        schemas: Dict[str, Any], 
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess risk factors from analysis"""
        
        risk_factors = []
        risk_level = 0
        
        # Check for high-risk distortions
        high_risk_distortions = [
            'catastrophizing', 'all_or_nothing_thinking', 'fortune_telling'
        ]
        
        for distortion in distortions.get('distortions_found', []):
            if distortion.get('type') in high_risk_distortions and distortion.get('severity', 0) > 0.7:
                risk_factors.append(f"High severity {distortion['type']}")
                risk_level += 0.2
        
        # Check for vulnerable schemas
        vulnerable_schemas = [
            'abandonment_instability', 'mistrust_abuse', 'defectiveness_shame'
        ]
        
        for schema in schemas.get('schemas_identified', []):
            if schema.get('schema') in vulnerable_schemas and schema.get('strength', 0) > 0.7:
                risk_factors.append(f"Strong {schema['schema']} schema")
                risk_level += 0.15
        
        return {
            'risk_level': min(1.0, risk_level),
            'risk_factors': risk_factors,
            'protective_factors': self._identify_protective_factors(segments, schemas),
            'monitoring_recommendations': self._generate_monitoring_recommendations(risk_level)
        }
    
    def _generate_recommendations(
        self, 
        distortions: Dict[str, Any], 
        schemas: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate therapeutic recommendations"""
        
        recommendations = []
        
        # CBT recommendations based on distortions
        common_distortions = distortions.get('distortion_summary', {}).get('patterns', [])
        for pattern in common_distortions:
            if 'catastrophizing' in pattern.lower():
                recommendations.append({
                    'type': 'CBT_technique',
                    'intervention': 'Thought challenging for catastrophic thinking',
                    'priority': 'high',
                    'description': 'Use probability estimation and evidence examination'
                })
        
        # Schema therapy recommendations
        dominant_mode = schemas.get('mode_summary', {}).get('dominant_mode')
        if dominant_mode == 'vulnerable_child':
            recommendations.append({
                'type': 'schema_therapy',
                'intervention': 'Limited reparenting and nurturing interventions',
                'priority': 'high',
                'description': 'Address unmet childhood needs safely'
            })
        
        return recommendations
    
    def _assess_readiness(self, segments: List[Dict[str, Any]]) -> float:
        """Assess therapeutic readiness from engagement indicators"""
        # Simple heuristic based on segment engagement
        return min(1.0, len(segments) / 20)
    
    def _assess_urgency(self, distortions: Dict[str, Any], schemas: Dict[str, Any]) -> str:
        """Assess intervention urgency"""
        severity = distortions.get('distortion_summary', {}).get('severity_average', 0)
        if severity > 0.8:
            return 'high'
        elif severity > 0.5:
            return 'medium'
        return 'low'
    
    def _identify_key_themes(self, distortions: Dict[str, Any], schemas: Dict[str, Any]) -> List[str]:
        """Identify key therapeutic themes"""
        themes = []
        
        # Add distortion-based themes
        patterns = distortions.get('distortion_summary', {}).get('patterns', [])
        themes.extend(patterns)
        
        # Add schema-based themes
        dominant_mode = schemas.get('mode_summary', {}).get('dominant_mode')
        if dominant_mode:
            themes.append(f"{dominant_mode}_mode_work")
        
        return themes
    
    def _identify_progress_indicators(self, segments: List[Dict[str, Any]]) -> List[str]:
        """Identify positive progress indicators"""
        # Simple implementation - could be enhanced with NLP
        return ['active engagement', 'insight development']
    
    def _assess_relationship(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess therapeutic relationship quality"""
        return {
            'engagement_level': 'moderate',
            'resistance_indicators': [],
            'alliance_strength': 0.7
        }
    
    def _identify_protective_factors(
        self, 
        segments: List[Dict[str, Any]], 
        schemas: Dict[str, Any]
    ) -> List[str]:
        """Identify protective factors"""
        factors = []
        
        if schemas.get('mode_summary', {}).get('healthy_adult_present'):
            factors.append('Healthy adult mode present')
        
        return factors
    
    def _generate_monitoring_recommendations(self, risk_level: float) -> List[str]:
        """Generate monitoring recommendations based on risk level"""
        if risk_level > 0.7:
            return ['Weekly check-ins', 'Safety planning', 'Crisis contact information']
        elif risk_level > 0.4:
            return ['Bi-weekly monitoring', 'Progress tracking']
        return ['Standard follow-up', 'Monthly assessment']
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'cognitive_distortions': {
                'distortions_found': [],
                'distortion_summary': {
                    'total_distortions': 0,
                    'most_common': None,
                    'severity_average': 0,
                    'patterns': []
                },
                'therapeutic_focus_areas': []
            },
            'schema_analysis': {
                'active_modes': [],
                'schemas_identified': [],
                'mode_summary': {
                    'dominant_mode': None,
                    'healthy_adult_present': False,
                    'mode_switches': 0
                },
                'schema_domains': {}
            },
            'therapeutic_insights': {},
            'risk_assessment': {
                'risk_level': 0,
                'risk_factors': [],
                'protective_factors': [],
                'monitoring_recommendations': []
            },
            'recommendations': [],
            'analysis_metadata': {
                'analyzed_at': datetime.utcnow().isoformat(),
                'segments_analyzed': 0,
                'model_used': self.model
            }
        }

def analyse(transcript: str, api_key: str = "") -> Dict[str, Any]:
    """
    Analyze transcript for cognitive distortions and schema patterns
    
    Args:
        transcript: Therapy session transcript
        api_key: OpenAI API key
        
    Returns:
        Therapeutic analysis results
    """
    # Convert transcript to session data format
    session_data = {
        'transcription': [{'text': transcript, 'speaker': 'CLIENT', 'start': 0}]
    }
    
    analyzer = CognitiveDistortionAnalyzer(api_key=api_key)
    return analyzer.analyze_session(session_data)
