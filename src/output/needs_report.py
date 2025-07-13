# src/5_output/needs_report.py

from uuid import UUID
from typing import Dict, Any
from src.database.models import ClientNeedSummary
from src.database.database import get_session
from sqlmodel import Session, select

def get_client_needs_profile(client_id: UUID) -> ClientNeedSummary:
    """Get client needs profile from database"""
    db = next(get_session())
    statement = select(ClientNeedSummary).where(ClientNeedSummary.client_id == client_id)
    profile = db.exec(statement).first()
    
    if not profile:
        # Return empty profile if not found
        return ClientNeedSummary(
            client_id=client_id,
            life_segment_scores={},
            need_fulfillment_scores={},
            unmet_needs=[],
            fulfilled_needs=[]
        )
    
    return profile

class NeedsProfileReport:
    def generate_needs_assessment_report(self, client_id: UUID) -> str:
        """Generate comprehensive needs assessment report"""

        profile = get_client_needs_profile(client_id)

        report = f"""
        # Needs Assessment Report

        ## Life Segment Analysis

        ### Most Positive Areas:
        {self._format_positive_segments(profile.life_segment_scores)}

        ### Areas of Concern:
        {self._format_concerning_segments(profile.life_segment_scores)}

        ## Needs Fulfillment Analysis

        ### Well-Met Needs:
        {self._format_fulfilled_needs(profile.fulfilled_needs)}

        ### Unmet Needs Requiring Attention:
        {self._format_unmet_needs(profile.unmet_needs)}

        ## Therapeutic Recommendations:
        {self._generate_recommendations(profile)}
        """

        return report

    def generate_visual_profile(self, client_id: UUID) -> Dict[str, Any]:
        """Generate data for visual representation"""

        profile = get_client_needs_profile(client_id)

        return {
            'life_segment_radar': self._create_radar_data(profile.life_segment_scores),
            'needs_fulfillment_bar': self._create_bar_chart_data(profile.need_fulfillment_scores),
            'sentiment_heatmap': self._create_heatmap_data(profile),
            'progress_timeline': self._create_timeline_data(client_id)
        }

    def _format_positive_segments(self, life_segment_scores: Dict[str, Dict[str, float]]) -> str:
        """Format positive life segments for report"""
        positive_segments = []
        for segment, scores in life_segment_scores.items():
            if scores.get('sentiment', 0) > 0.3 and scores.get('fulfillment', 0) > 0.6:
                positive_segments.append(f"- **{segment.title()}**: High satisfaction ({scores['sentiment']:.2f}) and fulfillment ({scores['fulfillment']:.2f})")
        
        return "\n".join(positive_segments) if positive_segments else "No highly positive areas identified."

    def _format_concerning_segments(self, life_segment_scores: Dict[str, Dict[str, float]]) -> str:
        """Format concerning life segments for report"""
        concerning_segments = []
        for segment, scores in life_segment_scores.items():
            if scores.get('sentiment', 0) < -0.2 or scores.get('fulfillment', 0) < 0.4:
                concerning_segments.append(f"- **{segment.title()}**: Low satisfaction ({scores['sentiment']:.2f}) or fulfillment ({scores['fulfillment']:.2f})")
        
        return "\n".join(concerning_segments) if concerning_segments else "No concerning areas identified."

    def _format_fulfilled_needs(self, fulfilled_needs: list) -> str:
        """Format fulfilled needs for report"""
        if not fulfilled_needs:
            return "No highly fulfilled needs identified."
        
        formatted_needs = []
        for need_data in fulfilled_needs:
            need = need_data.get('need', 'Unknown')
            score = need_data.get('score', 0)
            formatted_needs.append(f"- **{need.title()}**: {score:.2f}")
        
        return "\n".join(formatted_needs)

    def _format_unmet_needs(self, unmet_needs: list) -> str:
        """Format unmet needs for report"""
        if not unmet_needs:
            return "No unmet needs identified."
        
        formatted_needs = []
        for need_data in unmet_needs:
            need = need_data.get('need', 'Unknown')
            score = need_data.get('score', 0)
            formatted_needs.append(f"- **{need.title()}**: {score:.2f}")
        
        return "\n".join(formatted_needs)

    def _generate_recommendations(self, profile: ClientNeedSummary) -> str:
        """Generate therapeutic recommendations based on profile"""
        recommendations = []
        
        # Recommendations based on unmet needs
        for need_data in profile.unmet_needs:
            need = need_data.get('need', 'Unknown')
            recommendations.append(f"- Focus on addressing **{need.title()}** needs through targeted interventions")
        
        # Recommendations based on concerning life segments
        for segment, scores in profile.life_segment_scores.items():
            if scores.get('sentiment', 0) < -0.2:
                recommendations.append(f"- Explore challenges in **{segment.title()}** area and develop coping strategies")
        
        return "\n".join(recommendations) if recommendations else "Continue current therapeutic approach."

    def _create_radar_data(self, life_segment_scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Create radar chart data for life segments"""
        labels = list(life_segment_scores.keys())
        sentiment_data = [scores.get('sentiment', 0) for scores in life_segment_scores.values()]
        fulfillment_data = [scores.get('fulfillment', 0) for scores in life_segment_scores.values()]
        
        return {
            'labels': labels,
            'datasets': [
                {
                    'label': 'Sentiment',
                    'data': sentiment_data,
                    'borderColor': 'rgb(75, 192, 192)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)'
                },
                {
                    'label': 'Fulfillment',
                    'data': fulfillment_data,
                    'borderColor': 'rgb(255, 99, 132)',
                    'backgroundColor': 'rgba(255, 99, 132, 0.2)'
                }
            ]
        }

    def _create_bar_chart_data(self, need_fulfillment_scores: Dict[str, float]) -> Dict[str, Any]:
        """Create bar chart data for needs fulfillment"""
        needs = list(need_fulfillment_scores.keys())
        scores = list(need_fulfillment_scores.values())
        
        return {
            'labels': needs,
            'datasets': [{
                'label': 'Need Fulfillment',
                'data': scores,
                'backgroundColor': ['rgba(75, 192, 192, 0.8)' if score > 0.6 else 'rgba(255, 99, 132, 0.8)' for score in scores]
            }]
        }

    def _create_heatmap_data(self, profile: ClientNeedSummary) -> Dict[str, Any]:
        """Create heatmap data for sentiment analysis"""
        # This would create a more complex visualization showing sentiment over time
        # For now, return a simplified structure
        return {
            'data': [],
            'labels': [],
            'title': 'Sentiment Heatmap'
        }

    def _create_timeline_data(self, client_id: UUID) -> Dict[str, Any]:
        """Create timeline data for progress tracking"""
        # This would fetch historical data and create timeline visualization
        # For now, return a simplified structure
        return {
            'data': [],
            'labels': [],
            'title': 'Progress Timeline'
        }
