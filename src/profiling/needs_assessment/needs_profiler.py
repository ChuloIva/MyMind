# src/4_profiling/needs_assessment/needs_profiler.py

from typing import List, Dict, Any
from uuid import UUID
import numpy as np
from src.preprocessing.llm_processing.needs_extraction import NeedsExtractor
from src.database.models import ClientNeedSummary

class NeedsProfiler:
    def __init__(self):
        self.extractor = NeedsExtractor()

    def build_client_profile(self, client_id: UUID, session_ids: List[UUID]) -> ClientNeedSummary:
        """Build comprehensive needs profile for client"""

        all_extractions = []

        # Process each session
        for session_id in session_ids:
            segments = self._get_session_segments(session_id)
            extractions = self.extractor.extract_needs_and_segments(segments)

            # Save to database
            self._save_extractions(client_id, session_id, extractions)
            all_extractions.extend(extractions)

        # Aggregate data
        profile = self._aggregate_profile(client_id, all_extractions)

        return profile

    def _aggregate_profile(self, client_id: UUID, extractions: List[Dict]) -> ClientNeedSummary:
        """Aggregate extractions into summary profile"""

        life_segment_data = {}
        need_fulfillment_data = {}

        # Process each extraction
        for extraction in extractions:
            life_seg = extraction['life_segment']
            need = extraction['need']

            # Aggregate by life segment
            if life_seg not in life_segment_data:
                life_segment_data[life_seg] = {
                    'sentiments': [],
                    'fulfillments': [],
                    'count': 0
                }

            life_segment_data[life_seg]['sentiments'].append(extraction['sentiment_score'])
            life_segment_data[life_seg]['fulfillments'].append(extraction['need_fulfillment_score'])
            life_segment_data[life_seg]['count'] += 1

            # Aggregate by need
            if need not in need_fulfillment_data:
                need_fulfillment_data[need] = []

            need_fulfillment_data[need].append(extraction['need_fulfillment_score'])

        # Calculate aggregated scores
        life_segment_scores = {}
        for seg, data in life_segment_data.items():
            life_segment_scores[seg] = {
                'sentiment': np.mean(data['sentiments']),
                'fulfillment': np.mean(data['fulfillments']),
                'frequency': data['count'] / len(extractions)  # Relative frequency
            }

        need_fulfillment_scores = {
            need: np.mean(scores)
            for need, scores in need_fulfillment_data.items()
        }

        # Identify top unmet and fulfilled needs
        sorted_needs = sorted(need_fulfillment_scores.items(), key=lambda x: x[1])
        unmet_needs = [
            {'need': need, 'score': score}
            for need, score in sorted_needs[:5] if score < 0.4
        ]
        fulfilled_needs = [
            {'need': need, 'score': score}
            for need, score in sorted_needs[-5:] if score > 0.7
        ]

        # Create summary
        summary = ClientNeedSummary(
            client_id=client_id,
            life_segment_scores=life_segment_scores,
            need_fulfillment_scores=need_fulfillment_scores,
            unmet_needs=unmet_needs,
            fulfilled_needs=fulfilled_needs
        )

        return summary

    def _get_session_segments(self, session_id: UUID) -> List[Dict[str, Any]]:
        """Get transcript segments for a session"""
        # TODO: Implement database query to get session segments
        # This should query the SessionSentence table for the given session_id
        # For now, return empty list as placeholder
        return []

    def _save_extractions(self, client_id: UUID, session_id: UUID, extractions: List[Dict[str, Any]]) -> None:
        """Save extractions to database"""
        # TODO: Implement database save for extractions
        # This should save to ClientNeedProfile table
        # For now, just a placeholder
        pass
