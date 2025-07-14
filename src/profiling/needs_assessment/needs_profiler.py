# src/4_profiling/needs_assessment/needs_profiler.py

from typing import List, Dict, Any
from uuid import UUID
import numpy as np
import json
from datetime import datetime
from sqlmodel import Session, select
from src.preprocessing.llm_processing.needs_extraction import NeedsExtractor
from src.database.models import ClientNeedSummary, ClientNeedProfile, SessionSentence, NeedCategory, LifeSegment
from src.database.database import get_session

class NeedsProfiler:
    def __init__(self, db_session: Session):
        self.db = db_session
        # We don't initialize extractor here since needs extraction is done via API endpoint

    def build_client_profile(self, client_id: UUID, session_ids: List[UUID]) -> ClientNeedSummary:
        """Build or update comprehensive needs profile for client."""
        # 1. Fetch all existing extractions for the client
        statement = select(ClientNeedProfile).where(ClientNeedProfile.client_id == client_id)
        all_extractions = self.db.exec(statement).all()

        # 2. Aggregate data into a summary object
        summary = self._aggregate_profile(client_id, all_extractions)

        # 3. Save or update the summary in the database
        existing_summary = self.db.exec(
            select(ClientNeedSummary).where(ClientNeedSummary.client_id == client_id)
        ).first()

        if existing_summary:
            existing_summary.summary_data = summary.summary_data
            existing_summary.last_updated = datetime.utcnow()
            self.db.add(existing_summary)
            summary_to_return = existing_summary
        else:
            self.db.add(summary)
            summary_to_return = summary

        self.db.commit()
        self.db.refresh(summary_to_return)
        return summary_to_return

    def _aggregate_profile(self, client_id: UUID, extractions: List[ClientNeedProfile]) -> ClientNeedSummary:
        """Aggregate extractions into summary profile"""

        if not extractions:
            # Return empty summary
            return ClientNeedSummary(
                client_id=client_id,
                summary_data=json.dumps({
                    "life_segment_scores": {},
                    "need_fulfillment_scores": {},
                    "unmet_needs": [],
                    "fulfilled_needs": []
                })
            )

        life_segment_data = {}
        need_fulfillment_data = {}

        # Process each extraction
        for extraction in extractions:
            # Get the need and life segment names via relationships
            need_category = self.db.get(NeedCategory, extraction.need_category_id)
            life_segment = self.db.get(LifeSegment, extraction.life_segment_id)
            
            if not need_category or not life_segment:
                continue
                
            life_seg = life_segment.life_area
            need = need_category.need

            # Aggregate by life segment
            if life_seg not in life_segment_data:
                life_segment_data[life_seg] = {
                    'sentiments': [],
                    'fulfillments': [],
                    'count': 0
                }

            life_segment_data[life_seg]['sentiments'].append(extraction.sentiment_score)
            life_segment_data[life_seg]['fulfillments'].append(extraction.need_fulfillment_score)
            life_segment_data[life_seg]['count'] += 1

            # Aggregate by need
            if need not in need_fulfillment_data:
                need_fulfillment_data[need] = []

            need_fulfillment_data[need].append(extraction.need_fulfillment_score)

        # Calculate aggregated scores
        life_segment_scores = {}
        for seg, data in life_segment_data.items():
            life_segment_scores[seg] = {
                'sentiment': float(np.mean(data['sentiments'])),
                'fulfillment': float(np.mean(data['fulfillments'])),
                'frequency': data['count'] / len(extractions)  # Relative frequency
            }

        need_fulfillment_scores = {
            need: float(np.mean(scores))
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

        # Create summary data as JSON string
        summary_data = {
            "life_segment_scores": life_segment_scores,
            "need_fulfillment_scores": need_fulfillment_scores,
            "unmet_needs": unmet_needs,
            "fulfilled_needs": fulfilled_needs
        }

        # Create summary
        summary = ClientNeedSummary(
            client_id=client_id,
            summary_data=json.dumps(summary_data)
        )

        return summary

