from openai import OpenAI
from typing import List, Dict, Any, Optional
import json
import logging
import re
from datetime import datetime
from ...common.config import settings
from pathlib import Path

logger = logging.getLogger(__name__)

class KeywordExtractor:
    def __init__(self, api_key: str = settings.openai_api_key, model: str = "gpt-4.1-nano-2025-04-14"):
        """Initialize keyword extractor with OpenAI client"""
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        with open(Path(__file__).parent / "keyword_extraction.prompt", "r") as f:
            self.prompt_template = f.read()
        
    def extract_keywords_and_sentiment(
        self, 
        text_segments: List[Dict[str, Any]], 
        chunk_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract keywords and sentiment from text segments
        
        Args:
            text_segments: List of transcription segments with timestamps
            chunk_size: Number of sentences to process together
            
        Returns:
            List of processed segments with keywords and sentiment
        """
        processed_segments = []
        
        # Group segments into chunks
        for i in range(0, len(text_segments), chunk_size):
            chunk = text_segments[i:i + chunk_size]
            
            try:
                # Process chunk
                chunk_result = self._process_chunk(chunk)
                processed_segments.extend(chunk_result)
                
            except Exception as e:
                logger.error(f"Failed to process chunk {i}-{i+chunk_size}: {e}")
                # Add segments without processing if failed
                for segment in chunk:
                    segment['keywords'] = []
                    segment['sentiment_scores'] = {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1}
                    processed_segments.append(segment)
        
        return processed_segments
    
    def _process_chunk(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of segments for keywords and sentiment"""
        
        # Prepare text for analysis
        chunk_text = "\n".join([
            f"Segment {i+1} ({seg.get('start', 0):.2f}s-{seg.get('end', 0):.2f}s): {seg.get('text', '')}"
            for i, seg in enumerate(segments)
        ])
        
        prompt = self._build_prompt(chunk_text, len(segments))
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Apply results to segments
            return self._apply_results_to_segments(segments, result)
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _build_prompt(self, text: str, num_segments: int) -> str:
        """Build prompt for keyword extraction and sentiment analysis"""
        
        return self.prompt_template.format(text=text, num_segments=num_segments)
    
    def _apply_results_to_segments(
        self, 
        segments: List[Dict[str, Any]], 
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply extraction results back to original segments"""
        
        processed = []
        result_segments = results.get('segments', [])
        
        for i, segment in enumerate(segments):
            processed_segment = segment.copy()
            
            # Find corresponding result
            if i < len(result_segments):
                result = result_segments[i]
                
                processed_segment['keywords'] = result.get('keywords', [])
                processed_segment['sentiment_scores'] = result.get('sentiment_scores', {
                    'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1
                })
                processed_segment['emotional_indicators'] = result.get('emotional_indicators', [])
                processed_segment['therapeutic_themes'] = result.get('therapeutic_themes', [])
            else:
                # Default values if no result
                processed_segment['keywords'] = []
                processed_segment['sentiment_scores'] = {
                    'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1
                }
                processed_segment['emotional_indicators'] = []
                processed_segment['therapeutic_themes'] = []
            
            processed_segment['processed_at'] = datetime.utcnow().isoformat()
            processed.append(processed_segment)
        
        return processed

def extract_session_keywords(
    transcription_data: Dict[str, Any], 
    chunk_size: int = 3
) -> Dict[str, Any]:
    """
    Extract keywords and sentiment from session transcription
    
    Args:
        transcription_data: Session transcription with segments
        chunk_size: Number of sentences to process together
        
    Returns:
        Enhanced transcription data with keywords and sentiment
    """
    extractor = KeywordExtractor()
    
    # Get segments from transcription data
    segments = transcription_data.get('combined_segments', [])
    if not segments:
        segments = transcription_data.get('transcription', [])
    
    # Process segments
    processed_segments = extractor.extract_keywords_and_sentiment(segments, chunk_size)
    
    # Update transcription data
    enhanced_data = transcription_data.copy()
    enhanced_data['processed_segments'] = processed_segments
    enhanced_data['processing_metadata'] = {
        'processed_at': datetime.utcnow().isoformat(),
        'total_segments': len(processed_segments),
        'chunk_size': chunk_size,
        'model_used': extractor.model
    }
    
    return enhanced_data

def extract(text: str) -> Dict[str, Any]:
    """
    Simple keyword extraction function (for compatibility)
    
    Args:
        text: Text to analyze
        
    Returns:
        Extracted keywords and sentiment
    """
    # Convert text to segment format
    segments = [{'text': text, 'start': 0, 'end': 0}]
    
    extractor = KeywordExtractor()
    results = extractor.extract_keywords_and_sentiment(segments, chunk_size=1)
    
    return results[0] if results else {}