from pyannote.audio import Pipeline
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import logging
import os
from ...common.config import settings

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SpeakerDiarizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = settings.pyannote_model, hf_token: str = settings.hf_token):
        """Initialize speaker diarization pipeline"""
        if self._initialized:
            return
        self.model_name = model_name
        self.hf_token = hf_token
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialize_pipeline()
        self._initialized = True
        
    def initialize_pipeline(self):
        """Initialize the pyannote pipeline"""
        if not self.hf_token:
            raise ValueError("HuggingFace token is required for pyannote models")
        
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            )
            self.pipeline = self.pipeline.to(torch.device(self.device))
            logger.info(f"Speaker diarization pipeline initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize speaker diarization pipeline: {e}")
            raise
    
    def diarize(self, audio_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional number of speakers (if known)
            
        Returns:
            List of diarization segments with speaker labels
        """
        if self.pipeline is None:
            self.initialize_pipeline()
        
        try:
            # Run diarization
            if num_speakers:
                diarization = self.pipeline(str(audio_path), num_speakers=num_speakers)
            else:
                diarization = self.pipeline(str(audio_path))
            
            # Convert to list of dictionaries
            segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'duration': segment.duration,
                    'speaker': speaker
                })
            
            logger.info(f"Diarization completed: {len(segments)} segments, {len(diarization.labels())} speakers")
            return segments
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            raise
    
    def get_speaker_statistics(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get speaking time statistics for each speaker"""
        speaker_times = {}
        
        for segment in segments:
            speaker = segment['speaker']
            duration = segment['duration']
            
            if speaker not in speaker_times:
                speaker_times[speaker] = 0
            speaker_times[speaker] += duration
        
        return {
            'speaker_times': speaker_times,
            'total_speakers': len(speaker_times),
            'dominant_speaker': max(speaker_times.items(), key=lambda x: x[1]) if speaker_times else None
        }

def diarise(wav_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Convenience function for speaker diarization
    
    Args:
        wav_path: Path to audio file
        num_speakers: Optional number of speakers
        
    Returns:
        List of diarization segments
    """
    diarizer = SpeakerDiarizer()
    return diarizer.diarize(wav_path, num_speakers=num_speakers)
