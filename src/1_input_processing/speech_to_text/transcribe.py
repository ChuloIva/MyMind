from faster_whisper import WhisperModel
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json
from .speaker_diarisation import diarise
from ...common.config import settings

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(WhisperTranscriber, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_size: str = settings.whisper_model, device: str = settings.whisper_device, compute_type: str = "float16"):
        """Initialize Whisper transcription model"""
        if self._initialized:
            return
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self.initialize_model()
        self._initialized = True
        
    def initialize_model(self):
        """Initialize the Whisper model"""
        try:
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
            )
            logger.info(f"Whisper model {self.model_size} initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise
    
    def transcribe(self, audio_path: Path, beam_size: int = 5, word_timestamps: bool = True) -> List[Dict[str, Any]]:
        """
        Transcribe audio file with word-level timestamps
        
        Args:
            audio_path: Path to audio file
            beam_size: Beam size for transcription
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            List of transcription segments with timestamps
        """
        if self.model is None:
            self.initialize_model()
        
        try:
            segments, info = self.model.transcribe(
                str(audio_path), 
                beam_size=beam_size, 
                word_timestamps=word_timestamps
            )
            
            # Convert segments to list of dictionaries
            result = []
            for segment in segments:
                segment_dict = {
                    'id': segment.id,
                    'seek': segment.seek,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'tokens': segment.tokens,
                    'temperature': segment.temperature,
                    'avg_logprob': segment.avg_logprob,
                    'compression_ratio': segment.compression_ratio,
                    'no_speech_prob': segment.no_speech_prob
                }
                
                # Add word-level timestamps if available
                if word_timestamps and hasattr(segment, 'words') and segment.words:
                    segment_dict['words'] = [
                        {
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        }
                        for word in segment.words
                    ]
                
                result.append(segment_dict)
            
            logger.info(f"Transcription completed: {len(result)} segments")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

def transcribe_with_speakers(
    audio_path: Path, 
    num_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Transcribe audio with speaker diarization
    
    Args:
        audio_path: Path to audio file
        num_speakers: Optional number of speakers
        
    Returns:
        Dictionary containing transcription and speaker information
    """
    logger.info(f"Starting transcription with speaker diarization for {audio_path}")
    
    # Initialize transcriber
    transcriber = WhisperTranscriber()
    
    # Get transcription
    transcription = transcriber.transcribe(audio_path)
    
    # Get speaker diarization
    diarization = []
    try:
        diarization = diarise(audio_path, num_speakers)
    except Exception as e:
        logger.warning(f"Speaker diarization failed: {e}")
    
    # Combine transcription with speaker information
    combined_segments = align_transcription_with_speakers(transcription, diarization)
    
    return {
        'transcription': transcription,
        'diarization': diarization,
        'combined_segments': combined_segments,
        'metadata': {
            'audio_path': str(audio_path),
            'num_transcription_segments': len(transcription),
            'num_diarization_segments': len(diarization),
            'num_speakers': len(set(seg['speaker'] for seg in diarization)) if diarization else 0
        }
    }

def align_transcription_with_speakers(
    transcription: List[Dict[str, Any]], 
    diarization: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Align transcription segments with speaker diarization
    
    Args:
        transcription: List of transcription segments
        diarization: List of speaker diarization segments
        
    Returns:
        List of combined segments with speaker labels
    """
    combined = []
    
    for trans_seg in transcription:
        # Find overlapping speaker segments
        trans_start = trans_seg['start']
        trans_end = trans_seg['end']
        
        # Find the speaker with the most overlap
        best_speaker = "UNKNOWN"
        max_overlap = 0
        
        for diar_seg in diarization:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            
            # Calculate overlap
            overlap_start = max(trans_start, diar_start)
            overlap_end = min(trans_end, diar_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = diar_seg['speaker']
        
        # Create combined segment
        combined_segment = trans_seg.copy()
        combined_segment['speaker'] = best_speaker
        combined_segment['speaker_confidence'] = max_overlap / (trans_end - trans_start) if trans_end > trans_start else 0
        
        combined.append(combined_segment)
    
    return combined

def transcribe(audio_path: Path) -> List[Dict[str, Any]]:
    """
    Simple transcription function (maintains compatibility)
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        List of transcription segments
    """
    transcriber = WhisperTranscriber()
    return transcriber.transcribe(audio_path)