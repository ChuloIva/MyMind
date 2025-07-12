# Speech-to-Text with Whisper & Speaker Diarization

This module provides high-quality speech-to-text transcription using OpenAI's Whisper model combined with Pyannote speaker diarization for therapy session analysis.

## Core Components

### 1. Transcription (`transcribe.py`)
- **Model**: `whisper-large-v3` (most accurate)
- **Device**: CUDA-accelerated when available
- **Precision**: `float16` for optimal performance
- **Features**: Word-level timestamps, confidence scores

### 2. Speaker Diarization (`speaker_diarisation.py`)
- **Model**: `pyannote/speaker-diarization-3.1`
- **Capability**: Automatic speaker detection
- **Output**: Speaker labels with time boundaries
- **Integration**: Merged with transcription results

## Implementation Details

### Whisper Configuration
```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3",
    device="cuda",           # Falls back to CPU if no GPU
    compute_type="float16"   # Optimized precision
)

def transcribe(audio_path: Path) -> list[dict]:
    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,            # Accuracy vs speed balance
        word_timestamps=True    # Word-level timing
    )
    return [s._asdict() for s in segments]
```

### Speaker Diarization Setup
```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="your_token"  # Required for model access
)

def diarise(wav: Path) -> list[tuple]:
    diarization = pipeline(wav)
    return [
        (turn.start, turn.end, turn.label)
        for turn in diarization.itertracks(yield_label=True)
    ]
```

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. GPU Setup (Optional)
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Pyannote Authentication
```bash
# Get token from https://huggingface.co/pyannote/speaker-diarization-3.1
huggingface-cli login
```

### 4. Install FFmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage Examples

### Basic Transcription
```python
from pathlib import Path
from transcribe import transcribe

# Transcribe audio file
audio_file = Path("therapy_session.wav")
segments = transcribe(audio_file)

# Output structure
for segment in segments:
    print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")
```

### Speaker Diarization
```python
from speaker_diarisation import diarise

# Identify speakers
speakers = diarise(Path("therapy_session.wav"))

# Output: [(start_time, end_time, speaker_label)]
for start, end, speaker in speakers:
    print(f"{speaker}: {start:.1f}s - {end:.1f}s")
```

### Combined Processing
```python
from pathlib import Path
from transcribe import transcribe
from speaker_diarisation import diarise

def process_session(audio_path: Path):
    # Get transcription
    segments = transcribe(audio_path)
    
    # Get speaker information
    speakers = diarise(audio_path)
    
    # Combine results (implementation depends on alignment logic)
    return merge_transcription_speakers(segments, speakers)
```

## Performance Optimization

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 or better (4GB+ VRAM)
- **RAM**: 16GB+ recommended for large files
- **Storage**: SSD for faster audio file I/O
- **CPU**: Multi-core for parallel processing

### Processing Speed
- **Real-time factor**: 0.3x with GPU (30s to process 100s audio)
- **CPU processing**: 2-3x slower than GPU
- **Batch processing**: Process multiple files in parallel
- **Memory management**: Automatic cleanup between sessions

### Quality Settings
```python
# High accuracy (slower)
segments = model.transcribe(
    audio,
    beam_size=5,
    best_of=5,
    temperature=0.0
)

# Faster processing (slightly lower accuracy)
segments = model.transcribe(
    audio,
    beam_size=3,
    best_of=3,
    temperature=0.2
)
```

## Output Format

### Transcription Structure
```json
{
  "id": 0,
  "seek": 0,
  "start": 0.0,
  "end": 3.2,
  "text": "Hello, how are you feeling today?",
  "tokens": [50364, 2425, 11, 577, 366, 291, 2633, 965, 30, 50414],
  "temperature": 0.0,
  "avg_logprob": -0.15,
  "compression_ratio": 1.3,
  "no_speech_prob": 0.01,
  "words": [
    {
      "word": "Hello",
      "start": 0.0,
      "end": 0.5,
      "probability": 0.98
    }
  ]
}
```

### Speaker Diarization Output
```python
# Format: (start_time, end_time, speaker_label)
[
    (0.0, 3.2, "SPEAKER_00"),    # Therapist
    (3.2, 6.1, "SPEAKER_01"),    # Client
    (6.1, 9.5, "SPEAKER_00"),    # Therapist
]
```

## Quality Assurance

### Confidence Thresholds
- **High confidence**: `avg_logprob > -0.5`
- **Medium confidence**: `avg_logprob > -1.0`
- **Low confidence**: `avg_logprob <= -1.0` (flag for review)

### Error Detection
- **No speech**: `no_speech_prob > 0.7`
- **Repetitive output**: `compression_ratio > 2.4`
- **Hallucination**: Very low probability scores

### Speaker Accuracy
- **Validation**: Cross-reference with manual annotations
- **Consistency**: Speaker labels across session segments
- **Confidence**: Diarization confidence scores

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce compute type
model = WhisperModel("large-v3", device="cuda", compute_type="int8")
```

**2. Pyannote Authentication Error**
```bash
# Re-authenticate
huggingface-cli login --token YOUR_TOKEN
```

**3. Poor Audio Quality**
- Check sample rate (16kHz minimum)
- Verify file format (WAV preferred)
- Ensure adequate recording levels
- Reduce background noise

**4. Slow Processing**
- Use GPU acceleration
- Reduce beam size for faster processing
- Process shorter audio segments
- Use smaller Whisper model for development

### Performance Monitoring
```python
import time
import psutil
import torch

def monitor_processing(audio_path):
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    if torch.cuda.is_available():
        start_gpu = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Process audio
    segments = transcribe(audio_path)
    
    # Log performance metrics
    processing_time = time.time() - start_time
    memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
    
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Memory used: {memory_used:.2f}MB")
    
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated() / 1024 / 1024 - start_gpu
        print(f"GPU memory used: {gpu_used:.2f}MB")
```

## Integration with Database

### Session Storage
```python
from uuid import uuid4
from src.database.models import Session, SessionSentence

def save_transcription(audio_path: Path, client_id: str):
    session_id = uuid4()
    
    # Create session record
    session = Session(
        id=session_id,
        client_id=client_id,
        created_at=datetime.now().isoformat()
    )
    
    # Process audio
    segments = transcribe(audio_path)
    speakers = diarise(audio_path)
    
    # Save segments
    for segment in segments:
        sentence = SessionSentence(
            session_id=session_id,
            start_ms=int(segment['start'] * 1000),
            end_ms=int(segment['end'] * 1000),
            text=segment['text'],
            speaker=map_speaker(segment, speakers)
        )
        # Save to database
```

This implementation provides the foundation for all downstream analysis, ensuring high-quality structured data for therapeutic insights.
