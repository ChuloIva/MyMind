# Transcripts

This directory contains processed text transcripts generated from audio files using Whisper and Pyannote for speaker diarization.

## Transcript Format

Each transcript is a JSON file with the following structure:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration": 3600.0,
  "created_at": "2023-12-01T14:30:00Z",
  "model": "whisper-large-v3",
  "speakers": ["therapist", "client"],
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Hello, how are you feeling today?",
      "speaker": "therapist",
      "confidence": 0.95,
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "confidence": 0.98
        }
      ]
    }
  ]
}
```

## Processing Pipeline

### 1. Speech-to-Text (Whisper)
- **Model**: `whisper-large-v3`
- **Features**: Word-level timestamps, confidence scores
- **Language**: Auto-detection with English priority
- **Beam size**: 5 for optimal accuracy

### 2. Speaker Diarization (Pyannote)
- **Model**: `pyannote/speaker-diarization-3.1`
- **Speakers**: Automatic detection (2-4 speakers typical)
- **Labels**: Mapped to therapist/client roles
- **Overlap handling**: Concurrent speech detection

### 3. Post-Processing
- **Sentence segmentation**: Natural language boundaries
- **Punctuation restoration**: Automatic capitalization
- **Timestamp alignment**: Word-level precision
- **Quality filtering**: Low-confidence segment flagging

## Data Structure

### Session Metadata
- `session_id`: Unique identifier
- `duration`: Total session length in seconds
- `created_at`: Processing timestamp
- `model`: AI model version used
- `speakers`: List of identified speakers

### Segment Details
- `id`: Sequential segment identifier
- `start`/`end`: Timestamp boundaries in seconds
- `text`: Transcribed text content
- `speaker`: Speaker identification
- `confidence`: Transcription confidence score
- `words`: Word-level breakdown with timestamps

## Quality Metrics

### Accuracy Indicators
- **Confidence Score**: 0.8+ considered high quality
- **Speaker Accuracy**: 95%+ typical performance
- **Timestamp Precision**: ±100ms accuracy
- **Word Error Rate**: <5% for clear audio

### Quality Assurance
- **Automatic validation**: Confidence threshold filtering
- **Manual review**: Low-confidence segments flagged
- **Correction workflow**: Editor interface for fixes
- **Version control**: Track transcript revisions

## File Management

### Naming Convention
```
{session_id}_{timestamp}_transcript.json
```

### Storage
- **Format**: UTF-8 encoded JSON
- **Compression**: Optional gzip compression
- **Indexing**: Full-text search capability
- **Backup**: Encrypted cloud storage

## Integration Points

### Database Storage
- Segments stored in `SessionSentence` table
- Indexed for efficient keyword search
- JSONB fields for flexible querying

### Downstream Processing
- **Keyword extraction**: GPT-4o analysis
- **Sentiment analysis**: Per-segment scoring
- **Therapeutic insights**: Pattern recognition
- **Visualization**: Embedding generation

## Privacy & Compliance

- **Anonymization**: Speaker names removed
- **Encryption**: At-rest and in-transit
- **Access control**: Role-based permissions
- **Audit trail**: Processing history logged
- **Retention**: Configurable data lifecycle
