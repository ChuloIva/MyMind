# Data Storage

This directory contains all data for the MyMind therapeutic AI application, organized by processing stage and data type.

## Directory Structure

```
data/
├── raw_audio/           # Original audio files from therapy sessions
├── transcripts/         # Processed text transcripts with timestamps
└── processed_data/      # Analysis results and extracted insights
```

## Data Flow

1. **Raw Audio** → Upload therapy session recordings
2. **Transcripts** → Whisper transcription with speaker diarization
3. **Processed Data** → GPT-4o analysis, keywords, and therapeutic insights

## Storage Guidelines

### Raw Audio (`raw_audio/`)
- **Supported formats**: WAV, MP3, M4A, FLAC
- **Quality**: 16kHz+ sample rate recommended
- **Privacy**: All audio files are processed locally
- **Naming**: Use session UUIDs for file identification

### Transcripts (`transcripts/`)
- **Format**: JSON with word-level timestamps
- **Content**: Speaker-separated text with temporal markers
- **Metadata**: Session ID, duration, speaker count
- **Structure**: Compatible with downstream analysis pipeline

### Processed Data (`processed_data/`)
- **Keywords**: Extracted terms with sentiment scores
- **Embeddings**: Vector representations for visualization
- **Insights**: Therapeutic analysis results
- **Reports**: Generated therapeutic summaries

## Data Privacy & Security

- All processing occurs locally or on secure infrastructure
- No sensitive data is transmitted to external services
- Audio files are deleted after processing (configurable)
- Transcripts are anonymized and encrypted at rest

## File Naming Convention

```
{session_id}_{timestamp}_{type}.{extension}
```

Example:
- `550e8400-e29b-41d4-a716-446655440000_20231201_1430_raw.wav`
- `550e8400-e29b-41d4-a716-446655440000_20231201_1430_transcript.json`
- `550e8400-e29b-41d4-a716-446655440000_20231201_1430_analysis.json`

## Data Retention Policy

- **Raw Audio**: 30 days (configurable)
- **Transcripts**: 1 year (encrypted)
- **Processed Data**: 2 years (anonymized)
- **Reports**: Indefinite (client-controlled)
