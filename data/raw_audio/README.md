# Raw Audio Files

This directory stores original audio files from therapy sessions before processing.

## Supported Audio Formats

- **WAV**: Uncompressed, highest quality (recommended)
- **MP3**: Compressed format, good for file size
- **M4A**: Apple format, high quality
- **FLAC**: Lossless compression

## Audio Requirements

### Quality Specifications
- **Sample Rate**: 16kHz minimum (44.1kHz recommended)
- **Bit Depth**: 16-bit minimum (24-bit recommended)
- **Channels**: Mono or stereo (stereo preferred for speaker separation)
- **Duration**: 10 minutes to 2 hours typical session length

### Recording Guidelines
- **Environment**: Quiet room with minimal background noise
- **Microphone**: Use quality recording equipment
- **Positioning**: Microphone equidistant from speakers
- **Levels**: Avoid clipping, maintain consistent volume

## Processing Pipeline

1. **Upload**: Audio files placed in this directory
2. **Validation**: Format and quality checks
3. **Transcription**: Whisper large-v3 model processing
4. **Speaker Diarization**: Pyannote speaker separation
5. **Cleanup**: Original files moved to archive (optional)

## File Naming

Use session UUIDs for privacy and organization:
```
{session_id}_{timestamp}.{extension}
```

Example:
```
550e8400-e29b-41d4-a716-446655440000_20231201_1430.wav
```

## Storage Management

- **Automatic Processing**: Files are processed on upload
- **Backup**: Optional cloud backup with encryption
- **Retention**: 30-day default retention policy
- **Cleanup**: Automatic deletion after processing (configurable)

## Privacy & Security

- All audio files are processed locally
- No audio data transmitted to external services
- Encryption at rest for sensitive content
- Secure deletion with overwrite patterns

## Troubleshooting

### Common Issues
- **Format not supported**: Convert to WAV or MP3
- **File too large**: Compress or split sessions
- **Poor quality**: Check microphone setup
- **No speech detected**: Verify audio levels

### Performance Tips
- **Batch processing**: Process multiple files together
- **GPU acceleration**: Use CUDA for faster transcription
- **Storage optimization**: Use compression for archival
