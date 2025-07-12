# Speech to Text using OpenAI's Whisper

This directory contains the implementation of the speech-to-text functionality using OpenAI's Whisper model.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install FFmpeg:**
    Whisper requires FFmpeg to be installed on your system. You can install it using Homebrew on macOS:
    ```bash
    brew install ffmpeg
    ```
    For other operating systems, please refer to the official FFmpeg website for installation instructions.

## Usage

The `transcribe.py` script provides a basic example of how to use Whisper to transcribe an audio file.

```bash
python transcribe.py <path_to_audio_file>
```
