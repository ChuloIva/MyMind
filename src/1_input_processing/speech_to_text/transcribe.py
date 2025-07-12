from faster_whisper import WhisperModel
from pathlib import Path

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe(audio_path: Path) -> list[dict]:
    segs, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    return [s._asdict() for s in segs]