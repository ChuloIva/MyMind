import whisper
import sys

if len(sys.argv) < 2:
    print("Usage: python transcribe.py <path_to_audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

model = whisper.load_model("base")
result = model.transcribe(audio_file)
print(result["text"])
