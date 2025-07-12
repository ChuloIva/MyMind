from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

def diarise(wav: Path):
    diar = pipeline(wav)
    return [(t.start, t.end, t.label) for t in diar.itertracks(yield_label=True)]
