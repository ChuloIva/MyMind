# Therapeutic AI – Closer‑to‑Metal Blueprint 

```
├── data/ …                    # no change
├── docs/ …
├── src/
│   ├── 1_input_processing/
│   │   └── speech_to_text/
│   │       └── transcribe.py          <- Whisper wrapper lives here
│   ├── 2_preprocessing/
│   │   └── gemini_processing/
│   │       └── keyword_extraction.py  <- GPT‑4o keyword + sentiment
│   ├── 3_analysis/
│   │   ├── nlp/graph_construction/
│   │   │   └── graph_builder.py       <- embedding visualiser
│   │   ├── rag/rag.py                <- LangChain RetrievalQA
│   │   └── therapeutic_methods/
│   │       └── distortions.py        <- CBT/schema checks (NEW)
│   ├── 4_profiling/
│   │   └── needs_assessment/
│   │       └── summarise.py          <- trajectories & client metrics
│   ├── 5_output/
│   │   └── generate_report.py        <- Markdown/streaming output
│   ├── 6_api/                        <- FastAPI gateway
│   │   ├── main.py                   <- entry‑point (NEW)
│   │   └── routers/
│   │       ├── preprocess.py
│   │       ├── analyse.py
│   │       ├── rag.py
│   │       └── output.py
│   └── 7_database/
│       ├── models.py                 <- SQLModel tables
│       └── migrations/
└── ui/
    ├── chat/ …
    ├── dashboard/ …
    └── profile/ …
```

---

## 1  Input Processing (`src/1_input_processing`)

### `speech_to_text/transcribe.py`

```python
# whisper-large‑v3 wrapper (unchanged)
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe(audio_path: Path) -> list[dict]:
    segs, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    return [s._asdict() for s in segs]
```

**Add** `speaker_diarisation.py` beside `transcribe.py`:

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

def diarise(wav: Path):
    diar = pipeline(wav)
    return [(t.start, t.end, t.label) for t in diar.itertracks(yield_label=True)]
```

Update `requirements.txt`:

```
faster-whisper==1.0.0
pyannote-audio==2.1.0
```

---

## 2  Pre‑Processing (`src/2_preprocessing/gemini_processing`)

Rename to **`llm_processing`** (Gemini ➜ GPT‑4o) or just keep folder and add files.

### `keyword_extraction.py`

```python
from openai import OpenAI
client = OpenAI()

PROMPT = "Return JSON: [{sentence_id, keywords:[{term,sentiment,start_ms,end_ms}]}]"

def extract(text: str):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT + text}]
    )
    return json.loads(res.choices[0].message.content)
```

Persist results in Postgres via helper in `src/7_database/models.py` (see §7).

---

## 3  Analysis Engine (`src/3_analysis`)

### 3.1 Embedding Visualiser

Move UMAP/t‑SNE code into **`nlp/graph_construction/graph_builder.py`**:

```python
from libs.embeddings import embed_batch  # … see §libs note below
import umap, numpy as np

def build(nodes: list[str]):
    vecs = embed_batch(nodes)
    xy = umap.UMAP(n_components=2).fit_transform(np.array(vecs))
    return [{"id": n, "x": float(x), "y": float(y)} for n,(x,y) in zip(nodes, xy)]
```

> **Note:** create `src/3_analysis/nlp/embeddings.py` if you prefer local helper; otherwise keep common util under a new internal package `src/common/`.

### 3.2 Therapeutic Method Evaluations

Add **`therapeutic_methods/distortions.py`**:

```python
from openai import OpenAI, AsyncOpenAI
client = OpenAI()
TEMPLATE = "Identify cognitive distortions… Return JSON: {distortions:[…]}"

def analyse(transcript: str):
    r = client.chat.completions.create(
        model="gpt-4o-large", temperature=0,
        response_format={"type":"json_object"},
        messages=[{"role":"user","content":TEMPLATE+transcript}]
    )
    return json.loads(r.choices[0].message.content)
```

---

## 4  Profiling (`src/4_profiling`)

Place trajectory summariser in `needs_assessment/summarise.py`:

```python
from openai import OpenAI; client = OpenAI()

def compute(client_id: UUID, transcript: str):
    prompt = "Summarise stress_index etc:" + transcript
    res = client.chat.completions.create(
        model="gpt-4o-mini", response_format={"type":"json_object"},
        messages=[{"role":"user", "content": prompt}])
    return json.loads(res.choices[0].message.content)
```

---

## 5  Output Layer (`src/5_output/generate_report.py`)

```python
from fastapi.responses import StreamingResponse
from openai import OpenAI, AsyncStream
from .templates import build_prompt  # build from DB

client = OpenAI()

def stream(session_id: UUID):
    stream = client.chat.completions.create(
        model="gpt-4o-large", stream=True,
        messages=[{"role":"user","content": build_prompt(session_id)}])
    return StreamingResponse(to_event_stream(stream), media_type="text/event-stream")
```

---

## 6  API Gateway (`src/6_api`)

```
6_api/
├─ main.py            # ``uvicorn src.6_api.main:app --reload``
└─ routers/
   ├─ preprocess.py   # POST /preprocess/{session_id}
   ├─ analyse.py      # POST /analyse/{session_id}
   ├─ rag.py          # POST /qa/{session_id}
   └─ output.py       # GET  /output/{session_id}
```

Each router simply wraps corresponding library functions above.

`main.py` skeleton:

```python
from fastapi import FastAPI
from .routers import preprocess, analyse, rag, output
app = FastAPI()
for r in (preprocess, analyse, rag, output):
    app.include_router(r.router)
```

---

## 7  Database Layer (`src/7_database`)

### `models.py`

```python
from sqlmodel import Field, SQLModel, Index
class SessionSentence(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    start_ms: int; end_ms: int; speaker: str; text: str
    keywords: dict | None   # jsonb
    chunks:   dict | None   # jsonb
    __table_args__ = (Index("idx_keywords", "keywords", postgresql_using="gin"),)
```

Create `alembic` env in `migrations/` or just use `sqlmodel.SQLModel.metadata.create_all(engine)` for local dev.

---

## 8  Shared Libraries (**new \*\*\*\*\*\*\*\*\*\*\*\*`src/common`**)

If you prefer not to duplicate utilities, add a folder:

```
src/common/
├─ embeddings.py   # text-embedding-3-small helper
├─ tsne.py         # UMAP/t-SNE wrapper
└─ openai_utils.py # streaming helpers
```

Import via `from common.embeddings import embed_batch`.

---

## 9  UI (`ui/…`)

The React/D3 hooks from the previous blueprint drop into:

```
ui/dashboard/src/hooks/useScatter.ts
```

and so on—keeping the existing Vite/Tailwind setup.

---

## 10  Scripts & Tests

* **`scripts/`** → keep for ad‑hoc CLI (e.g. `fetch_papers.py`).
* **`tests/`** → add pytest suites using synthetic audio in `data/raw_audio`.

---
