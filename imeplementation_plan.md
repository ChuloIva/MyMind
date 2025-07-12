# Therapeutic AI Application — Comprehensive **LLM-Centric** System Report  
*(UI now shows **visualised embeddings** instead of a knowledge graph)*  

---

## 1  Input Processing Module  

### Audio → Text  
- **Whisper-large-v3** on-prem transcription  
- Low-confidence spans re-sent to **GPT-4o audio modality** for QC  

### Speaker Attribution  
- **pyannote-audio v2.1** diarisation  
- GPT-4o labels each turn as **Therapist** / **Client**  

---

## 2  Pre-Processing Module  

| Task | What GPT-4o Returns | Stored As |
|------|--------------------|-----------|
| Keyword + sentiment | `[{"sentence_id":…,"keywords":[{term,sentiment,start_ms,end_ms}]}]` | `keywords.json` |
| Chunking + timestamps | Embedded in same payload | `chunks.json` |

- All JSON persisted in **Postgres 16 `jsonb`** columns  

---

## 3  Analysis Engine  

### 3.1 Embedding Visualiser  
1. GPT-4o embeds each keyword chunk with `text-embedding-3-small`.  
2. Performs a **t-SNE/UMAP** projection to 2-D.  
3. Returns `{node_id,x,y,weight,…}` for **D3.js** to draw.  

### 3.2 Therapeutic Method Evaluations  
- Few-shot GPT-4o detects CBT distortions, schema modes, cognitive biases and suggests reframes.  
- The transcript of session is compared against the descriptions of schemas, and cognitive biases. LLM_context(promtp= compare this tanscript with following schemas/biases) = transcript + schemas/cogbiases
---

## 4  Web Search & RAG Layer  

1. Text chunks → embeddings → **pgvector**.  
2. **LangChain RetrievalQA** fetches contexts.  
3. GPT-4o synthesises answers citing DSM-5 / papers.  

---

## 5  Profiling System  

| Feature | Implementation |
|---------|----------------|
| Needs & sentiment trajectories | GPT-4o batch summarisation (`stress_index`, `positive_affect`, …). |
| Client-specific fine-tune | **LoRA adapters** via HF PEFT → uploaded to OpenAI Custom Model. |

---

## 6  Output & Combination Layer  

- GPT-4o function consumes **embedding map + CBT/Schema tags + RAG snippets**.  
- Emits Markdown: session summary, SMART goals, `priority_score`, next steps.  
- Streamed via FastAPI for real-time UI updates.  

---

## 7  User Interface & Experience  

| Page | Key Elements |
|------|--------------|
| **Analysis Dashboard** | D3 scatterplot of embeddings, KPI cards, priority list, top 5 questions. |
| **Client Profile** | Time-series charts, narrative summaries, goal tracking. |
| **Interactive Chat** | GPT-4o streaming via Socket.io; context-aware dialogue. |

---

## 8  Advanced Features  

- **Temporal Analysis**: scheduled GPT snapshots, trend & predictive insights.  
- **Question Generation**: GPT-4o crafts therapist prompts + client homework.  
- **Database Management**: linear timestamp tables, session tags, meta-snapshots.  

---

## 9  Technical Implementation  

- **FastAPI** gateway, streaming GPT deltas; immutable prompt/response log.  
- **CI/CD**: GitHub Actions → Docker → AWS Fargate; GPT-generated synthetic tests.  
- **Observability**: Sentry middleware, structured JSON logs.  
- **Security/Compliance**: pgcrypto at rest, EU AI Act-ready audit trails.  

---

## 10  Benefits & Applications  

### Therapists  
- Automated insight extraction, evidence-based suggestions, objective metrics.  

### Clients  
- Personalised resources, real-time feedback, progress dashboards.  

### Healthcare Systems  
- Scalable, data-driven mental-health delivery, cost optimisation, quality assurance.  

---

## Architecture Snapshot  

```text
┌─ Frontend (React/D3) ───────────────────────────────┐
│  WebSocket  ← Streamed GPT-4o deltas               │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─ API Gateway (FastAPI) ──────────────────────────────────────────────┐
│  pgvector DB  │  Whisper STT  │  pyannote Diariser                  │
│               │               │                                    │
│  Orchestrator → GPT-4o Fns:                                        │
│    • /preprocess   • /analyse   • /rag   • /output                 │
│                                                                  │
│  Immutable Audit Log (append-only)                                │
└─────────────────────────────────────────────────────────────────────┘