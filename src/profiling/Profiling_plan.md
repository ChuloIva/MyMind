Of course. Here is a detailed, file-by-file implementation plan for the Needs-Based Profiling System, created by analyzing your provided codebase and high-level plan.

---

# Implementation Plan: Needs-Based Profiling System

This document outlines the step-by-step plan to implement the Needs-Based Profiling System. The plan leverages the existing codebase, focusing on connecting, refining, and extending the modules to deliver the new feature.

## Phase 1: Data Foundation & Setup

**Goal:** Finalize the database schema for needs profiling and load the essential reference data.

### 1.1. Review and Finalize Database Models
**File:** `src/database/models.py`

The necessary models (`NeedCategory`, `LifeSegment`, `ClientNeedProfile`, `ClientNeedSummary`) are already defined. This task is to review and confirm their structure.

**Action:**
- No code changes are immediately needed. The existing models align perfectly with the plan.

```python
# src/database/models.py (Relevant excerpts - No changes needed)

class NeedCategory(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    need: str = Field(unique=True, index=True)
    # ...

class LifeSegment(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    life_area: str = Field(index=True)
    # ...

class ClientNeedProfile(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id")
    # ...

class ClientNeedSummary(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    client_id: UUID = Field(foreign_key="client.id", unique=True)
    # ...
```

### 1.2. Enable Table Creation
**File:** `init_db.py`

The database initialization script needs to be aware of the new models to create their corresponding tables.

**Action:**
- Import the new models into `init_db.py`.

```python
# init_db.py (UPDATE)

#!/usr/bin/env python3
"""Database initialization script for MyMind project."""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlmodel import SQLModel
from src.database.database import engine, create_db_and_tables

# Import all models to ensure they're registered
from src.database.models import (
    Client, 
    Session, 
    SessionAnalysis, 
    NeedCategory,          # ADD THIS
    LifeSegment,           # ADD THIS
    ClientNeedProfile,     # ADD THIS
    ClientNeedSummary      # ADD THIS
)

def main():
    """Initialize the database and create all tables."""
    print("Initializing MyMind database...")
    
    try:
        # Create all tables
        create_db_and_tables()
        print("✅ Database tables created successfully!")
        
        # Test basic connection
        with engine.connect() as conn:
            print("✅ Database connection test passed!")
            
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 1.3. Load Reference Data
**File:** `scripts/load_reference_data.py`

This script is ready to use. It will populate the `NeedCategory` and `LifeSegment` tables.

**Action:**
1.  Place the required CSV files (`universal_needs.csv`, `sdt_needs.csv`, `life_segments.csv`) in the same directory as the script.
2.  Run the database initialization script first: `python init_db.py`.
3.  Run the data loading script: `python scripts/load_reference_data.py`.

---

## Phase 2: AI-Powered Extraction Pipeline

**Goal:** Implement the AI logic to analyze transcripts and map content to needs and life segments.

### 2.1. Integrate Needs Extraction into Processing Flow
**File:** `src/api/routers/preprocess.py`

We need a new endpoint to trigger the needs extraction for a completed transcript. This will follow the pattern of the existing `transcribe` and `keywords` endpoints.

**Action:**
- Add a new endpoint `POST /needs/{session_id}` to the `preprocess.py` router.
- This endpoint will queue a background task, `process_needs_background`.

```python
# src/api/routers/preprocess.py (ADDITIONS)

# Add to top-level imports
from ...preprocessing.llm_processing.needs_extraction import NeedsExtractor
from ...database.models import ClientNeedProfile, NeedCategory, LifeSegment

# Add new background task function
async def process_needs_background(session_id: UUID):
    """Background task for needs extraction."""
    from ...database.database import SessionLocal
    try:
        with SessionLocal() as db:
            sentences = db.query(SessionSentence).filter(SessionSentence.session_id == session_id).order_by(SessionSentence.sentence_index).all()
            if not sentences:
                logger.error(f"No sentences found for needs extraction in session {session_id}")
                return

            segments_to_process = [{'text': s.text, 'start': s.start_ms / 1000} for s in sentences]
            
            extractor = NeedsExtractor(api_key=settings.openai_api_key)
            extracted_data = extractor.extract_needs_and_segments(segments_to_process)

            # Store results in ClientNeedProfile table
            # (Implementation details for matching needs/segments to IDs omitted for brevity)
            # ... database saving logic here ...
            logger.info(f"Needs extraction completed for session {session_id}")

    except Exception as e:
        logger.error(f"Needs extraction background task failed: {e}")

# Add new endpoint to the router
@router.post("/needs/{session_id}")
async def extract_needs_from_session(
    session_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_session)
):
    """Extract needs and life segments from session transcription."""
    session = db.get(SessionModel, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    background_tasks.add_task(process_needs_background, session_id=session_id)
    
    return {
        "session_id": str(session_id),
        "message": "Needs extraction started",
        "status": "processing"
    }
```

### 2.2. Review Core Extraction Logic
**File:** `src/preprocessing/llm_processing/needs_extraction.py`

The `NeedsExtractor` class and its prompt are well-defined.

**Action:**
- No changes needed initially. The prompt will be refined in Phase 6 after testing with real data.

---

## Phase 3: Aggregation & Analytics

**Goal:** Implement the logic to aggregate session-level data into a comprehensive client profile.

### 3.1. Implement Profiler Logic
**File:** `src/profiling/needs_assessment/needs_profiler.py`

The `NeedsProfiler` class contains placeholder logic. We need to implement the database interactions.

**Action:**
- Implement the `_get_session_segments` and `_save_extractions` methods.
- Refine `build_client_profile` to fetch all necessary data from the database.
- Implement logic in `_aggregate_profile` to save the `ClientNeedSummary` to the database.

```python
# src/profiling/needs_assessment/needs_profiler.py (UPDATE)

from sqlmodel import Session, select
from src.database.database import get_session
from src.database.models import ClientNeedProfile, ClientNeedSummary, SessionSentence

class NeedsProfiler:
    def __init__(self, db_session: Session):
        self.db = db_session
        # ... (extractor init)

    def build_client_profile(self, client_id: UUID, session_ids: List[UUID]) -> ClientNeedSummary:
        """Build or update comprehensive needs profile for client."""
        # 1. Fetch all existing extractions for the client
        statement = select(ClientNeedProfile).where(ClientNeedProfile.client_id == client_id)
        all_extractions = self.db.exec(statement).all()

        # 2. Aggregate data into a summary object
        summary = self._aggregate_profile(client_id, all_extractions)

        # 3. Save or update the summary in the database
        existing_summary = self.db.exec(
            select(ClientNeedSummary).where(ClientNeedSummary.client_id == client_id)
        ).first()

        if existing_summary:
            existing_summary.life_segment_scores = summary.life_segment_scores
            existing_summary.need_fulfillment_scores = summary.need_fulfillment_scores
            existing_summary.unmet_needs = summary.unmet_needs
            existing_summary.fulfilled_needs = summary.fulfilled_needs
            existing_summary.last_updated = datetime.utcnow()
            self.db.add(existing_summary)
            summary_to_return = existing_summary
        else:
            self.db.add(summary)
            summary_to_return = summary

        self.db.commit()
        self.db.refresh(summary_to_return)
        return summary_to_return

    # ... (rest of the class)
```

---

## Phase 4: API & Integration

**Goal:** Expose the profiling system through secure and efficient API endpoints.

### 4.1. Activate Profiling Router
**File:** `src/api/main.py`

The main FastAPI application needs to include the new `profiling` router.

**Action:**
- Uncomment or add the `profiling` router.

```python
# src/api/main.py (MODIFICATION)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .routers import mvp, profiling # ADD 'profiling'

app = FastAPI(...)

app.include_router(mvp.router)
app.include_router(profiling.router) # ADD THIS LINE
```

### 4.2. Implement API Endpoint Logic
**File:** `src/api/routers/profiling.py`

The endpoints are defined but their helper functions are placeholders.

**Action:**
- Replace placeholder functions with actual database queries and service calls.
- `analyze-needs` should trigger a background task for `NeedsProfiler`.

```python
# src/api/routers/profiling.py (UPDATE)

# ... (imports)
from fastapi import BackgroundTasks
from src.database.models import Session as SessionModel

# Remove placeholder helpers and implement logic directly or in a service layer.

async def run_profiling_background(client_id: UUID, session_ids: List[UUID]):
    """Background task to run the needs profiler."""
    from src.database.database import SessionLocal
    with SessionLocal() as db:
        profiler = NeedsProfiler(db_session=db)
        profiler.build_client_profile(client_id, session_ids)
        logger.info(f"Needs profile updated for client {client_id}")

@router.post("/clients/{client_id}/analyze-needs")
async def analyze_client_needs(
    client_id: UUID,
    background_tasks: BackgroundTasks,
    session_count: int = 10,
    db: Session = Depends(get_session)
):
    """Trigger a background task to build/update a client's needs profile."""
    # Implement database query to get recent sessions
    session_ids = db.exec(
        select(SessionModel.id)
        .where(SessionModel.client_id == client_id)
        .order_by(SessionModel.created_at.desc())
        .limit(session_count)
    ).all()

    if not session_ids:
        raise HTTPException(status_code=404, detail="No sessions found for this client.")

    background_tasks.add_task(run_profiling_background, client_id, session_ids)
    
    return {"message": "Needs profile analysis has been triggered.", "client_id": client_id}

@router.get("/clients/{client_id}/needs-dashboard")
async def get_needs_dashboard(
    client_id: UUID, 
    db: Session = Depends(get_session)
):
    """Get dashboard data for needs visualization."""
    report_generator = NeedsProfileReport()
    # The get_client_needs_profile function already queries the DB
    profile = report_generator.get_client_needs_profile(client_id)

    if not profile.life_segment_scores: # Check if profile is empty
         raise HTTPException(status_code=404, detail="Needs profile not generated yet. Please trigger analysis first.")

    # generate_visual_profile will create the data structures for the frontend charts
    return report_generator.generate_visual_profile(client_id)

```

---

## Phase 5: Visualization & Reporting

**Goal:** Create an intuitive frontend dashboard to display the needs profile.

### 5.1. Implement Backend Report Generation
**File:** `src/output/needs_report.py`

The `NeedsProfileReport` class has placeholders for generating chart data.

**Action:**
- Implement the `_create_radar_data`, `_create_bar_chart_data`, etc., methods to format the `ClientNeedSummary` data into a structure that Chart.js or D3.js can easily consume.

### 5.2. Build Frontend Components
**Files:**
- Create `ui/profile/src/components/NeedsDashboard.tsx`
- Create `ui/profile/src/components/LifeSegmentRadar.tsx`
- Create `ui/profile/src/components/NeedsFulfillmentBar.tsx`
- Create `ui/profile/src/hooks/useNeedsProfile.ts`

**Action:**
1.  **Create a data-fetching hook** using React Query to call the `/api/profiling/clients/{id}/needs-dashboard` endpoint.
2.  **Create the `NeedsDashboard` component** to orchestrate the layout and pass data to child chart components.
3.  **Implement the `LifeSegmentRadar` component** using `react-chartjs-2` to display the radar chart.
4.  **Implement the `NeedsFulfillmentBar` component** to display the bar chart for met/unmet needs.

```typescript
// Example: ui/profile/src/hooks/useNeedsProfile.ts

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../utils/api'; // Assuming you have this

export const useNeedsProfile = (clientId: string) => {
  return useQuery({
    queryKey: ['needsProfile', clientId],
    queryFn: () => apiClient.getNeedsDashboard(clientId), // a new method in your api client
    enabled: !!clientId,
  });
};
```

```typescript
// Example: ui/profile/src/components/LifeSegmentRadar.tsx

import React from 'react';
import { Radar } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

interface RadarProps {
  data: any; // Define a proper type for your chart data
}

export const LifeSegmentRadar: React.FC<RadarProps> = ({ data }) => {
  if (!data || !data.labels) return <div>Loading chart...</div>;

  const chartData = {
    labels: data.labels,
    datasets: data.datasets,
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      r: {
        angleLines: { display: false },
        suggestedMin: -1,
        suggestedMax: 1,
      },
    },
  };

  return <Radar data={chartData} options={options} />;
};
```

---

## Phase 6: Testing & Refinement

**Goal:** Ensure the system is robust, accurate, and performs well.

### 6.1. Create a New Test Suite
**File:** Create `tests/test_profiling.py`

**Action:**
- Write unit tests for the `NeedsProfiler` aggregation logic using sample data.
- Write integration tests for the `/api/profiling` endpoints using FastAPI's `TestClient`.
- Mock the OpenAI API calls to test the pipeline without incurring costs.

### 6.2. Prompt Engineering and Validation

**Action:**
1.  Create a "golden dataset" of 5-10 manually annotated transcripts.
2.  Run the `NeedsExtractor` against this dataset.
3.  Compare the AI's output with the manual annotations.
4.  Refine the prompt in `src/preprocessing/llm_processing/needs_extraction.py` based on discrepancies to improve accuracy.

### 6.3. Performance Optimization

**Action:**
1.  Use `EXPLAIN ANALYZE` on the PostgreSQL queries in `NeedsProfiler` to identify bottlenecks.
2.  Ensure proper database indexes are used for fetching `ClientNeedProfile` records.
3.  Implement caching for the `/needs-dashboard` endpoint, as profiles may not update frequently. Use a tool like `fastapi-cache`.