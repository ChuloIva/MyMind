# /src/6_api/main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routers import mvp, analyse, rag, output, profiling
from .routers import preprocess_simple as preprocess
from .routers import client_management, session_management

app = FastAPI(
    title="MyMind Therapeutic AI",
    description="Therapeutic AI platform with needs-based profiling",
    version="1.0.0"
)

app.include_router(mvp.router)
app.include_router(preprocess.router)
app.include_router(analyse.router)
app.include_router(rag.router)
app.include_router(output.router)
app.include_router(profiling.router)
app.include_router(client_management.router)
app.include_router(session_management.router)

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="."), name="static")

# Add an endpoint to serve the therapy admin UI
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_root():
    """Serves the therapy admin UI."""
    with open("therapy_admin.html") as f:
        return f.read()

# Add endpoint for the old analysis UI
@app.get("/analysis", response_class=HTMLResponse, tags=["UI"])
async def analysis_ui():
    """Serves the simple analysis UI."""
    with open("index.html") as f:
        return f.read()