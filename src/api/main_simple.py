# Simplified API main file for therapy admin
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routers import client_management, session_management

app = FastAPI(
    title="MyMind Therapy Admin",
    description="Therapy practice management system",
    version="1.0.0"
)

# Include the client and session management routers
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}