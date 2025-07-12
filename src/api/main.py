# /src/api/main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from .routers import preprocess, analyse, rag, output # analyse is already imported

app = FastAPI(
    title="MyMind Therapeutic AI",
    description="Full system with integrated MVP for text analysis.",
    version="1.0.0"
)

# Include all your existing routers
app.include_router(preprocess.router)
app.include_router(analyse.router) # This now includes our new endpoint
app.include_router(rag.router)
app.include_router(output.router)

# Add an endpoint to serve the index.html file
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_root():
    """Serves the simple analysis UI."""
    with open("index.html") as f:
        return f.read()