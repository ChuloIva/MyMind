# /src/6_api/main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .routers import mvp, preprocess, analyse, rag, output, profiling

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

# Add an endpoint to serve the index.html file
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_root():
    """Serves the simple analysis UI."""
    with open("index.html") as f:
        return f.read()