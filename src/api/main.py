# /src/6_api/main.py

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# --- MODIFICATION START ---
# We are only importing the router we need for our simple text analysis
from .routers import mvp 

# Comment out the routers that depend on the database for now
# from .routers import preprocess, analyse, rag, output 
# --- MODIFICATION END ---

app = FastAPI(
    title="MyMind Therapeutic AI - MVP Mode",
    description="Running in a minimal mode for text file analysis.",
    version="1.0.0"
)

# --- MODIFICATION START ---
# Include only the MVP router
app.include_router(mvp.router)

# Comment out the other routers
# app.include_router(preprocess.router)
# app.include_router(analyse.router)
# app.include_router(rag.router)
# app.include_router(output.router)
# --- MODIFICATION END ---

# Add an endpoint to serve the index.html file
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def read_root():
    """Serves the simple analysis UI."""
    with open("index.html") as f:
        return f.read()