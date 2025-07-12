from fastapi import FastAPI
from .routers import preprocess, analyse, rag, output
app = FastAPI()
for r in (preprocess, analyse, rag, output):
    app.include_router(r.router)
