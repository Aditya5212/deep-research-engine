from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.research.checkpointer import close_checkpointer
from src.research.controller import router as research_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_checkpointer()


app = FastAPI(lifespan=lifespan)

app.include_router(research_router)


@app.get("/")
def root():
    return {"message": "Hello from deep-research-engine!"}


@app.get("/health")
def health():
    return {"status": "ok"}
