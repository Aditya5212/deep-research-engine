from fastapi import FastAPI

from src.research.controller import router as research_router

app = FastAPI()

app.include_router(research_router)


@app.get("/")
def root():
    return {"message": "Hello from deep-research-engine!"}


@app.get("/health")
def health():
    return {"status": "ok"}
