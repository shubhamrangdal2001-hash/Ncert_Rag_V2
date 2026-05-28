import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import load_chunks_from_disk, rebuild_retriever
from stage3_generation import StudyAssistantV2, AgenticStudyAssistant, build_llm


ROOT = Path(__file__).parent
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="NCERT Agentic RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


class AskRequest(BaseModel):
    question: str
    agentic: bool = True
    k: int = 5
    api_key: Optional[str] = None


class AssistantManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: Dict[str, Any] = {}

    def get(self, *, agentic: bool, k: int, api_key: Optional[str]) -> Any:
        key_name = "agentic" if agentic else "classic"
        cache_key = f"{key_name}:{k}:{'provided' if api_key else 'env'}"

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

            chunks = load_chunks_from_disk()
            retriever = rebuild_retriever(chunks, k=k)
            llm = build_llm(api_key or os.environ.get("GROQ_API_KEY", ""))
            cls = AgenticStudyAssistant if agentic else StudyAssistantV2
            assistant = cls(retriever, llm, k=k, use_strict_prompt=True)
            self._cache[cache_key] = assistant
            return assistant


manager = AssistantManager()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        assistant = manager.get(agentic=req.agentic, k=req.k, api_key=req.api_key)
        return assistant.ask(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def root() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index_file))
