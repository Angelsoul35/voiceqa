"""VoiceQA HTTP API."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .service import VoiceQAService

app = FastAPI(
    title="VoiceQA",
    version="1.0.0",
    summary="Spoken-question answering with retrieval grounding and LLM grading.",
)

_service: VoiceQAService | None = None


def get_service() -> VoiceQAService:
    global _service
    if _service is None:
        _service = VoiceQAService()
    return _service


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class TranscriptRequest(BaseModel):
    transcript: str = Field(min_length=1)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", tags=["qa"])
def ask(request: QuestionRequest) -> dict:
    return get_service().answer_question(request.question)


@app.post("/ask/voice", tags=["qa"])
def ask_voice(request: TranscriptRequest) -> dict:
    # The transcript stands in for uploaded audio; the default transcriber
    # treats it as the recognised text.
    return get_service().answer_audio(request.transcript)
