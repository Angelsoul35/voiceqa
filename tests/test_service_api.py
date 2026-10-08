from __future__ import annotations

from fastapi.testclient import TestClient

from voiceqa.config import Settings
from voiceqa.llm.fake import FailingLLM
from voiceqa.service import VoiceQAService


def _service() -> VoiceQAService:
    # FailingLLM -> retrieval answers and heuristic grading (deterministic, offline).
    return VoiceQAService(settings=Settings(), llm=FailingLLM())


def test_in_scope_question_is_answered_and_confident():
    result = _service().answer_question("what happens if I pay late?")
    assert "fee" in result["answer"].lower()
    assert result["confident"] is True
    assert result["escalate"] is False
    assert "late_payment" in result["citations"]


def test_out_of_scope_question_escalates():
    result = _service().answer_question("what is the capital of France?")
    assert result["relevant"] is False
    assert result["escalate"] is True


def test_answer_audio_uses_transcript():
    result = _service().answer_audio("how do I replace a lost sim?")
    assert result["transcript"] == "how do I replace a lost sim?"
    assert result["transcription_engine"] == "provided"
    assert "sim_replacement" in result["citations"]


def _client(service):
    import voiceqa.main as main

    main._service = service
    return TestClient(main.app)


def test_api_endpoints():
    client = _client(_service())
    assert client.get("/health").json() == {"status": "ok"}

    ask = client.post("/ask", json={"question": "when is my bill issued?"})
    assert ask.status_code == 200
    assert ask.json()["citations"]

    voice = client.post("/ask/voice", json={"transcript": "how do I cancel my contract?"})
    assert voice.status_code == 200
    assert voice.json()["transcript"] == "how do I cancel my contract?"

    assert client.post("/ask", json={"question": ""}).status_code == 422
