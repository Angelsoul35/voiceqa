"""The VoiceQA pipeline: transcribe -> retrieve -> answer -> grade."""

from __future__ import annotations

from .answerer import Answerer
from .config import Settings, get_settings
from .grader import Grader
from .knowledge import KnowledgeBase
from .llm.base import LLMClient
from .llm.factory import build_llm
from .transcribe import ProvidedTranscript, Transcriber


class VoiceQAService:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        transcriber: Transcriber | None = None,
        knowledge: KnowledgeBase | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else build_llm(self.settings.llm_spec)
        self.transcriber = transcriber or ProvidedTranscript()
        self.knowledge = knowledge or KnowledgeBase()
        self.answerer = Answerer(llm=self.llm, max_tokens=self.settings.llm_max_tokens)
        self.grader = Grader(
            llm=self.llm,
            min_groundedness=self.settings.min_groundedness,
            max_tokens=self.settings.llm_max_tokens,
        )

    def answer_question(self, question: str) -> dict:
        passages = self.knowledge.retrieve(question, k=self.settings.top_k)
        answer = self.answerer.answer(question, passages)
        grade = self.grader.grade(question, answer.text, passages)
        return {
            "question": question,
            "answer": answer.text,
            "answer_source": answer.source,
            "citations": answer.citations,
            "retrieved": [{"topic": p.topic, "score": p.score} for p in passages],
            "groundedness": grade.groundedness,
            "relevant": grade.relevant,
            "confident": grade.confident,
            "grade_source": grade.source,
            "grade_rationale": grade.rationale,
            "escalate": not grade.confident,
        }

    def answer_audio(self, audio_path: str) -> dict:
        transcript = self.transcriber.transcribe(audio_path)
        result = self.answer_question(transcript.text)
        result["transcript"] = transcript.text
        result["transcription_engine"] = transcript.engine
        return result
