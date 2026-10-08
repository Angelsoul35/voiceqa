"""Answer a question, grounded in retrieved knowledge-base passages.

The answerer asks the LLM to answer using only the retrieved context, so the
answer stays grounded and the grader can check it. If the LLM is unreachable, it
falls back to returning the single best-matching passage's answer verbatim — a
safe, correct (if less fluent) response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge import Passage
from .llm.base import LLMClient, LLMError

_SYSTEM = (
    "You are a precise customer-support assistant. Answer the question using ONLY "
    "the provided context. If the answer is not in the context, say exactly: "
    "'I don't have that information.' Keep the answer to one or two sentences."
)


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    source: str = "retrieval"


class Answerer:
    def __init__(self, llm: LLMClient | None = None, max_tokens: int = 400) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def _prompt(self, question: str, passages: list[Passage]) -> str:
        context = "\n".join(f"- {p.context_text}" for p in passages)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def answer(self, question: str, passages: list[Passage]) -> Answer:
        citations = [p.topic for p in passages]
        if self.llm is not None and passages:
            try:
                result = self.llm.complete(
                    _SYSTEM, self._prompt(question, passages), max_tokens=self.max_tokens
                )
                text = result.text.strip()
                if text:
                    return Answer(text=text, citations=citations, source="llm")
            except LLMError:
                pass
        # Fallback: the best retrieved passage's answer.
        if passages:
            best = passages[0]
            return Answer(text=best.answer, citations=[best.topic], source="retrieval")
        return Answer(text="I don't have that information.", citations=[], source="retrieval")
