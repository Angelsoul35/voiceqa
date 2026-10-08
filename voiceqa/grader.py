"""Grade an answer for groundedness and relevance.

Grading is the quality gate of the pipeline: it decides whether an answer is
supported by the retrieved context (groundedness) and actually addresses the
question (relevance). An LLM does this well and explains itself; when it is
unavailable, a token-overlap heuristic gives a conservative estimate. A
low-confidence answer is flagged for escalation to a human.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .knowledge import Passage
from .llm.base import LLMClient, LLMError

_SYSTEM = (
    "You are a strict answer grader. Given a question, a proposed answer, and the "
    "context it should be based on, judge how well the answer is supported by the "
    "context and whether it addresses the question."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an of to for and or is are do does how what when where your you i my on in "
    "with this that it as be can will shall".split()
)


# Minimum fraction of the question's content tokens that the retrieved context
# must cover for the offline heuristic to call an answer relevant. Below this,
# the grader cannot tell a topically-adjacent passage from an answering one, so
# it conservatively declines to be confident and the answer is escalated. This
# bias is deliberate: a false escalation (a human looks at it) is cheap, a false
# "confident" (a wrong answer shipped as correct) is not. The reliable relevance
# judgement is the LLM grader; this is its offline fallback.
_MIN_RELEVANCE_OVERLAP = 0.15


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2}


@dataclass
class Grade:
    groundedness: float
    relevant: bool
    confident: bool
    rationale: str
    source: str


class Grader:
    def __init__(
        self, llm: LLMClient | None = None, min_groundedness: float = 0.5, max_tokens: int = 300
    ) -> None:
        self.llm = llm
        self.min_groundedness = min_groundedness
        self.max_tokens = max_tokens

    def _heuristic(self, question: str, answer: str, passages: list[Passage]) -> tuple[float, bool]:
        context_tokens: set[str] = set()
        for p in passages:
            context_tokens |= _tokens(p.context_text)
        answer_tokens = _tokens(answer)
        if not answer_tokens:
            return 0.0, False
        grounded = len(answer_tokens & context_tokens) / len(answer_tokens)
        # Relevance = how much of the question the *retrieved context* covers.
        # Overlapping the answer alone was too permissive: a single shared word
        # (e.g. "month") marked a billing-cycle answer as relevant to a
        # plan-change question. Measuring against the retrieved context, relative
        # to the question length, catches that mismatch.
        question_tokens = _tokens(question)
        coverage = len(question_tokens & context_tokens) / max(1, len(question_tokens))
        relevant = coverage >= _MIN_RELEVANCE_OVERLAP
        return round(grounded, 4), relevant

    def _prompt(self, question: str, answer: str, passages: list[Passage]) -> str:
        context = "\n".join(f"- {p.context_text}" for p in passages)
        return (
            "Respond with ONLY a JSON object: "
            '{"groundedness": <0.0-1.0>, "relevant": <true|false>, '
            '"rationale": "<one short sentence>"}\n\n'
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"
        )

    def grade(self, question: str, answer: str, passages: list[Passage]) -> Grade:
        source = "heuristic"
        grounded, relevant = self._heuristic(question, answer, passages)
        rationale = "token-overlap estimate"

        if self.llm is not None:
            try:
                result = self.llm.complete(
                    _SYSTEM, self._prompt(question, answer, passages), max_tokens=self.max_tokens
                )
                match = _JSON_RE.search(result.text)
                if match:
                    data = json.loads(match.group(0))
                    grounded = max(0.0, min(1.0, float(data["groundedness"])))
                    relevant = bool(data["relevant"])
                    rationale = str(data.get("rationale", "")).strip() or "graded by model"
                    source = "llm"
            except (LLMError, KeyError, ValueError, json.JSONDecodeError):
                pass  # keep heuristic result

        confident = grounded >= self.min_groundedness and relevant
        return Grade(
            groundedness=round(grounded, 4),
            relevant=relevant,
            confident=confident,
            rationale=rationale,
            source=source,
        )
