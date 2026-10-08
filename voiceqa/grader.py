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


# Offline relevance is satisfied when EITHER of two complementary signals is
# clearly met (and escalates when both are weak). The bias is deliberate: a false
# escalation (a human looks at it) is cheap, a false "confident" (a wrong answer
# shipped as correct) is not. The reliable relevance judgement is the LLM grader;
# this is its offline fallback.
#
# 1. Lexical coverage: the fraction of the question's content tokens the retrieved
#    context covers. One shared word is not enough (that let a billing-cycle answer
#    pass as relevant to a plan-change question); a real overlap is.
_MIN_RELEVANCE_OVERLAP = 0.15
# 2. Retrieval confidence: the best passage's cosine similarity to the question.
#    This is a SUFFICIENT high-confidence signal, not a floor every match clears.
#    On this corpus off-topic questions top out below ~0.18 ("capital of France"
#    scores 0.17) while genuine matches run from ~0.24 to ~0.45; 0.30 sits well
#    above any off-topic score, so clearing it means retrieval matched strongly
#    and the answer is relevant even when the surface words differ. Genuine matches
#    below 0.30 are the common case and are caught by the coverage signal instead —
#    the two signals are complementary, which is the point of the OR.
_MIN_RETRIEVAL_SCORE = 0.30


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
        # Relevance from two complementary signals, so the offline grader is
        # confident when EITHER is clearly satisfied and escalates when both are
        # weak (the safe failure for a support bot — never a confident wrong answer):
        #   1. coverage: how much of the question the retrieved context covers
        #      (relative to the question length); and
        #   2. retrieval score: the best passage's cosine similarity to the
        #      question. A strongly similar retrieval (above any off-topic score)
        #      is relevant even if the exact words differ; this is a sufficient
        #      extra path to relevant, not a floor every genuine match clears.
        # A single shared word no longer suffices (that marked a billing-cycle
        # answer relevant to a plan-change question); but a real lexical overlap or
        # a strong retrieval does. Pure synonymy with no shared words and a low
        # score ("end my agreement" vs "cancel contract") is genuinely beyond a
        # lexical grader, so it conservatively escalates — the LLM grader resolves it.
        question_tokens = _tokens(question)
        coverage = len(question_tokens & context_tokens) / max(1, len(question_tokens))
        top_score = max((p.score for p in passages), default=0.0)
        relevant = coverage >= _MIN_RELEVANCE_OVERLAP or top_score >= _MIN_RETRIEVAL_SCORE
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
