"""A small knowledge base and a TF-IDF retriever.

The retriever grounds answers in a fixed set of FAQ entries. It is a classic
TF-IDF + cosine-similarity search: simple, explainable, and fast, with no model
to train or host. Grounding answers in retrieved text is what lets the grader
check whether an answer is actually supported.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

# A synthetic telecom customer-support knowledge base.
FAQ: list[dict] = [
    {
        "topic": "billing_cycle",
        "question": "When is my bill issued?",
        "answer": "Bills are issued on the first day of each month and cover the previous month.",
    },
    {
        "topic": "late_payment",
        "question": "What happens if I pay late?",
        "answer": "A late payment adds a 2% fee after a 10-day grace period from the due date.",
    },
    {
        "topic": "data_roaming",
        "question": "How do I enable data roaming abroad?",
        "answer": "Enable roaming in the app under Settings, then confirm the daily pass.",
    },
    {
        "topic": "plan_change",
        "question": "Can I change my plan mid-cycle?",
        "answer": "You can change your plan once per cycle; the new price starts next bill.",
    },
    {
        "topic": "cancel_contract",
        "question": "How do I cancel my contract?",
        "answer": "Cancel in writing with 30 days notice; early exit before 12 months has a fee.",
    },
    {
        "topic": "coverage",
        "question": "How do I check network coverage?",
        "answer": "Check coverage by postcode on the coverage map; 5G is in major cities.",
    },
    {
        "topic": "sim_replacement",
        "question": "How do I replace a lost SIM?",
        "answer": "Order a replacement SIM in the app; it keeps your number and ships in 2 days.",
    },
    {
        "topic": "refund",
        "question": "How are refunds handled?",
        "answer": "Approved refunds go back to the original payment method within 5 working days.",
    },
]


@dataclass(frozen=True)
class Passage:
    topic: str
    question: str
    answer: str
    score: float

    @property
    def context_text(self) -> str:
        return f"[{self.topic}] {self.answer}"


class KnowledgeBase:
    def __init__(self, entries: list[dict] | None = None) -> None:
        self.entries = entries if entries is not None else FAQ
        # A FeatureUnion of a word view (1-2 grams: phrasing) and a character view
        # (char_wb 3-5): the character n-grams match morphological variants a bare
        # word model misses ("refund"/"refunds", "cancel"/"cancellation"), so a
        # reworded spoken question still retrieves the right FAQ. Both the FAQ
        # question and its answer are indexed, so the user's phrasing can match the
        # FAQ's phrasing, not only the answer text.
        self._vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
            ]
        )
        corpus = [f"{e['question']} {e['answer']}" for e in self.entries]
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 3) -> list[Passage]:
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        order = sims.argsort()[::-1][:k]
        passages: list[Passage] = []
        for i in order:
            e = self.entries[int(i)]
            passages.append(
                Passage(
                    topic=e["topic"],
                    question=e["question"],
                    answer=e["answer"],
                    score=round(float(sims[int(i)]), 4),
                )
            )
        return passages
