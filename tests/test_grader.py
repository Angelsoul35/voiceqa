from __future__ import annotations

from voiceqa.grader import Grader
from voiceqa.knowledge import KnowledgeBase
from voiceqa.llm.fake import FailingLLM, FakeLLM


def _passages(query="what happens if I pay late?"):
    return KnowledgeBase().retrieve(query, k=3)


def test_heuristic_grounded_answer_is_confident():
    passages = _passages()
    grounded_answer = passages[0].answer  # verbatim from context
    grade = Grader(llm=FailingLLM(), min_groundedness=0.5).grade(
        "what happens if I pay late?", grounded_answer, passages
    )
    assert grade.source == "heuristic"
    assert grade.groundedness >= 0.5
    assert grade.relevant is True
    assert grade.confident is True


def test_irrelevant_answer_is_not_confident():
    passages = _passages("what happens if I pay late?")
    # An answer that is grounded in context but does not address the question.
    grade = Grader(llm=FailingLLM(), min_groundedness=0.5).grade(
        "what is the capital of France?", passages[0].answer, passages
    )
    assert grade.relevant is False
    assert grade.confident is False


def test_llm_json_grade_is_used():
    passages = _passages()
    llm = FakeLLM('{"groundedness": 0.9, "relevant": true, "rationale": "supported"}')
    grade = Grader(llm=llm, min_groundedness=0.5).grade("late payment?", "a 2% fee", passages)
    assert grade.source == "llm"
    assert grade.groundedness == 0.9
    assert grade.confident is True


def test_bad_llm_json_falls_back_to_heuristic():
    passages = _passages()
    grade = Grader(llm=FakeLLM("not json"), min_groundedness=0.5).grade(
        "what happens if I pay late?", passages[0].answer, passages
    )
    assert grade.source == "heuristic"


def test_groundedness_is_clamped():
    passages = _passages()
    llm = FakeLLM('{"groundedness": 5, "relevant": true, "rationale": "x"}')
    grade = Grader(llm=llm).grade("late payment?", "a fee", passages)
    assert 0.0 <= grade.groundedness <= 1.0
