from __future__ import annotations

from voiceqa.answerer import Answerer
from voiceqa.knowledge import KnowledgeBase
from voiceqa.llm.fake import FailingLLM, FakeLLM


def _passages():
    return KnowledgeBase().retrieve("what happens if I pay late?", k=3)


def test_llm_path_uses_model_answer():
    answerer = Answerer(llm=FakeLLM("A 2% fee applies after a grace period."))
    answer = answerer.answer("late payment?", _passages())
    assert answer.source == "llm"
    assert "2%" in answer.text
    assert answer.citations  # topics cited


def test_fallback_returns_best_passage():
    answerer = Answerer(llm=FailingLLM())
    answer = answerer.answer("what happens if I pay late?", _passages())
    assert answer.source == "retrieval"
    assert "fee" in answer.text.lower()
    assert answer.citations == ["late_payment"]


def test_no_passages_returns_dont_know():
    answerer = Answerer(llm=FailingLLM())
    answer = answerer.answer("anything", [])
    assert "don't have" in answer.text.lower()
    assert answer.citations == []


def test_empty_llm_text_falls_back():
    answerer = Answerer(llm=FakeLLM("   "))
    answer = answerer.answer("late payment?", _passages())
    assert answer.source == "retrieval"
