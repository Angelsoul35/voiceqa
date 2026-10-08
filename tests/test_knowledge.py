from __future__ import annotations

from voiceqa.knowledge import KnowledgeBase


def test_retrieve_returns_k_passages_sorted():
    kb = KnowledgeBase()
    passages = kb.retrieve("what happens if I pay late?", k=3)
    assert len(passages) == 3
    scores = [p.score for p in passages]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_finds_the_relevant_topic():
    kb = KnowledgeBase()
    assert kb.retrieve("how do I replace a lost sim card?", k=1)[0].topic == "sim_replacement"
    assert kb.retrieve("enable roaming when travelling abroad", k=1)[0].topic == "data_roaming"


def test_context_text_includes_topic_and_answer():
    kb = KnowledgeBase()
    p = kb.retrieve("late payment fee", k=1)[0]
    assert p.topic in p.context_text
    assert p.answer in p.context_text
