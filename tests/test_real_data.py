"""Functional tests for the real Spoken-SQuAD loader and triple builder.

Every test here fails if the behaviour it guards breaks: the loader must return
the real transcribed data in the right shape, the builder must assign correct and
incorrect labels the right way round, and the networked fetch must retry on
failure and give up after exhausting its attempts.
"""

from __future__ import annotations

import json

import pytest

from voiceqa import real_data as rd

# ---------------------------------------------------------------------------
# Loader + builder on the vendored real data.
# ---------------------------------------------------------------------------


def test_load_records_returns_real_shaped_data():
    records = rd.load_records("default", limit=50)
    assert len(records) == 50
    for rec in records:
        assert isinstance(rec, rd.QARecord)
        assert rec.question and rec.context and rec.answer
        assert rec.question.endswith("?") or "?" in rec.question
        # The passage is substantially longer than the answer: it is a paragraph,
        # not a label swapped into the wrong field.
        assert len(rec.context) > len(rec.answer)


def test_gold_answers_are_grounded_in_the_transcript():
    # Real Spoken-SQuAD property: the gold answer is a span of the (transcribed)
    # passage, so for most items the answer's content tokens appear in the
    # context. If the loader mixed up fields or returned synthetic data, this
    # relationship would not hold.
    records = rd.load_records("default", limit=100)
    grounded = 0
    for rec in records:
        ans_tokens = set(rd._TOKEN_RE.findall(rec.answer.lower()))
        ctx_tokens = set(rd._TOKEN_RE.findall(rec.context.lower()))
        if ans_tokens and ans_tokens <= ctx_tokens:
            grounded += 1
    assert grounded >= 80  # the low-WER transcript keeps most answer spans intact


def test_build_examples_labels_and_pairing():
    records = rd.load_records("default", limit=60)
    examples = rd.build_examples(records, shift=37)

    labels = {e.label for e in examples}
    assert labels == {0, 1}
    # One correct example per record; negatives only when the foreign answer
    # genuinely differs.
    assert sum(1 for e in examples if e.label == 1) == len(records)
    assert sum(1 for e in examples if e.label == 0) >= len(records) - 5

    for e in examples:
        if e.label == 1:
            assert e.answer == e.reference  # the gold answer belongs here
        else:
            assert e.answer != e.reference  # a foreign answer


def test_negative_answer_comes_from_the_shifted_record():
    records = rd.load_records("default", limit=60)
    shift = 37
    examples = rd.build_examples(records, shift=shift)
    by_question: dict[str, list[rd.GradingExample]] = {}
    for e in examples:
        by_question.setdefault(e.question, []).append(e)
    n = len(records)
    checked = 0
    for i, rec in enumerate(records):
        foreign = records[(i + shift) % n].answer
        negs = [e for e in by_question[rec.question] if e.label == 0]
        if negs:  # skipped only when foreign == gold
            assert negs[0].answer == foreign
            checked += 1
    assert checked > 0


def test_build_examples_rejects_degenerate_input():
    with pytest.raises(ValueError):
        rd.build_examples([rd.QARecord("1", "t", "q?", "ctx", "a")])


def test_token_f1_matches_squad_semantics():
    assert rd.token_f1("Denver Broncos", "denver broncos") == pytest.approx(1.0)
    assert rd.token_f1("Carolina Panthers", "Denver Broncos") == pytest.approx(0.0)
    # pred=[a,b,c] gold=[a,b,d]: 2 overlap, precision 2/3, recall 2/3 -> F1 2/3.
    assert rd.token_f1("a b c", "a b d") == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Networked fetch: retry/backoff behaviour, driven entirely offline.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._payload


def _one_page(n: int) -> bytes:
    rows = [
        {
            "row": {
                "id": str(i),
                "title": "T",
                "question": f"q{i}?",
                "context": f"context number {i} with words",
                "answers": {"text": [f"answer {i}"], "answer_start": [0]},
            }
        }
        for i in range(n)
    ]
    return json.dumps({"rows": rows, "num_rows_total": n}).encode()


def test_fetch_retries_then_succeeds(monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def flaky_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:  # fail the first two attempts
            raise urllib.error.URLError("transient")
        return _FakeResponse(_one_page(5))

    slept: list[float] = []
    monkeypatch.setattr(rd, "urlopen", flaky_urlopen)

    rows = rd.fetch_rows("default", limit=5, page=5, retries=4, backoff=0.5, sleep=slept.append)

    assert calls["n"] == 3  # two failures + one success
    assert slept == [0.5, 1.0]  # exponential backoff between the three attempts
    assert [r["question"] for r in rows] == [f"q{i}?" for i in range(5)]


def test_fetch_gives_up_after_exhausting_retries(monkeypatch):
    import urllib.error

    calls = {"n": 0}

    def always_fail(url, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("down")

    slept: list[float] = []
    monkeypatch.setattr(rd, "urlopen", always_fail)

    with pytest.raises(RuntimeError, match="after 4 attempts"):
        rd.fetch_rows("default", limit=5, page=5, retries=4, backoff=0.5, sleep=slept.append)

    assert calls["n"] == 4  # all attempts used
    assert slept == [0.5, 1.0, 2.0]  # a sleep between each pair of attempts


def test_fetch_paginates_and_respects_limit(monkeypatch):
    pages = [_one_page(3), _one_page(3)]
    seen: list[int] = []

    def paged_urlopen(url, timeout=None):
        seen.append(len(seen))
        return _FakeResponse(pages[len(seen) - 1])

    monkeypatch.setattr(rd, "urlopen", paged_urlopen)
    rows = rd.fetch_rows("default", limit=6, page=3, retries=2, sleep=lambda _: None)
    assert len(rows) == 6
    assert len(seen) == 2  # two pages of three
