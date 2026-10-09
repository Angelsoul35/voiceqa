"""Load a real Spoken-SQuAD slice and build labeled grading triples.

Source dataset: ``alinet/spoken_squad`` (Spoken-SQuAD) — SQuAD 1.1 questions over
passages that were synthesised to speech and transcribed back with an ASR system,
so the ``context`` field is a genuinely *spoken/transcribed* passage with real
recognition errors. This is the faithful match for VoiceQA, whose grader checks
whether an answer is supported by a retrieved (here: transcribed) passage.

The module has three layers:

* :func:`fetch_rows` pulls rows from the Hugging Face rows API over HTTP, with
  bounded retries and exponential backoff. The network call goes through the
  module-level :data:`urlopen` seam so tests can drive the retry logic offline.
* :func:`load_records` reads a vendored snapshot under ``datasets/spoken_squad/``
  so the benchmark and tests run with no network.
* :func:`build_examples` turns records into labeled grading triples: a *correct*
  example pairs each question with its own gold answer; an *incorrect* example
  pairs the same question/context with a foreign gold answer taken from another
  item. The grader's job is to tell the two apart.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "alinet/spoken_squad"
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets" / "spoken_squad"

# Each benchmark config maps to a Hugging Face (config, split) and a vendored
# snapshot file. "default" is the lower word-error-rate transcription; "wer44" is
# the 44%-WER stress transcription from the same source, used to show how the
# lexical grader's discrimination degrades as ASR error rises.
CONFIGS: dict[str, dict[str, str]] = {
    "default": {"config": "default", "split": "train", "file": "spoken_squad_default.json"},
    "wer44": {"config": "WER44", "split": "test", "file": "spoken_squad_wer44.json"},
}

# Seam for tests: monkeypatch ``real_data.urlopen`` to exercise the retry path
# without a network. It must accept a URL and ``timeout`` and return an object
# whose ``read()`` yields the JSON response bytes (a context manager, like the
# real ``urllib.request.urlopen``).
urlopen: Callable[..., object] = urllib.request.urlopen

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QARecord:
    """One Spoken-SQuAD item: a transcribed passage, a question, a gold answer."""

    id: str
    title: str
    question: str
    context: str  # ASR-transcribed passage (the spoken side)
    answer: str  # gold reference answer (from the original written SQuAD)


@dataclass(frozen=True)
class GradingExample:
    """A labeled triple handed to VoiceQA's grader.

    ``label`` is 1 when ``answer`` is the gold answer that belongs to this
    (question, context) and 0 when it is a foreign answer from another item.
    ``reference`` is always the gold answer for this item, so a continuous
    token-F1 correctness can be computed independently of the label.
    """

    question: str
    answer: str  # the candidate answer being graded
    context: str  # the passage the answer should be grounded in
    title: str
    label: int
    reference: str


def _rows_url(config: str, split: str, offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    return f"{HF_ROWS_URL}?{query}"


def _parse_rows(payload: bytes) -> list[dict]:
    data = json.loads(payload)
    if "rows" not in data:
        raise ValueError(f"unexpected rows payload: {sorted(data)[:5]}")
    return [entry["row"] for entry in data["rows"]]


def fetch_rows(
    config_key: str = "default",
    limit: int = 400,
    page: int = 100,
    retries: int = 4,
    backoff: float = 0.5,
    timeout: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Fetch up to ``limit`` raw rows from the Hugging Face rows API.

    The API is paginated, so rows are fetched in ``page``-sized chunks. Each
    request is retried up to ``retries`` times with exponential ``backoff``
    (``backoff * 2**attempt`` seconds) on network errors; the final failure is
    re-raised. ``sleep`` and the module-level :data:`urlopen` are injectable so
    the retry logic is testable offline.
    """
    if config_key not in CONFIGS:
        raise KeyError(f"unknown config {config_key!r}; known: {sorted(CONFIGS)}")
    cfg = CONFIGS[config_key]
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        length = min(page, limit - len(rows))
        url = _rows_url(cfg["config"], cfg["split"], offset, length)
        payload = _fetch_once(url, retries=retries, backoff=backoff, timeout=timeout, sleep=sleep)
        batch = _parse_rows(payload)
        if not batch:
            break  # fewer rows available than requested
        rows.extend(batch)
        offset += len(batch)
    return rows[:limit]


def _fetch_once(
    url: str,
    retries: int,
    backoff: float,
    timeout: float,
    sleep: Callable[[float], None],
) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=timeout) as resp:  # type: ignore[operator]
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            last = exc
            if attempt < retries - 1:
                sleep(backoff * (2**attempt))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from last


def _record_from_row(row: dict) -> QARecord:
    answers = row.get("answers", {}).get("text") or [""]
    return QARecord(
        id=str(row.get("id", "")),
        title=str(row.get("title", "")),
        question=str(row["question"]),
        context=str(row["context"]),
        answer=str(answers[0]),
    )


def fetch_and_vendor(config_key: str = "default", limit: int = 400, **kwargs) -> Path:
    """Fetch a live slice and write it as a vendored JSON snapshot. Returns the path."""
    rows = fetch_rows(config_key, limit=limit, **kwargs)
    records = [_record_from_row(r) for r in rows]
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASETS_DIR / CONFIGS[config_key]["file"]
    snapshot = {
        "dataset": DATASET_ID,
        "config": CONFIGS[config_key]["config"],
        "split": CONFIGS[config_key]["split"],
        "count": len(records),
        "records": [vars(r) for r in records],
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_records(config_key: str = "default", limit: int | None = None) -> list[QARecord]:
    """Load records from the vendored snapshot (offline)."""
    if config_key not in CONFIGS:
        raise KeyError(f"unknown config {config_key!r}; known: {sorted(CONFIGS)}")
    path = DATASETS_DIR / CONFIGS[config_key]["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"snapshot {path} is missing; regenerate it with real_data.fetch_and_vendor()"
        )
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    records = [QARecord(**r) for r in snapshot["records"]]
    return records[:limit] if limit is not None else records


def build_examples(records: list[QARecord], shift: int = 37) -> list[GradingExample]:
    """Build correct/incorrect grading triples from records.

    For record ``i`` the correct example uses its own gold answer; the incorrect
    example reuses the same question and transcribed context but swaps in the gold
    answer of record ``(i + shift) % n`` — a real answer that does not belong to
    this context. Items whose foreign answer happens to equal the true answer
    (duplicate answers across questions) contribute only the correct example, so a
    mislabeled pair never enters the set.
    """
    n = len(records)
    if n < 2:
        raise ValueError("need at least two records to build negatives")
    examples: list[GradingExample] = []
    for i, rec in enumerate(records):
        foreign = records[(i + shift) % n].answer
        examples.append(
            GradingExample(
                question=rec.question,
                answer=rec.answer,
                context=rec.context,
                title=rec.title,
                label=1,
                reference=rec.answer,
            )
        )
        if _norm(foreign) and _norm(foreign) != _norm(rec.answer):
            examples.append(
                GradingExample(
                    question=rec.question,
                    answer=foreign,
                    context=rec.context,
                    title=rec.title,
                    label=0,
                    reference=rec.answer,
                )
            )
    return examples


def _norm(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-level F1 between two answer strings."""
    pred = _TOKEN_RE.findall(prediction.lower())
    gold = _TOKEN_RE.findall(reference.lower())
    if not pred or not gold:
        return 1.0 if not pred and not gold else 0.0
    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for t in gold:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if gold_counts.get(t, 0) - common.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)
