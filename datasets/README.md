# Real evaluation data — Spoken-SQuAD

This directory vendors a small, deterministic slice of a **real spoken
question-answering dataset**, used to validate VoiceQA's grader against ground
truth (see [`../voiceqa/real_benchmark.py`](../voiceqa/real_benchmark.py)). No
synthetic data is used for the validation headline.

## Source

| | |
| --- | --- |
| Hugging Face dataset | [`alinet/spoken_squad`](https://huggingface.co/datasets/alinet/spoken_squad) |
| Upstream task | Spoken-SQuAD — Li *et al.*, 2018 ([arXiv:1804.00320](https://arxiv.org/abs/1804.00320)) |
| Original text QA | SQuAD 1.1 — Rajpurkar *et al.*, 2016 (CC BY-SA 4.0) |
| Rows API | `https://datasets-server.huggingface.co/rows` |

Spoken-SQuAD takes SQuAD 1.1 reading-comprehension passages, synthesises them to
speech, and transcribes them back with an ASR system. The `context` field is
therefore a genuine **transcribed spoken passage** carrying real recognition
errors (e.g. *"in 1858"* surfaces as *"eighteen fifty eight"*; *"upraised"*
becomes *"appraised"*). That is exactly the input VoiceQA is built for: an answer
graded against a passage recovered from speech.

### Snapshots

| File | HF config / split | Word error rate | Rows |
| --- | --- | --- | --- |
| `spoken_squad/spoken_squad_default.json` | `default` / `train` | ~22% (lower-noise ASR) | 380 |
| `spoken_squad/spoken_squad_wer44.json` | `WER44` / `test` | ~44% (stress ASR) | 380 |

Both are the first 380 rows of their split, taken in dataset order so the
selection is reproducible and unbiased. Two noise levels are kept on purpose: the
`default` slice shows the grader on clean-ish transcripts, and `WER44` shows how
its discrimination degrades as ASR error rises.

## License and attribution

The Hugging Face card lists the mirror's license as *unknown*. The underlying
**SQuAD 1.1** text is released under **CC BY-SA 4.0** by its authors; Spoken-SQuAD
is a derived research artifact distributed for academic use. Only a small slice is
vendored here, for reproducible offline evaluation, with attribution to both
upstream works above. If you redistribute this slice, keep this attribution and
honour the upstream terms.

## Schema

Each snapshot is a JSON object:

```json
{
  "dataset": "alinet/spoken_squad",
  "config": "default",
  "split": "train",
  "count": 380,
  "records": [
    {
      "id": "5733be284776f4190066117f",
      "title": "University_of_Notre_Dame",
      "question": "What is in front of the Notre Dame Main Building?",
      "context": "architecturally the school has a catholic character. atop the main building ...",
      "answer": "a copper statue of christ"
    }
  ]
}
```

## How the grading triples are derived

[`voiceqa/real_data.py`](../voiceqa/real_data.py) turns each record into labeled
triples for the grader:

- **Correct (`label = 1`):** `(question, gold answer, own transcribed passage)`.
  The answer belongs to this passage, so a working grounding check should score
  it high.
- **Incorrect (`label = 0`):** `(question, foreign gold answer, own transcribed
  passage)`, where the foreign answer is the gold answer of record `(i + 37) mod
  n` — a real answer that does not belong to this passage. A record whose foreign
  answer happens to equal its own gold answer contributes no negative, so no pair
  is ever mislabeled.

The grader's **groundedness** score is then the quantity under test, and the
label is the ground truth it is measured against.

## Regenerating the snapshots

The loader exposes a live fetch with retry/backoff behind a monkeypatchable
`urlopen` seam:

```python
from voiceqa import real_data as rd

rd.fetch_and_vendor("default", limit=380)  # rewrites spoken_squad_default.json
rd.fetch_and_vendor("wer44", limit=380)    # rewrites spoken_squad_wer44.json
```

This hits the Hugging Face rows API live and overwrites the snapshot in place.
The vendored copies let CI and the test suite run with no network.
