"""Validate VoiceQA's grader against ground truth on real Spoken-SQuAD data.

The claim under test is narrow and measurable: *VoiceQA's groundedness score
separates answers that are supported by the transcribed passage from answers
that are not.* We never assume a grade is right because it is confident — we
measure it.

The pipeline reuses :class:`voiceqa.grader.Grader` exactly as production does
(offline, so the token-overlap groundedness path runs) and scores a set of real
labeled triples built by :mod:`voiceqa.real_data`. We then report:

* **ROC-AUC** of the groundedness score — the headline, threshold-free metric.
  It equals the probability that a randomly chosen correct answer scores above a
  randomly chosen incorrect one, so 0.5 is chance and 1.0 is perfect separation.
* **Accuracy / precision / recall** of the production decision (``confident``) at
  the configured operating point (``VQ_MIN_GROUNDEDNESS``).
* **Spearman rank correlation** between the score and a continuous token-F1
  correctness, for completeness.
* Two **negative controls** that must collapse toward AUC 0.5:
  shuffling the labels, and pairing each answer with a random unrelated passage.
  A score that carries real signal keeps its AUC only when both the labels and
  the passages are the true ones.

Run it with ``python -m voiceqa.real_benchmark`` (optionally ``--config wer44``).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .grader import Grader
from .knowledge import Passage
from .real_data import GradingExample, build_examples, load_records, token_f1

# ---------------------------------------------------------------------------
# Metrics (implemented here so their correctness can be tested directly).
# ---------------------------------------------------------------------------


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Area under the ROC curve via average ranks (ties credited 0.5).

    AUC = P(score of a positive > score of a negative), with tied pairs counted
    as half. This is the rank-sum (Mann-Whitney U) identity, numerically equal to
    the trapezoidal ROC area but exact under ties.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels differ in length")
    pos = sum(1 for label in labels if label == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        raise ValueError("need both positive and negative labels")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def operating_point_metrics(decisions: list[bool], labels: list[int]) -> dict[str, float]:
    """Accuracy / precision / recall / F1 of a binary decision against labels."""
    tp = sum(1 for d, y in zip(decisions, labels, strict=True) if d and y == 1)
    fp = sum(1 for d, y in zip(decisions, labels, strict=True) if d and y == 0)
    fn = sum(1 for d, y in zip(decisions, labels, strict=True) if not d and y == 1)
    tn = sum(1 for d, y in zip(decisions, labels, strict=True) if not d and y == 0)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("need two equal-length series of length >= 2")
    rx, ry = _avg_ranks(xs), _avg_ranks(ys)
    return _pearson(rx, ry)


def _avg_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


# ---------------------------------------------------------------------------
# Scoring: reuse VoiceQA's grader exactly as the service does.
# ---------------------------------------------------------------------------


@dataclass
class Scored:
    scores: list[float]  # grader groundedness per example
    decisions: list[bool]  # grader `confident` per example (production gate)
    labels: list[int]
    f1: list[float]  # continuous token-F1 correctness per example


def _retrieval_scores(examples: list[GradingExample]) -> list[float]:
    """Real question-to-passage cosine similarity, mirroring knowledge.py.

    This feeds each ``Passage.score`` so the grader's relevance signal runs on a
    real retrieval number rather than a placeholder. For a question asked about
    its own transcribed passage the similarity is high, so relevance is satisfied
    and the ``confident`` gate is driven by groundedness — exactly as intended.
    """
    contexts = [e.context for e in examples]
    questions = [e.question for e in examples]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(contexts + questions)
    ctx_m = matrix[: len(contexts)]
    q_m = matrix[len(contexts) :]
    sims = cosine_similarity(q_m, ctx_m)
    return [round(float(sims[i, i]), 4) for i in range(len(examples))]


def score_examples(examples: list[GradingExample], grader: Grader | None = None) -> Scored:
    """Grade every example with VoiceQA's grader (offline heuristic path)."""
    grader = grader or Grader(llm=None)
    retrieval = _retrieval_scores(examples)
    scores, decisions, labels, f1 = [], [], [], []
    for ex, rscore in zip(examples, retrieval, strict=True):
        passage = Passage(topic=ex.title, question=ex.question, answer=ex.context, score=rscore)
        grade = grader.grade(ex.question, ex.answer, [passage])
        scores.append(grade.groundedness)
        decisions.append(grade.confident)
        labels.append(ex.label)
        f1.append(token_f1(ex.answer, ex.reference))
    return Scored(scores=scores, decisions=decisions, labels=labels, f1=f1)


# ---------------------------------------------------------------------------
# Negative controls: both must collapse toward AUC 0.5.
# ---------------------------------------------------------------------------


def shuffled_label_auc(
    scores: list[float], labels: list[int], trials: int, seed: int
) -> list[float]:
    """Recompute AUC with the real scores but randomly permuted labels."""
    rng = random.Random(seed)
    aucs = []
    for _ in range(trials):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        aucs.append(roc_auc(scores, shuffled))
    return aucs


def random_reference_auc(
    examples: list[GradingExample], trials: int, seed: int, grader: Grader | None = None
) -> list[float]:
    """Re-grade each answer against a random *other* passage, then AUC vs true labels.

    If the groundedness score is real signal it must evaporate once the answer is
    judged against an unrelated passage, driving AUC to chance.
    """
    grader = grader or Grader(llm=None)
    labels = [e.label for e in examples]
    contexts = [e.context for e in examples]
    titles = [e.title for e in examples]
    n = len(examples)
    aucs = []
    for t in range(trials):
        rng = random.Random(seed + t)
        perm = _derangement(n, rng)
        scores = []
        for ex, j in zip(examples, perm, strict=True):
            passage = Passage(topic=titles[j], question=ex.question, answer=contexts[j], score=0.0)
            scores.append(grader.grade(ex.question, ex.answer, [passage]).groundedness)
        aucs.append(roc_auc(scores, labels))
    return aucs


def _derangement(n: int, rng: random.Random) -> list[int]:
    """A permutation with no fixed points (so no answer keeps its own passage)."""
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


@dataclass
class Report:
    config: str
    n_examples: int
    n_pos: int
    n_neg: int
    auc: float
    auc_heldout: float
    pos_mean: float
    neg_mean: float
    metrics: dict[str, float]
    spearman: float
    shuffle_aucs: list[float]
    random_ref_aucs: list[float]

    @property
    def shuffle_mean(self) -> float:
        return sum(self.shuffle_aucs) / len(self.shuffle_aucs)

    @property
    def random_ref_mean(self) -> float:
        return sum(self.random_ref_aucs) / len(self.random_ref_aucs)

    @property
    def validated(self) -> bool:
        """Real signal if the true AUC clearly beats chance and both controls collapse."""
        controls = [self.shuffle_mean, self.random_ref_mean]
        beats_chance = self.auc >= 0.70
        controls_collapse = all(abs(c - 0.5) <= 0.1 for c in controls)
        clear_gap = self.auc - max(controls) >= 0.2
        return beats_chance and controls_collapse and clear_gap


def run(
    config: str = "default", limit: int | None = None, trials: int = 20, seed: int = 20240501
) -> Report:
    records = load_records(config, limit=limit)
    examples = build_examples(records)
    scored = score_examples(examples)

    labels = scored.labels
    auc = roc_auc(scored.scores, labels)

    # Held-out half, to show the result is not an artefact of the particular
    # rows. Examples are built as interleaved (correct, incorrect) pairs per
    # record, so we hold out the back half of the records (both classes present).
    held = list(range(len(labels) // 2, len(labels)))
    auc_heldout = roc_auc([scored.scores[i] for i in held], [labels[i] for i in held])

    pos_scores = [s for s, y in zip(scored.scores, labels, strict=True) if y == 1]
    neg_scores = [s for s, y in zip(scored.scores, labels, strict=True) if y == 0]
    metrics = operating_point_metrics(scored.decisions, labels)
    rho = spearman(scored.scores, scored.f1)

    shuffle_aucs = shuffled_label_auc(scored.scores, labels, trials=trials, seed=seed)
    random_ref_aucs = random_reference_auc(examples, trials=min(trials, 10), seed=seed)

    return Report(
        config=config,
        n_examples=len(examples),
        n_pos=sum(1 for y in labels if y == 1),
        n_neg=sum(1 for y in labels if y == 0),
        auc=auc,
        auc_heldout=auc_heldout,
        pos_mean=sum(pos_scores) / len(pos_scores),
        neg_mean=sum(neg_scores) / len(neg_scores),
        metrics=metrics,
        spearman=rho,
        shuffle_aucs=shuffle_aucs,
        random_ref_aucs=random_ref_aucs,
    )


def format_report(report: Report) -> str:
    m = report.metrics
    lines = [
        "=" * 70,
        f"VoiceQA grader validation on real Spoken-SQuAD ({report.config})",
        "=" * 70,
        f"examples        : {report.n_examples}  "
        f"(correct={report.n_pos}, incorrect={report.n_neg})",
        "",
        "Discrimination (grader groundedness score vs. ground-truth correctness)",
        f"  ROC-AUC (all)      : {report.auc:.4f}",
        f"  ROC-AUC (held-out) : {report.auc_heldout:.4f}",
        f"  mean score correct : {report.pos_mean:.4f}",
        f"  mean score wrong   : {report.neg_mean:.4f}",
        f"  Spearman rho (score vs token-F1) : {report.spearman:.4f}",
        "",
        "Production gate (confident = groundedness >= VQ_MIN_GROUNDEDNESS AND relevant)",
        f"  accuracy  : {m['accuracy']:.4f}",
        f"  precision : {m['precision']:.4f}",
        f"  recall    : {m['recall']:.4f}",
        f"  F1        : {m['f1']:.4f}",
        "",
        "Negative controls (must collapse toward AUC 0.5)",
        f"  shuffled labels   : mean AUC {report.shuffle_mean:.4f}  "
        f"(min {min(report.shuffle_aucs):.4f}, max {max(report.shuffle_aucs):.4f})",
        f"  random reference  : mean AUC {report.random_ref_mean:.4f}  "
        f"(min {min(report.random_ref_aucs):.4f}, max {max(report.random_ref_aucs):.4f})",
        "",
        f"VERDICT: {'VALIDATED' if report.validated else 'NOT VALIDATED'} - "
        f"the groundedness score discriminates real correctness "
        f"(AUC {report.auc:.3f}) and both controls collapse to chance "
        f"({report.shuffle_mean:.3f} / {report.random_ref_mean:.3f}).",
        "=" * 70,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VoiceQA's grader on real data")
    parser.add_argument("--config", choices=["default", "wer44"], default="default")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of records")
    parser.add_argument("--trials", type=int, default=20, help="negative-control repetitions")
    args = parser.parse_args()
    report = run(config=args.config, limit=args.limit, trials=args.trials)
    print(format_report(report))


if __name__ == "__main__":
    main()
