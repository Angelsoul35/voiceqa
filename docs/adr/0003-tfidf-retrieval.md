# ADR 0003: TF-IDF retrieval for grounding

## Status

Accepted.

## Context

Answers must be grounded in a known source, so we need to retrieve the passages
most relevant to a question. The options ranged from dense vector embeddings (a
model that maps text to vectors, plus a vector store) to classic sparse
retrieval (TF-IDF / BM25).

For this knowledge base — a few dozen short, distinctive FAQ entries — the main
goals are explainability, zero external dependencies, and determinism, not
maximum recall over millions of documents.

## Decision

Use **TF-IDF with cosine similarity** (scikit-learn), over a **`FeatureUnion` of
two complementary views** of the text:

- a **word** view (`analyzer="word"`, 1–2 grams) that captures distinctive terms
  and short phrases ("late payment", "data roaming"); and
- a **character** view (`analyzer="char_wb"`, 3–5 grams) that matches *inside*
  words, so morphological variants a word model treats as unrelated tokens
  ("refund"/"refunds", "cancel"/"cancellation", "roaming"/"roam") still align.

The knowledge base is vectorised once over question + answer text; a query is
vectorised through the same union and scored by cosine similarity, and the top-k
passages are returned with their scores.

The character view matters because the questions arrive by **voice**: a spoken
question is reworded, not copied from the FAQ, so a bare word match is brittle.
The character n-grams make retrieval robust to that rewording without a model to
train or host.

## Consequences

- **Explainable.** A similarity score per passage shows why it was retrieved; the
  retrieved topics are cited in the result.
- **No extra infrastructure.** No embedding model to host and no vector database;
  it runs anywhere scikit-learn does, offline.
- **Deterministic.** The same query always retrieves the same passages, which
  makes the pipeline and its tests reproducible.
- **Robust to rewording.** The word + character union retrieves the right topic
  for a question phrased differently from the FAQ, which a word-only model would
  miss — the common case for spoken questions.
- **Trade-off.** TF-IDF still matches surface form, not meaning, so a pure
  synonym that shares neither words nor character runs ("end my agreement" for
  "cancel contract") can still be missed; the grader treats a weak best score as
  "not relevant" and escalates rather than answering confidently. The `retrieve`
  interface is designed so a dense embedding retriever can replace it later
  without changing the answerer or the grader, if the corpus grows to need it.
