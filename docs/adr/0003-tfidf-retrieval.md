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

Use **TF-IDF with cosine similarity** (scikit-learn). The knowledge base is
vectorised once over question + answer text; a query is vectorised and scored
against it, and the top-k passages are returned with their similarity scores.

## Consequences

- **Explainable.** A similarity score per passage shows why it was retrieved; the
  retrieved topics are cited in the result.
- **No extra infrastructure.** No embedding model to host and no vector database;
  it runs anywhere scikit-learn does, offline.
- **Deterministic.** The same query always retrieves the same passages, which
  makes the pipeline and its tests reproducible.
- **Fast and sufficient.** For a small, distinctive FAQ, lexical overlap is a
  strong signal, and the demo and tests show it retrieving the right topics.
- **Trade-off.** TF-IDF matches words, not meaning, so it can miss paraphrases
  that share no vocabulary. The `retrieve` interface is designed so a dense
  embedding retriever can replace it later without changing the answerer or the
  grader, if the corpus grows to need it.
