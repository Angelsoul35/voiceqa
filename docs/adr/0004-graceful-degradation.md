# ADR 0004: Graceful degradation at every stage

## Status

Accepted.

## Context

The pipeline has two external dependencies that can be absent or fail: the
speech-to-text engine and the LLM. A system that only works when both are present
and healthy cannot run in tests or demos, cannot run air-gapped, and falls over
entirely when a provider has an outage or a key is missing.

## Decision

Give every externally-dependent stage a **fallback**, so the pipeline always
returns a useful, honest result:

- **Transcription:** `ProvidedTranscript` stands in when there is no ASR, treating
  input as already-recognised text.
- **Answering:** if the LLM cascade fails, return the single best-retrieved
  passage verbatim — correct, if less fluent.
- **Grading:** if the LLM fails or returns unparseable output, score with a
  token-overlap heuristic for groundedness and relevance.

Every result records which path produced it (`answer_source`, `grade_source`,
`transcription_engine`), so a reader knows whether an LLM or a fallback was used.

## Consequences

- **Always available.** The pipeline runs end to end with no keys and no network,
  which is how CI and the demo run.
- **Degrades, never dies.** A provider outage drops quality (a verbatim passage, a
  coarser grade) instead of returning nothing.
- **Honest about itself.** The `*_source` fields make the degradation visible
  rather than hidden, which matters when a human decides whether to trust a
  result.
- **Trade-off.** The fallbacks are deliberately coarser than the LLM path, so
  quality varies with what is available. That is the right behaviour for a
  resilient system, and the source fields make the trade-off explicit rather than
  silent.
