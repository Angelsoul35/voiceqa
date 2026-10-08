# ADR 0001: Ground the answer, then grade it

## Status

Accepted.

## Context

The characteristic failure of a voice assistant or support bot is a **confident
wrong answer**: it responds fluently to a question it has no basis to answer, and
a caller trusts it. A bigger or better model reduces this but does not remove it,
and it gives no signal about *when* to doubt an answer.

## Decision

Build the pipeline so an answer must be both **grounded** and **checked**:

- **Ground.** The answerer may use only the retrieved knowledge-base passages. It
  is instructed to say "I don't have that information" when the context does not
  cover the question, and it returns the passages it used as citations.
- **Grade.** A separate step scores the answer's groundedness (supported by the
  context?) and relevance (addresses the question?). An answer that is not both
  is marked low-confidence, and the pipeline recommends escalation to a human.

Answering and grading are **separate steps**: the component that writes the
answer is not the one that judges it.

## Consequences

- **Confident wrong answers are caught.** An unsupported or off-topic answer is
  flagged rather than served.
- **Escalation is a first-class outcome.** "I'm not sure — a human should take
  this" is an explicit result, not a failure.
- **Traceable.** Every answer cites the passages it used, so a reviewer can see
  its basis.
- **Trade-off.** Grounding limits the assistant to what is in the knowledge base;
  it will not freelance an answer from the model's general knowledge. For a
  support assistant that is the point — correctness and traceability beat
  coverage. The knowledge base is the place to extend coverage.
