# ADR 0002: Pluggable transcription behind one interface

## Status

Accepted.

## Context

The pipeline starts from speech, but speech-to-text is a fast-moving, swappable
component: a hosted Whisper endpoint, a different cloud ASR, a local model, or an
upstream system that already produced a transcript. We also need the pipeline to
run in tests and demos with no audio and no network.

If the pipeline called a specific ASR API directly, it would be hard to test,
hard to run offline, and locked to one engine.

## Decision

Define a small `Transcriber` protocol — `transcribe(audio_path) -> Transcript` —
and provide two implementations:

- **`WhisperTranscriber`** posts an audio file to an OpenAI-compatible
  `/audio/transcriptions` endpoint.
- **`ProvidedTranscript`** treats its input as already-recognised text, for tests,
  demos, and pipelines fed by an upstream ASR.

The pipeline depends only on the protocol. Any failure becomes a
`TranscriptionError`.

## Consequences

- **Testable and offline.** `ProvidedTranscript` lets the whole pipeline run with
  no audio and no network, which is how the tests and the demo work.
- **Engine-independent.** Switching ASR providers is implementing one protocol,
  not touching the pipeline.
- **Composable.** A deployment whose audio is transcribed elsewhere feeds the
  transcript straight in.
- **Trade-off.** The protocol is minimal (a path in, text out); it does not model
  word timings, confidences, or diarisation. Those can be added to `Transcript`
  later without changing the pipeline's shape, and keeping it minimal now keeps
  every implementation simple.
