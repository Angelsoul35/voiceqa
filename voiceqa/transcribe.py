"""Speech-to-text, behind one small interface.

Transcription is pluggable so the pipeline does not depend on any one speech
engine. ``WhisperTranscriber`` sends audio to an OpenAI-compatible
``/audio/transcriptions`` endpoint; ``ProvidedTranscript`` returns text it is
given, which is what the tests and the offline demo use (the "audio" is already a
transcript). Any failure becomes a :class:`TranscriptionError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


class TranscriptionError(Exception):
    """Raised when speech-to-text fails."""


@dataclass(frozen=True)
class Transcript:
    text: str
    engine: str


@runtime_checkable
class Transcriber(Protocol):
    name: str

    def transcribe(self, audio_path: str) -> Transcript: ...


class ProvidedTranscript:
    """An offline transcriber: the 'audio path' is treated as literal text.

    Useful for tests, demos, and pipelines fed by an upstream ASR system.
    """

    name = "provided"

    def transcribe(self, audio_path: str) -> Transcript:
        text = audio_path.strip()
        if not text:
            raise TranscriptionError("empty transcript")
        return Transcript(text=text, engine=self.name)


class WhisperTranscriber:
    """Send an audio file to an OpenAI-compatible transcription endpoint."""

    name = "whisper"

    def __init__(
        self,
        model: str = "whisper-1",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._url = base_url.rstrip("/") + "/audio/transcriptions"
        self._timeout = timeout

    def transcribe(self, audio_path: str) -> Transcript:
        if not self._key:
            raise TranscriptionError("whisper: OPENAI_API_KEY is not set")
        try:
            with open(audio_path, "rb") as fh:
                resp = httpx.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._key}"},
                    data={"model": self.model},
                    files={"file": (os.path.basename(audio_path), fh)},
                    timeout=self._timeout,
                )
            resp.raise_for_status()
            return Transcript(text=resp.json()["text"], engine=self.name)
        except (OSError, httpx.HTTPError, KeyError, ValueError) as exc:
            raise TranscriptionError(f"whisper: {exc}") from exc
