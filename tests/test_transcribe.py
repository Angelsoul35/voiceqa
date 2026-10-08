from __future__ import annotations

import pytest

from voiceqa.transcribe import ProvidedTranscript, TranscriptionError, WhisperTranscriber


def test_provided_transcript_returns_text():
    t = ProvidedTranscript().transcribe("  when is my bill issued?  ")
    assert t.text == "when is my bill issued?"
    assert t.engine == "provided"


def test_provided_transcript_rejects_empty():
    with pytest.raises(TranscriptionError):
        ProvidedTranscript().transcribe("   ")


def test_whisper_without_key_raises():
    with pytest.raises(TranscriptionError):
        WhisperTranscriber(api_key="").transcribe("/tmp/does-not-matter.wav")
