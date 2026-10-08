from __future__ import annotations

import pytest

from voiceqa.llm.base import LLMError
from voiceqa.llm.cascade import CascadeLLM
from voiceqa.llm.factory import build_llm
from voiceqa.llm.fake import FailingLLM, FakeLLM


def test_cascade_falls_back_to_next_provider():
    cascade = CascadeLLM([FailingLLM("a"), FakeLLM("recovered")])
    assert cascade.complete("s", "p").text == "recovered"


def test_cascade_raises_when_all_fail():
    with pytest.raises(LLMError):
        CascadeLLM([FailingLLM("a"), FailingLLM("b")]).complete("s", "p")


def test_factory_builds_cascade_and_single():
    assert build_llm("anthropic,openai").model.startswith("cascade[")
    assert build_llm("ollama").name == "ollama"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        build_llm("nope")
