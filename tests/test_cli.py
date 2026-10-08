from __future__ import annotations

from voiceqa import cli


def test_demo_runs(capsys):
    cli.demo()
    out = capsys.readouterr().out
    assert "Answering spoken questions" in out
    assert "escalate" in out.lower()


def test_ask_prints_answer(capsys):
    cli.ask("what happens if I pay late?")
    out = capsys.readouterr().out
    assert "fee" in out.lower()


def test_main_dispatches_demo(monkeypatch):
    called = {"demo": False}
    monkeypatch.setattr(cli, "demo", lambda: called.__setitem__("demo", True))
    monkeypatch.setattr("sys.argv", ["voiceqa", "demo"])
    cli.main()
    assert called["demo"] is True
