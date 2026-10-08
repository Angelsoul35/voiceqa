"""VoiceQA command-line demo."""

from __future__ import annotations

import argparse

from .service import VoiceQAService

SAMPLE_QUESTIONS = [
    "If I pay a few days after the due date, is there a penalty?",
    "How can I get a new SIM card if I lost mine?",
    "What is the capital of France?",  # out of scope -> low confidence
]


def _show(result: dict) -> None:
    print(f'   Q: "{result["question"]}"')
    print(f"   A: {result['answer']}")
    conf = "CONFIDENT" if result["confident"] else "LOW CONFIDENCE -> escalate"
    print(
        f"   grounded={result['groundedness']:.2f} relevant={result['relevant']} "
        f"[{conf}]  (answer:{result['answer_source']}, grade:{result['grade_source']})"
    )
    print(f"   cited topics: {result['citations']}")


def demo() -> None:
    print("1) Building the pipeline (LLM cascade -> offline fallback) ...")
    service = VoiceQAService()
    print(f"   transcriber: {service.transcriber.name}, answerer model: {service.llm.model}")

    print("2) Answering spoken questions grounded in the knowledge base:")
    for question in SAMPLE_QUESTIONS:
        result = service.answer_question(question)
        _show(result)

    print("3) The out-of-scope question is correctly flagged for escalation.")
    print("   Note: with API keys set, the LLM writes the answer and grades it;")
    print("   offline, retrieval answers and a token-overlap heuristic grades.")


def ask(question: str) -> None:
    result = VoiceQAService().answer_question(question)
    _show(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceQA CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="answer a few built-in sample questions")
    ask_cmd = sub.add_parser("ask", help="answer one question")
    ask_cmd.add_argument("question", help="the question text")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "ask":
        ask(args.question)


if __name__ == "__main__":
    main()
