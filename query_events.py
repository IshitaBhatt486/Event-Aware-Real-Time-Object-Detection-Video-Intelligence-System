"""Ask questions about the saved vision event history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from event_query import DeterministicResponder, EventRetriever, GroqQueryResponder, JsonlEventStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", type=Path, default=Path("events.jsonl"),
        help="Path to the JSONL event history (default: events.jsonl)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Use deterministic answers without contacting Gemini",
    )
    args = parser.parse_args()
    retriever = EventRetriever(JsonlEventStore(args.events))
    if args.no_llm:
        responder = DeterministicResponder()
    else:
        try:
            responder = GroqQueryResponder()
        except RuntimeError as error:
            print(f"LLM disabled: {error}", file=sys.stderr)
            responder = DeterministicResponder()
    llm_available = isinstance(responder, GroqQueryResponder)

    print("Vision event query. Type 'exit' to quit.")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"q", "exit", "quit"}:
            break
        if not question:
            continue
        result = retriever.retrieve(question)
        if not result.events and not result.objects and not result.relationships:
            print("AI: The available event history is insufficient to answer that question.")
            continue
        deterministic = DeterministicResponder()
        lowered = question.lower()
        deterministic_query = "how many" in lowered or any(
            term in lowered
            for term in ("left of", "right of", "above", "below", "near", "far", "did ", "picked up", "put down")
        )
        if deterministic_query or not llm_available:
            answer = deterministic.answer(question, result.as_context())
        else:
            try:
                answer = responder.answer(question, result.as_context())
            except Exception as error:
                print(f"LLM unavailable; disabling it for this session: {error}", file=sys.stderr)
                llm_available = False
                answer = deterministic.answer(question, result.as_context())
        print(f"AI: {answer}")


if __name__ == "__main__":
    main()
