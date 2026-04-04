"""
Structured Output — Getting typed responses from the LLM (Python)

This sample demonstrates how to request structured JSON output from an
agent using Pydantic models, so you get validated Python objects instead
of free-form text. It mirrors the Julia vignette 05_structured_output.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama pydantic
"""

import asyncio
import json

from pydantic import BaseModel

from agent_framework.ollama import OllamaChatClient


# ── Define response types as Pydantic models ─────────────────────────────

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str
    recommended: bool


class Author(BaseModel):
    name: str
    nationality: str


class BookReview(BaseModel):
    title: str
    author: Author
    rating: int
    themes: list[str]


async def main() -> None:
    # Create a chat client pointing at the local Ollama instance.
    client = OllamaChatClient(
        host="http://localhost:11434",
        model_id="qwen3:8b",
    )

    # Ollama's structured output uses a JSON schema dict as `format`,
    # not a Pydantic class directly.  We embed the schema in the agent
    # instructions so the LLM knows the expected shape and use
    # format=<json-schema> to enforce valid JSON.

    movie_schema = MovieReview.model_json_schema()
    agent = client.as_agent(
        name="MovieCritic",
        instructions=(
            "You are a movie critic. Always return structured reviews as JSON "
            f"matching this schema:\n{json.dumps(movie_schema, indent=2)}"
        ),
    )

    # ── Typed response using response_format ─────────────────────────────
    print("=== Movie Review (structured) ===")
    result = await agent.run(
        "Review the movie 'Inception' (2010).",
        options={"response_format": "json"},
    )
    print(f"Raw response: {result.text}")

    # Parse the JSON response into a Pydantic model.
    review = MovieReview.model_validate_json(result.text)
    print(f"Title: {review.title}")
    print(f"Rating: {review.rating}")
    print(f"Summary: {review.summary}")
    print(f"Recommended: {review.recommended}")
    print()

    # ── Nested types ─────────────────────────────────────────────────────
    print("=== Book Review (nested types) ===")
    book_schema = BookReview.model_json_schema()
    book_agent = client.as_agent(
        name="BookCritic",
        instructions=(
            "You are a book critic. Always return structured reviews as JSON "
            f"matching this schema:\n{json.dumps(book_schema, indent=2)}"
        ),
    )
    result = await book_agent.run(
        "Review '1984' by George Orwell.",
        options={"response_format": "json"},
    )
    book = BookReview.model_validate_json(result.text)
    print(f"Title: {book.title}")
    print(f"Author: {book.author.name} ({book.author.nationality})")
    print(f"Rating: {book.rating}")
    print(f"Themes: {book.themes}")
    print()

    # ── Inspect the generated schema ─────────────────────────────────────
    print("=== Generated JSON Schema ===")
    print(json.dumps(MovieReview.model_json_schema(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
