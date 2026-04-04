"""
Telemetry — Observability and tracing (Python)

This sample demonstrates how to add observability to agents using
OpenTelemetry-style tracing patterns. It mirrors the Julia vignette
12_telemetry.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_framework.ollama import OllamaChatClient


# ── Telemetry Span ───────────────────────────────────────────────────────────


@dataclass
class TelemetrySpan:
    """Represents a span/activity in a trace, following OpenTelemetry semantics."""

    name: str
    kind: str = "internal"
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    status: str = "unset"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Append a timestamped event to the span."""
        self.events.append({
            "name": name,
            "time": datetime.now(timezone.utc),
            "attributes": attributes or {},
        })

    def finish(self, status: str = "ok") -> None:
        """Mark the span as finished."""
        self.end_time = datetime.now(timezone.utc)
        self.status = status

    @property
    def duration_ms(self) -> int | None:
        """Span duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return int(delta.total_seconds() * 1000)


# ── GenAI Semantic Conventions ───────────────────────────────────────────────

# Following OpenTelemetry Semantic Conventions for Generative AI
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"


# ── In-Memory Telemetry Backend ──────────────────────────────────────────────


class InMemoryTelemetryBackend:
    """Stores spans in memory for testing and debugging."""

    def __init__(self) -> None:
        self.spans: list[TelemetrySpan] = []

    def record_span(self, span: TelemetrySpan) -> None:
        self.spans.append(span)

    def get_spans(self) -> list[TelemetrySpan]:
        return list(self.spans)

    def clear(self) -> None:
        self.spans.clear()


# ── Instrumented Agent Wrapper ───────────────────────────────────────────────


class InstrumentedAgent:
    """Wraps an agent to emit telemetry spans on each call."""

    def __init__(self, agent: Any, backend: InMemoryTelemetryBackend, name: str = "agent"):
        self.agent = agent
        self.backend = backend
        self.name = name

    async def run(self, prompt: str) -> str:
        """Run the agent with telemetry instrumentation."""
        # Agent-level span
        agent_span = TelemetrySpan(name="agent.run", kind="internal")
        agent_span.attributes[GEN_AI_AGENT_NAME] = self.name
        agent_span.attributes["message_count"] = 1

        # Chat-level span
        chat_span = TelemetrySpan(name="chat.completion", kind="client")
        chat_span.attributes[GEN_AI_REQUEST_MODEL] = "qwen3:8b"

        try:
            result = await self.agent.run(prompt)
            result_text = str(result)

            chat_span.finish(status="ok")
            self.backend.record_span(chat_span)

            agent_span.attributes[GEN_AI_RESPONSE_MODEL] = "qwen3:8b"
            agent_span.finish(status="ok")
            self.backend.record_span(agent_span)

            return result_text

        except Exception as e:
            chat_span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            chat_span.finish(status="error")
            self.backend.record_span(chat_span)

            agent_span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            agent_span.finish(status="error")
            self.backend.record_span(agent_span)
            raise


async def main() -> None:
    # ── 1. Span Basics ───────────────────────────────────────────────────
    print("=== Span Basics ===")
    span = TelemetrySpan(name="my_operation", kind="client")
    span.attributes["custom.key"] = "some_value"
    span.add_event("checkpoint_reached", {"step": 1})
    span.add_event("data_processed", {"records": 42})
    time.sleep(0.01)
    span.finish(status="ok")
    print(f"  Span: {span.name}, duration: {span.duration_ms}ms, "
          f"events: {len(span.events)}, status: {span.status}")

    # ── 2. In-Memory Backend ─────────────────────────────────────────────
    print("\n=== In-Memory Backend ===")
    backend = InMemoryTelemetryBackend()
    backend.record_span(span)
    print(f"  Recorded spans: {len(backend.get_spans())}")
    backend.clear()
    print(f"  After clear: {len(backend.get_spans())}")

    # ── 3. Instrumented Agent ────────────────────────────────────────────
    print("\n=== Instrumented Agent ===")
    client = OllamaChatClient(host="http://localhost:11434", model_id="qwen3:8b")
    agent = client.as_agent(
        name="TracedAgent",
        instructions="You are a helpful assistant. Keep answers brief.",
    )

    backend = InMemoryTelemetryBackend()
    traced = InstrumentedAgent(agent, backend, name="TracedAgent")

    result = await traced.run("What is 2 + 2?")
    print(f"  Answer: {result}")

    print(f"\n  Total spans: {len(backend.get_spans())}")
    for s in backend.get_spans():
        model = s.attributes.get(GEN_AI_REQUEST_MODEL, "—")
        print(f"    [{s.kind}] {s.name} — {s.duration_ms}ms, "
              f"model={model}, status={s.status}")


if __name__ == "__main__":
    asyncio.run(main())
