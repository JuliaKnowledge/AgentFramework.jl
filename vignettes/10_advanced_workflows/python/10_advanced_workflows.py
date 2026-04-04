"""
Advanced Workflows — Fan-out, fan-in, conditional routing (Python)

This sample demonstrates advanced workflow patterns: parallel fan-out,
fan-in aggregation, conditional routing, shared state, and human-in-the-loop.
It mirrors the Julia vignette 10_advanced_workflows.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio
from dataclasses import dataclass

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    Executor,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.ollama import OllamaChatClient
from typing_extensions import Never


# ── Fan-Out / Fan-In Demo ────────────────────────────────────────────────────


class Dispatcher(Executor):
    """Broadcasts input to all downstream executors (fan-out)."""

    @handler
    async def dispatch(self, msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(msg)


class WorkerA(Executor):
    """Processes input and forwards result."""

    @handler
    async def process(self, msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(f"A processed: {msg}")


class WorkerB(Executor):
    """Processes input and forwards result."""

    @handler
    async def process(self, msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(f"B processed: {msg}")


class Collector(Executor):
    """Collects fan-in results and yields combined output."""

    @handler
    async def collect(self, results: list[str], ctx: WorkflowContext[Never, str]) -> None:
        for r in results:
            await ctx.yield_output(r)


# ── Conditional Routing Demo ─────────────────────────────────────────────────


class Classifier(Executor):
    """Forwards input for conditional routing."""

    @handler
    async def classify(self, msg: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(msg)


class SentimentHandler(Executor):
    """Handles messages routed by sentiment."""

    def __init__(self, id: str, label: str):
        super().__init__(id=id)
        self.label = label

    @handler
    async def handle(self, msg: str, ctx: WorkflowContext[Never, str]) -> None:
        await ctx.yield_output(f"{self.label}: {msg}")


# ── Text Analysis Pipeline ───────────────────────────────────────────────────


@dataclass
class AnalysisResult:
    """Container for aggregated analysis results."""

    sentiment: str = ""
    keywords: str = ""
    summary: str = ""


class TextDispatcher(Executor):
    """Stores original text in state and fans out to analyzers."""

    @handler
    async def dispatch(self, text: str, ctx: WorkflowContext[AgentExecutorRequest]) -> None:
        ctx.set_state("original_text", text)
        msg = Message("user", text=text)
        await ctx.send_message(AgentExecutorRequest(messages=[msg], should_respond=True))


class AnalysisMerger(Executor):
    """Merges results from parallel analyzers."""

    @handler
    async def merge(
        self, results: list[AgentExecutorResponse], ctx: WorkflowContext[Never, str]
    ) -> None:
        parts = []
        for r in results:
            parts.append(f"{r.executor_id}: {r.agent_response.text}")
        original = ctx.get_state("original_text") or ""
        report = f"=== Analysis of: {original[:50]}... ===\n" + "\n".join(parts)
        await ctx.yield_output(report)


async def main() -> None:
    # ── 1. Fan-Out / Fan-In ──────────────────────────────────────────────
    print("=== Fan-Out / Fan-In ===")
    dispatcher = Dispatcher(id="dispatcher")
    worker_a = WorkerA(id="worker_a")
    worker_b = WorkerB(id="worker_b")
    collector = Collector(id="collector")

    fan_workflow = (
        WorkflowBuilder(start_executor=dispatcher)
        .add_fan_out_edges(dispatcher, [worker_a, worker_b])
        .add_fan_in_edges([worker_a, worker_b], collector)
        .build()
    )

    async for event in fan_workflow.run("hello", stream=True):
        if event.type == "output":
            print(f"  Output: {event.data}")

    # ── 2. Conditional Routing ───────────────────────────────────────────
    print("\n=== Conditional Routing ===")
    classifier = Classifier(id="classifier")
    positive = SentimentHandler(id="positive", label="😊 Positive")
    negative = SentimentHandler(id="negative", label="😞 Negative")
    neutral = SentimentHandler(id="neutral", label="😐 Neutral")

    switch_workflow = (
        WorkflowBuilder(start_executor=classifier)
        .add_edge(classifier, positive, condition=lambda d: "good" in d.lower())
        .add_edge(classifier, negative, condition=lambda d: "bad" in d.lower())
        .add_edge(classifier, neutral, condition=lambda d: "good" not in d.lower() and "bad" not in d.lower())
        .build()
    )

    for text in ["This is good news", "Bad weather ahead", "The sky is blue"]:
        async for event in switch_workflow.run(text, stream=True):
            if event.type == "output":
                print(f"  {event.data}")

    # ── 3. Text Analysis Pipeline (requires Ollama) ──────────────────────
    print("\n=== Text Analysis Pipeline ===")
    client = OllamaChatClient(host="http://localhost:11434", model_id="qwen3:8b")

    text_dispatcher = TextDispatcher(id="text_dispatcher")

    sentiment_agent = AgentExecutor(
        client.as_agent(
            name="sentiment",
            instructions="Analyze sentiment. Reply with one word: Positive, Negative, or Neutral.",
        )
    )
    keyword_agent = AgentExecutor(
        client.as_agent(
            name="keywords",
            instructions="Extract 3-5 keywords. Return them comma-separated.",
        )
    )
    summary_agent = AgentExecutor(
        client.as_agent(
            name="summary",
            instructions="Summarize the text in one sentence.",
        )
    )
    merger = AnalysisMerger(id="merger")

    pipeline = (
        WorkflowBuilder(start_executor=text_dispatcher)
        .add_fan_out_edges(text_dispatcher, [sentiment_agent, keyword_agent, summary_agent])
        .add_fan_in_edges([sentiment_agent, keyword_agent, summary_agent], merger)
        .build()
    )

    async for event in pipeline.run(
        "Julia is a high-level, high-performance programming language for "
        "technical computing. It combines the ease of Python with the speed of C.",
        stream=True,
    ):
        if event.type == "output":
            print(event.data)


if __name__ == "__main__":
    asyncio.run(main())
