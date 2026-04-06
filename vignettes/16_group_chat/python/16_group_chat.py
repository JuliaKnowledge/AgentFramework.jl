"""
Group Chat Orchestrations — Python

This sample demonstrates three multi-agent group chat patterns:
  1. Round-robin: agents take turns in fixed order.
  2. Selector-based: an orchestrator agent picks the next speaker.
  3. Magentic: a manager plans, tracks progress, and adapts.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio

from agent_framework.ollama import OllamaChatClient
from agent_framework.orchestrations import (
    GroupChatBuilder,
    GroupChatState,
    MagenticBuilder,
)


async def main() -> None:
    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    # ── Debate participants ──────────────────────────────────────────────

    optimist = client.as_agent(
        name="Optimist",
        instructions=(
            "You are an optimistic philosopher. You see the bright side "
            "of every argument. Keep responses to 2-3 sentences. Build on "
            "what others have said and find the silver lining."
        ),
    )

    pessimist = client.as_agent(
        name="Pessimist",
        instructions=(
            "You are a pessimistic philosopher. You challenge assumptions "
            "and point out risks. Keep responses to 2-3 sentences. "
            "Respectfully counter the previous speaker's points."
        ),
    )

    moderator = client.as_agent(
        name="Moderator",
        instructions=(
            "You are a neutral moderator. Summarise the discussion so far "
            "and ask a probing follow-up question to deepen the debate. "
            "Keep responses to 2-3 sentences."
        ),
    )

    # ── 1. Round-Robin Group Chat ────────────────────────────────────────

    print("=== Round-Robin Group Chat ===\n")

    # Round-robin selection function
    def round_robin(state: GroupChatState) -> str:
        names = list(state.participants.keys())
        return names[state.current_round % len(names)]

    group = GroupChatBuilder(
        participants=[optimist, pessimist, moderator],
        selection_func=round_robin,
        max_rounds=6,
        intermediate_outputs=True,
    )

    workflow = group.build()

    async for event in workflow.run(
        "Is technology making humanity better or worse?", stream=True
    ):
        if hasattr(event, "type") and hasattr(event, "data"):
            data = event.data
            if hasattr(data, "text") and data.text:
                name = getattr(data, "author_name", None) or getattr(event, "executor_id", "?")
                print(f"[{name}] {data.text}\n")

    # ── 2. Selector-Based Group Chat ─────────────────────────────────────

    print("\n=== Selector-Based Group Chat ===\n")

    coordinator = client.as_agent(
        name="Coordinator",
        instructions=(
            "You are a debate coordinator. Given the conversation so far, "
            "decide which participant should speak next. Consider who would "
            "add the most value. Reply with ONLY the name: Optimist, "
            "Pessimist, or Moderator."
        ),
    )

    selector_group = GroupChatBuilder(
        participants=[optimist, pessimist, moderator],
        orchestrator_agent=coordinator,
        max_rounds=8,
        intermediate_outputs=True,
    )

    selector_workflow = selector_group.build()

    async for event in selector_workflow.run(
        "Should artificial intelligence have rights?", stream=True
    ):
        if hasattr(event, "type") and hasattr(event, "data"):
            data = event.data
            if hasattr(data, "text") and data.text:
                name = getattr(data, "author_name", None) or getattr(event, "executor_id", "?")
                print(f"[{name}] {data.text}\n")

    # ── 3. Magentic Orchestration ────────────────────────────────────────

    print("\n=== Magentic Orchestration ===\n")

    researcher = client.as_agent(
        name="Researcher",
        instructions=(
            "You are a philosophical researcher. Provide relevant historical "
            "context, cite key thinkers, and identify core tensions. Keep "
            "responses to 3-4 sentences."
        ),
    )

    analyst = client.as_agent(
        name="Analyst",
        instructions=(
            "You are a philosophical analyst. Evaluate arguments for logical "
            "consistency, identify fallacies, and suggest stronger "
            "formulations. Keep responses to 3-4 sentences."
        ),
    )

    manager_agent = client.as_agent(
        name="Manager",
        instructions=(
            "You are a project manager. Plan and coordinate tasks between "
            "the researcher and analyst. Track progress and decide when the "
            "analysis is complete."
        ),
    )

    magentic = MagenticBuilder(
        participants=[researcher, analyst],
        manager_agent=manager_agent,
        max_stall_count=3,
        max_round_count=10,
        intermediate_outputs=True,
    )

    magentic_workflow = magentic.build()

    last_event = None
    async for event in magentic_workflow.run(
        "Analyse the trolley problem and its implications for autonomous vehicles.",
        stream=True,
    ):
        last_event = event
        if hasattr(event, "type") and hasattr(event, "data"):
            data = event.data
            if hasattr(data, "text") and data.text:
                name = getattr(data, "author_name", None) or getattr(event, "executor_id", "?")
                print(f"[{name}] {data.text}\n")

    # Print final outputs
    if last_event is not None:
        outputs = last_event.get_outputs() if hasattr(last_event, "get_outputs") else []
        for output in outputs:
            print(f"=== Final Answer ===\n{output}\n")


if __name__ == "__main__":
    asyncio.run(main())
