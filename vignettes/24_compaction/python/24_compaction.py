"""
Message Compaction — Managing context windows (Python)

This sample demonstrates the compaction strategies available in the Python
agent-framework for reducing message history when conversations exceed the
model's context window.  It mirrors the Julia vignette 24_compaction.

Prerequisites:
  - pip install agent-framework
"""

import asyncio

from agent_framework import Message, Content
from agent_framework._compaction import (
    SlidingWindowStrategy,
    SelectiveToolCallCompactionStrategy,
    ToolResultCompactionStrategy,
    SummarizationStrategy,
    TokenBudgetComposedStrategy,
    TruncationStrategy,
    annotate_message_groups,
    extend_compaction_messages,
    included_messages,
    included_token_count,
)


# ── Helper: build a long conversation ────────────────────────────────────

def build_conversation(n_turns: int = 50) -> list[Message]:
    """Create a synthetic conversation with `n_turns` user/assistant pairs."""
    messages: list[Message] = [
        Message("system", ["You are a helpful assistant."]),
    ]
    for i in range(1, n_turns + 1):
        messages.append(
            Message("user", [f"Question {i}: {'x' * 200}"])
        )
        messages.append(
            Message("assistant", [f"Answer {i}: {'y' * 300}"])
        )
    return messages


def build_tool_conversation() -> list[Message]:
    """Create a conversation that includes tool calls and results."""
    return [
        Message("system", ["You are a calculator agent."]),
        Message("user", ["What is 2+2?"]),
        Message(
            "assistant",
            [
                Content.from_function_call(
                    call_id="call_1",
                    name="add",
                    arguments='{"a": 2, "b": 2}',
                )
            ],
        ),
        Message(
            "tool",
            [
                Content.from_function_result(
                    call_id="call_1",
                    result="4",
                )
            ],
        ),
        Message("assistant", ["2+2 = 4"]),
        Message("user", ["And 10*3?"]),
        Message(
            "assistant",
            [
                Content.from_function_call(
                    call_id="call_2",
                    name="multiply",
                    arguments='{"a": 10, "b": 3}',
                )
            ],
        ),
        Message(
            "tool",
            [
                Content.from_function_result(
                    call_id="call_2",
                    result="30",
                )
            ],
        ),
        Message("assistant", ["10*3 = 30"]),
    ]


async def main() -> None:
    messages = build_conversation(n_turns=50)
    print(f"Conversation has {len(messages)} messages")

    # ── Sliding Window ───────────────────────────────────────────────────
    print("\n=== Sliding Window ===")
    sliding = SlidingWindowStrategy(keep_last_groups=6)
    annotate_message_groups(messages)
    await sliding(messages)
    kept = included_messages(messages)
    print(f"Kept {len(kept)} of {len(messages)} messages")

    # ── Truncation ───────────────────────────────────────────────────────
    print("\n=== Truncation ===")
    messages = build_conversation(n_turns=50)
    truncation = TruncationStrategy(max_n=20, compact_to=10)
    annotate_message_groups(messages)
    await truncation(messages)
    kept = included_messages(messages)
    print(f"Kept {len(kept)} of {len(messages)} messages")

    # ── Selective Tool Call ───────────────────────────────────────────────
    print("\n=== Selective Tool Call ===")
    tool_messages = build_tool_conversation()
    selective = SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=1)
    annotate_message_groups(tool_messages)
    await selective(tool_messages)
    kept = included_messages(tool_messages)
    print(f"Kept {len(kept)} of {len(tool_messages)} messages")

    # ── Tool Result Replacement ──────────────────────────────────────────
    print("\n=== Tool Result Replacement ===")
    tool_messages = build_tool_conversation()
    tool_result = ToolResultCompactionStrategy(keep_last_tool_call_groups=1)
    annotate_message_groups(tool_messages)
    await tool_result(tool_messages)
    kept = included_messages(tool_messages)
    print(f"Kept {len(kept)} of {len(tool_messages)} messages")
    for msg in kept:
        if msg.role == "tool":
            text = msg.text if hasattr(msg, 'text') else str(msg.contents)
            print(f"  Tool result: {text}")

    # ── Pipeline (composed strategies) ───────────────────────────────────
    print("\n=== Pipeline (TokenBudgetComposedStrategy) ===")
    messages = build_conversation(n_turns=50)

    # A simple char-based tokenizer for demonstration
    class CharTokenizer:
        def count_tokens(self, text: str) -> int:
            return max(1, len(text) // 4)

    tokenizer = CharTokenizer()
    pipeline = TokenBudgetComposedStrategy(
        token_budget=2048,
        tokenizer=tokenizer,
        strategies=[
            SelectiveToolCallCompactionStrategy(keep_last_tool_call_groups=2),
            SlidingWindowStrategy(keep_last_groups=6),
        ],
    )
    annotate_message_groups(messages, tokenizer=tokenizer)
    await pipeline(messages)
    kept = included_messages(messages)
    print(f"Pipeline kept {len(kept)} of {len(messages)} messages")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n=== Strategy Summary ===")
    print("SlidingWindowStrategy     — Keep N most recent groups")
    print("TruncationStrategy        — Drop oldest until under budget")
    print("SelectiveToolCallStrategy — Remove old tool call/result pairs")
    print("ToolResultCompaction      — Replace old tool results with summary")
    print("SummarizationStrategy     — Summarize old messages via LLM")
    print("TokenBudgetComposed       — Chain strategies in a pipeline")


if __name__ == "__main__":
    asyncio.run(main())
