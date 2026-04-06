"""
Human-in-the-Loop Workflows — Python

This sample demonstrates human-in-the-loop patterns in workflows:
  1. Using request_info to pause a workflow and request human input.
  2. Resuming a workflow with responses.
  3. Function approval requests for sensitive tool calls.
  4. A multi-step approval pipeline with writer, checker, and approver.

Prerequisites:
  - Ollama running locally with `qwen3:8b` pulled
  - pip install agent-framework-ollama
"""

import asyncio

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    Content,
    Executor,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
    response_handler,
)
from agent_framework.ollama import OllamaChatClient
from typing_extensions import Never


# ── Review Executor with request_info ────────────────────────────────────────


class DraftReviewRequest:
    """Data class for draft review requests."""

    def __init__(self, prompt: str, draft: str):
        self.prompt = prompt
        self.draft = draft


class ReviewExecutor(Executor):
    """Pauses the workflow to request human feedback on a draft."""

    def __init__(self) -> None:
        super().__init__(id="reviewer")

    @handler
    async def on_draft(
        self,
        draft: AgentExecutorResponse,
        ctx: WorkflowContext[Never, str],
    ) -> None:
        text = draft.agent_response.text or ""
        await ctx.request_info(
            request_data=DraftReviewRequest(
                prompt=f"Please review this draft: {text[:100]}...",
                draft=text,
            ),
            response_type=str,
        )

    @response_handler
    async def on_human_feedback(
        self,
        original_request: DraftReviewRequest,
        feedback: str,
        ctx: WorkflowContext[Never, str],
    ) -> None:
        """Process human feedback after workflow resumes."""
        if feedback.strip().lower() == "approve":
            await ctx.yield_output(original_request.draft)
        else:
            await ctx.yield_output(f"Revision needed: {feedback}")


# ── Approval Gate Executor ───────────────────────────────────────────────────


class ApprovalRequest:
    """Data class for approval gate requests."""

    def __init__(self, prompt: str, content: str):
        self.prompt = prompt
        self.content = content


class ApprovalGate(Executor):
    """Requests final human approval before publishing."""

    def __init__(self) -> None:
        super().__init__(id="approver")

    @handler
    async def on_content(
        self,
        content: AgentExecutorResponse,
        ctx: WorkflowContext[Never, str],
    ) -> None:
        text = content.agent_response.text or ""
        await ctx.request_info(
            request_data=ApprovalRequest(
                prompt=f"Fact-check result: {text[:200]}\n\nApprove for publication?",
                content=text,
            ),
            response_type=str,
        )

    @response_handler
    async def on_approval(
        self,
        original_request: ApprovalRequest,
        feedback: str,
        ctx: WorkflowContext[Never, str],
    ) -> None:
        if feedback.strip().lower() == "approve":
            await ctx.yield_output(f"✅ Published: {original_request.content}")
        else:
            await ctx.yield_output(f"❌ Rejected: {feedback}")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    client = OllamaChatClient(
        host="http://localhost:11434",
        model="qwen3:8b",
    )

    # ── 1. Simple review workflow ────────────────────────────────────────
    print("=== Review Workflow with Human-in-the-Loop ===\n")

    writer = AgentExecutor(
        client.as_agent(
            name="Writer",
            instructions="Write a short paragraph about the given topic.",
        )
    )
    reviewer = ReviewExecutor()

    review_workflow = (
        WorkflowBuilder(start_executor=writer)
        .add_edge(writer, reviewer)
        .build()
    )

    # Step 1: Run until human input is needed
    events = await review_workflow.run("Write about the Julia programming language")
    request_info_events = events.get_request_info_events()
    print(f"State: {events.get_final_state()}")
    print(f"Pending requests: {len(request_info_events)}")

    # Step 2: Provide human feedback and resume
    if request_info_events:
        request_id = request_info_events[0].request_id
        print(f"  Request data: {request_info_events[0].data.prompt}")

        # Simulate human approving the draft
        events = await review_workflow.run(
            responses={request_id: "approve"}
        )
        outputs = events.get_outputs()
        print(f"\nApproved output:\n  {outputs[0][:200]}...")

    # ── 2. Multi-step approval pipeline ──────────────────────────────────
    print("\n\n=== Multi-Step Approval Pipeline ===\n")

    pipeline_writer = AgentExecutor(
        client.as_agent(
            name="Writer",
            instructions="Write a one-paragraph article about the topic.",
        )
    )
    fact_checker = AgentExecutor(
        client.as_agent(
            name="FactChecker",
            instructions="Review the text for factual accuracy. Output PASS or list corrections.",
        )
    )
    approver = ApprovalGate()

    pipeline = (
        WorkflowBuilder(start_executor=pipeline_writer)
        .add_edge(pipeline_writer, fact_checker)
        .add_edge(fact_checker, approver)
        .build()
    )

    # Run until approval is needed
    events = await pipeline.run("The history of the Julia programming language")
    request_info_events = events.get_request_info_events()
    print(f"State: {events.get_final_state()}")

    if request_info_events:
        req = request_info_events[0]
        print(f"Approval requested:\n  {req.data.prompt[:200]}...")

        # Human approves
        events = await pipeline.run(responses={req.request_id: "approve"})
        outputs = events.get_outputs()
        for output in outputs:
            print(f"\n{output}")

    # ── 3. Function approval request (conceptual) ──────────────────────
    print("\n\n=== Function Approval Request (Conceptual) ===\n")

    agent = client.as_agent(
        name="FileManager",
        instructions="You manage files. Describe what you would do when asked to delete a file.",
    )

    # Without tools requiring approval defined, the agent just responds with text.
    # In a real scenario, tools with requires_approval=True would trigger
    # function_approval_request content that the caller must approve/reject.
    response = await agent.run("Please delete old_report.txt")
    print(f"Agent response: {response.text[:300]}")
    print("\n(Full function-approval requires tools with requires_approval=True)")


if __name__ == "__main__":
    asyncio.run(main())
