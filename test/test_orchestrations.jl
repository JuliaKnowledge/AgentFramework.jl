using AgentFramework
using Test

mutable struct OrchMockChatClient <: AbstractChatClient
    responses::Vector{String}
    call_count::Ref{Int}
    received_messages::Vector{Vector{Message}}
end

OrchMockChatClient(responses::Vector{String}) = OrchMockChatClient(responses, Ref(0), Vector{Vector{Message}}())
OrchMockChatClient(response::String) = OrchMockChatClient([response])

function AgentFramework.get_response(client::OrchMockChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count[] += 1
    push!(client.received_messages, deepcopy(messages))
    idx = min(client.call_count[], length(client.responses))
    ChatResponse(
        messages = [Message(:assistant, client.responses[idx])],
        finish_reason = STOP,
        model_id = "orchestration-mock",
    )
end

function AgentFramework.get_response_streaming(client::OrchMockChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    response = AgentFramework.get_response(client, messages, options)
    channel = Channel{ChatResponseUpdate}(1)
    Threads.@spawn begin
        for message in response.messages
            for content in message.contents
                put!(channel, ChatResponseUpdate(role = message.role, contents = [content]))
            end
        end
        put!(channel, ChatResponseUpdate(finish_reason = response.finish_reason))
        close(channel)
    end
    return channel
end

function make_orchestration_agent(name::String, responses::Vector{String})
    client = OrchMockChatClient(responses)
    agent = Agent(name = name, instructions = "Be concise.", client = client)
    return agent, client
end

make_orchestration_agent(name::String, response::String) = make_orchestration_agent(name, [response])

struct TestMagenticManager <: AbstractMagenticManager end

function AgentFramework.magentic_plan(::TestMagenticManager, context::MagenticContext)
    MagenticTaskLedger(
        facts = ["task:" * (isempty(context.conversation) ? "" : get_text(context.conversation[1]))],
        plan = ["consult planner", "consult writer"],
        current_step = 1,
    )
end

function AgentFramework.magentic_select(::TestMagenticManager, context::MagenticContext)
    context.round == 0 && return "planner"
    context.round == 1 && return "writer"
    return nothing
end

AgentFramework.magentic_finalize(::TestMagenticManager, context::MagenticContext) = context.conversation

@testset "Orchestration Builders" begin
    @testset "Builders reject empty participants" begin
        @test_throws ArgumentError SequentialBuilder(participants = Agent[])
        @test_throws ArgumentError ConcurrentBuilder(participants = Agent[])
        @test_throws ArgumentError GroupChatBuilder(participants = Agent[])
        @test_throws ArgumentError MagenticBuilder(participants = Agent[])
    end

    @testset "SequentialBuilder chains agent conversations" begin
        writer, _ = make_orchestration_agent("writer", "Draft complete.")
        reviewer, _ = make_orchestration_agent("reviewer", "Looks good.")

        workflow = build(SequentialBuilder(participants = [writer, reviewer]))
        result = run_workflow(workflow, "Write a memo")

        outputs = get_outputs(result)
        @test length(outputs) == 1
        conversation = only(outputs)
        @test conversation isa Vector{Message}
        @test length(conversation) == 3
        @test conversation[1].role == :user
        @test get_text(conversation[2]) == "Draft complete."
        @test get_text(conversation[3]) == "Looks good."
    end

    @testset "SequentialBuilder supports conversation executors" begin
        writer, _ = make_orchestration_agent("writer", "Initial draft.")
        summarizer = ExecutorSpec(
            id = "summarizer",
            input_types = DataType[Vector{Message}],
            output_types = DataType[Vector{Message}],
            yield_types = DataType[],
            handler = (conversation, ctx) -> begin
                send_message(ctx, vcat(deepcopy(conversation), [Message(:assistant, "Summary added.")]))
            end,
        )

        workflow = build(SequentialBuilder(participants = [writer, summarizer]))
        result = run_workflow(workflow, "Summarize this")
        conversation = only(get_outputs(result))
        @test get_text(conversation[end]) == "Summary added."
    end

    @testset "ConcurrentBuilder aggregates default replies" begin
        analyst_a, _ = make_orchestration_agent("analyst-a", "Alpha")
        analyst_b, _ = make_orchestration_agent("analyst-b", "Beta")

        workflow = build(ConcurrentBuilder(participants = [analyst_a, analyst_b]))
        result = run_workflow(workflow, "Compare plans")

        outputs = get_outputs(result)
        @test length(outputs) == 1
        conversation = only(outputs)
        @test conversation isa Vector{Message}
        @test length(conversation) == 3
        @test get_text(conversation[1]) == "Compare plans"
        @test Set(get_text(message) for message in conversation[2:end]) == Set(["Alpha", "Beta"])
    end

    @testset "ConcurrentBuilder supports callback aggregators" begin
        analyst_a, _ = make_orchestration_agent("analyst-a", "Alpha")
        analyst_b, _ = make_orchestration_agent("analyst-b", "Beta")
        builder = ConcurrentBuilder(participants = [analyst_a, analyst_b])
        with_aggregator(builder, results -> join([get_text(result.conversation[end]) for result in results], " | "))

        workflow = build(builder)
        result = run_workflow(workflow, "Compare plans")
        @test only(get_outputs(result)) == "Alpha | Beta"
    end

    @testset "GroupChatBuilder supports selection functions" begin
        planner, planner_client = make_orchestration_agent("planner", "Plan from planner")
        writer, writer_client = make_orchestration_agent("writer", "Draft from writer")

        builder = GroupChatBuilder(
            participants = [planner, writer],
            selection_func = state -> state.round == 0 ? "writer" : nothing,
        )
        workflow = build(builder)
        result = run_workflow(workflow, "Start discussion")

        outputs = get_outputs(result)
        @test length(outputs) == 1
        conversation = only(outputs)
        @test length(conversation) == 2
        @test get_text(conversation[end]) == "Draft from writer"
        @test planner_client.call_count[] == 0
        @test writer_client.call_count[] == 1
    end

    @testset "GroupChatBuilder supports orchestrator agents" begin
        planner, planner_client = make_orchestration_agent("planner", "Plan from planner")
        writer, writer_client = make_orchestration_agent("writer", "Draft from writer")
        selector, selector_client = make_orchestration_agent("selector", ["planner", "DONE"])

        workflow = build(GroupChatBuilder(
            participants = [planner, writer],
            orchestrator_agent = selector,
            max_rounds = 3,
        ))
        result = run_workflow(workflow, "Start discussion")

        conversation = only(get_outputs(result))
        @test length(conversation) == 2
        @test get_text(conversation[end]) == "Plan from planner"
        @test planner_client.call_count[] == 1
        @test writer_client.call_count[] == 0
        @test selector_client.call_count[] == 2
    end

    @testset "MagenticBuilder coordinates manager-driven turns" begin
        planner, planner_client = make_orchestration_agent("planner", "Research complete")
        writer, writer_client = make_orchestration_agent("writer", "Draft complete")

        workflow = build(MagenticBuilder(
            participants = [planner, writer],
            manager = TestMagenticManager(),
            max_round_count = 3,
        ))
        result = run_workflow(workflow, "Ship the report")

        conversation = only(get_outputs(result))
        @test length(conversation) == 3
        @test get_text(conversation[2]) == "Research complete"
        @test get_text(conversation[3]) == "Draft complete"
        @test planner_client.call_count[] == 1
        @test writer_client.call_count[] == 1
    end

    @testset "MagenticBuilder plan review pauses and resumes" begin
        planner, planner_client = make_orchestration_agent("planner", "Research complete")
        writer, writer_client = make_orchestration_agent("writer", "Draft complete")
        workflow = build(MagenticBuilder(
            participants = [planner, writer],
            manager = TestMagenticManager(),
            enable_plan_review = true,
            max_round_count = 3,
        ))

        first_run = run_workflow(workflow, "Ship the report")
        @test get_final_state(first_run) == WF_IDLE_WITH_PENDING_REQUESTS
        requests = get_request_info_events(first_run)
        @test length(requests) == 1
        @test requests[1].data isa MagenticPlanReviewRequest

        resumed = run_workflow(
            workflow,
            nothing;
            responses = Dict{String, Any}(requests[1].request_id => MagenticPlanReviewResponse(approved = true)),
        )
        conversation = only(get_outputs(resumed))
        @test length(conversation) == 3
        @test get_text(conversation[2]) == "Research complete"
        @test get_text(conversation[3]) == "Draft complete"
        @test planner_client.call_count[] == 1
        @test writer_client.call_count[] == 1
    end
end
