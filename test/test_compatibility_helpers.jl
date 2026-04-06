using AgentFramework
using Test

mutable struct CompatMockChatClient <: AbstractChatClient
    responses::Vector{String}
    call_count::Ref{Int}
end

CompatMockChatClient(responses::Vector{String}) = CompatMockChatClient(responses, Ref(0))
CompatMockChatClient(response::String) = CompatMockChatClient([response])

function AgentFramework.get_response(client::CompatMockChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count[] += 1
    idx = min(client.call_count[], length(client.responses))
    return ChatResponse(
        messages = [Message(:assistant, client.responses[idx])],
        finish_reason = STOP,
        model_id = "compat-mock",
    )
end

function AgentFramework.get_response_streaming(client::CompatMockChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    response = AgentFramework.get_response(client, messages, options)
    ch = Channel{ChatResponseUpdate}(1)
    Threads.@spawn begin
        for message in response.messages
            for content in message.contents
                put!(ch, ChatResponseUpdate(role = message.role, contents = [content]))
            end
        end
        put!(ch, ChatResponseUpdate(finish_reason = response.finish_reason))
        close(ch)
    end
    return ch
end

function make_compat_agent(name::String, responses::Vector{String})
    client = CompatMockChatClient(responses)
    agent = Agent(name = name, instructions = "Be concise.", client = client)
    return agent, client
end

make_compat_agent(name::String, response::String) = make_compat_agent(name, [response])

struct CompatMagenticManager <: AbstractMagenticManager end

function AgentFramework.magentic_plan(::CompatMagenticManager, context::MagenticContext)
    MagenticTaskLedger(
        facts = ["seed:" * (isempty(context.conversation) ? "" : get_text(context.conversation[1]))],
        plan = ["planner", "writer"],
        current_step = 1,
    )
end

function AgentFramework.magentic_select(::CompatMagenticManager, context::MagenticContext)
    context.round == 0 && return "planner"
    context.round == 1 && return "writer"
    return nothing
end

AgentFramework.magentic_finalize(::CompatMagenticManager, context::MagenticContext) = context.conversation

@testset "Compatibility helpers" begin
    @testset "AssistantAgent resolves migration aliases" begin
        client = CompatMockChatClient("Hello")
        specialist = Agent(name = "Specialist", instructions = "Math only.", client = CompatMockChatClient("42"))
        handoff = HandoffTool(
            name = "transfer_to_specialist",
            description = "Transfer to the specialist.",
            target = specialist,
        )

        agent = AssistantAgent(
            model_client = client,
            system_message = "Be helpful.",
            tools = [handoff],
        )

        @test agent isa Agent
        @test agent.client === client
        @test agent.instructions == "Be helpful."
        @test length(agent.tools) == 1
        @test agent.tools[1].name == "transfer_to_specialist"
    end

    @testset "ChatCompletionAgent rejects conflicting aliases" begin
        client = CompatMockChatClient("Hello")
        other_client = CompatMockChatClient("Hi")

        @test_throws ArgumentError ChatCompletionAgent(
            client = client,
            model_client = other_client,
        )

        @test_throws ArgumentError ChatCompletionAgent(
            client = client,
            instructions = "One",
            system_message = "Two",
        )
    end

    @testset "RoundRobinGroupChat wraps group chat workflow" begin
        planner, planner_client = make_compat_agent("planner", "Plan")
        writer, writer_client = make_compat_agent("writer", "Draft")

        workflow = RoundRobinGroupChat(
            participants = [planner, writer],
            max_turns = 2,
        )
        result = run_workflow(workflow, "Start")

        conversation = only(get_outputs(result))
        @test length(conversation) == 3
        @test get_text(conversation[2]) == "Plan"
        @test get_text(conversation[3]) == "Draft"
        @test planner_client.call_count[] == 1
        @test writer_client.call_count[] == 1
    end

    @testset "SelectorGroupChat supports selector_func alias" begin
        planner, planner_client = make_compat_agent("planner", "Plan")
        writer, writer_client = make_compat_agent("writer", "Draft")

        workflow = SelectorGroupChat(
            participants = [planner, writer],
            selector_func = state -> state.round == 0 ? "writer" : nothing,
        )
        result = run_workflow(workflow, "Discuss")

        conversation = only(get_outputs(result))
        @test length(conversation) == 2
        @test get_text(conversation[end]) == "Draft"
        @test planner_client.call_count[] == 0
        @test writer_client.call_count[] == 1
    end

    @testset "SelectorGroupChat requires a selection mechanism" begin
        planner, _ = make_compat_agent("planner", "Plan")
        writer, _ = make_compat_agent("writer", "Draft")

        @test_throws ArgumentError SelectorGroupChat(
            participants = [planner, writer],
        )
    end

    @testset "MagenticOneGroupChat wraps MagenticBuilder" begin
        planner, planner_client = make_compat_agent("planner", "Research complete")
        writer, writer_client = make_compat_agent("writer", "Draft complete")

        workflow = MagenticOneGroupChat(
            participants = [planner, writer],
            manager = CompatMagenticManager(),
            max_turns = 3,
        )
        result = run_workflow(workflow, "Ship the report")

        conversation = only(get_outputs(result))
        @test length(conversation) == 3
        @test get_text(conversation[2]) == "Research complete"
        @test get_text(conversation[3]) == "Draft complete"
        @test planner_client.call_count[] == 1
        @test writer_client.call_count[] == 1
    end
end
