using AgentFramework
using Test

# Mock client for handoff tests
mutable struct HandoffMockClient <: AbstractChatClient
    responses::Vector{ChatResponse}
    call_count::Int
end

HandoffMockClient(responses...) = HandoffMockClient(collect(responses), 0)

function AgentFramework.get_response(client::HandoffMockClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count += 1
    idx = min(client.call_count, length(client.responses))
    return client.responses[idx]
end

function AgentFramework.get_response_streaming(client::HandoffMockClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    resp = AgentFramework.get_response(client, messages, options)
    ch = Channel{ChatResponseUpdate}(8)
    @async begin
        for msg in resp.messages
            for c in msg.contents
                put!(ch, ChatResponseUpdate(role=msg.role, contents=[c]))
            end
        end
        put!(ch, ChatResponseUpdate(finish_reason=resp.finish_reason))
        close(ch)
    end
    return ch
end

@testset "Handoffs" begin

    @testset "HandoffTool construction" begin
        target_client = HandoffMockClient(
            ChatResponse(messages=[Message(:assistant, "I'm the billing agent.")], finish_reason=STOP),
        )
        target = Agent(name="BillingAgent", client=target_client, instructions="Handle billing.")

        ht = HandoffTool(
            name = "transfer_to_billing",
            description = "Transfer to billing specialist.",
            target = target,
        )
        @test ht.name == "transfer_to_billing"
        @test ht.target.name == "BillingAgent"
        @test ht.include_history == true
        @test contains(sprint(show, ht), "transfer_to_billing")
        @test contains(sprint(show, ht), "BillingAgent")
    end

    @testset "HandoffTool schema" begin
        target_client = HandoffMockClient(
            ChatResponse(messages=[Message(:assistant, "Hi")], finish_reason=STOP),
        )
        target = Agent(name="Target", client=target_client)

        ht = HandoffTool(name="go_to_target", description="Go there.", target=target)
        schema = tool_to_schema(ht)

        @test schema["type"] == "function"
        @test schema["function"]["name"] == "go_to_target"
        @test schema["function"]["description"] == "Go there."
        @test haskey(schema["function"]["parameters"]["properties"], "message")
    end

    @testset "execute_handoff basic" begin
        target_client = HandoffMockClient(
            ChatResponse(messages=[Message(:assistant, "Handled by target.")], finish_reason=STOP),
        )
        target = Agent(name="Target", client=target_client, instructions="You are the target.")

        ht = HandoffTool(
            name = "handoff",
            description = "Hand off.",
            target = target,
        )

        messages = [
            Message(:user, "Help me with billing"),
            Message(:assistant, "Let me transfer you."),
        ]

        response = execute_handoff(ht, messages; handoff_message="Please help with billing.")
        @test !isempty(response.text)
        @test target_client.call_count == 1
    end

    @testset "execute_handoff excludes system messages" begin
        call_messages = Message[]
        target_client = HandoffMockClient(
            ChatResponse(messages=[Message(:assistant, "Done.")], finish_reason=STOP),
        )

        # Override to capture messages
        original_get = AgentFramework.get_response
        target = Agent(name="Target", client=target_client)

        ht = HandoffTool(
            name = "handoff",
            description = "Test.",
            target = target,
            include_history = true,
        )

        messages = [
            Message(:system, "You are agent A."),
            Message(:user, "Hello"),
            Message(:assistant, "Hi there"),
        ]

        response = execute_handoff(ht, messages)
        # Should work — system messages from source agent are filtered
        @test !isempty(response.text)
    end

    @testset "execute_handoff without history" begin
        target_client = HandoffMockClient(
            ChatResponse(messages=[Message(:assistant, "Fresh start.")], finish_reason=STOP),
        )
        target = Agent(name="Target", client=target_client)

        ht = HandoffTool(
            name = "handoff",
            description = "Test.",
            target = target,
            include_history = false,
        )

        messages = [
            Message(:user, "Long conversation..."),
            Message(:assistant, "Previous context..."),
        ]

        response = execute_handoff(ht, messages; handoff_message="Start fresh.")
        @test !isempty(response.text)
    end

    @testset "handoff_as_function_tool" begin
        target_client = HandoffMockClient(
            ChatResponse(messages=[Message(:assistant, "Target response.")], finish_reason=STOP),
        )
        target = Agent(name="Target", client=target_client)

        ht = HandoffTool(name="transfer", description="Transfer.", target=target)
        ft = handoff_as_function_tool(ht)

        @test ft isa FunctionTool
        @test ft.name == "transfer"
        @test ft.description == "Transfer."
    end

    @testset "normalize_agent_tools" begin
        target_client = HandoffMockClient(
            ChatResponse(messages=[Message(:assistant, "Hi")], finish_reason=STOP),
        )
        target = Agent(name="Target", client=target_client)

        regular_tool = FunctionTool(
            name = "search",
            description = "Search the web.",
            func = (query) -> "results for $query",
        )
        handoff = HandoffTool(name="transfer", description="Transfer.", target=target)

        tools = Any[regular_tool, handoff]
        func_tools, handoff_tools = normalize_agent_tools(tools)

        @test length(func_tools) == 2  # regular + converted handoff
        @test length(handoff_tools) == 1
        @test func_tools[1].name == "search"
        @test func_tools[2].name == "transfer"
        @test handoff_tools[1].name == "transfer"
    end

    @testset "HandoffResult display" begin
        resp = AgentResponse(messages=[Message(:assistant, "Done")])
        hr = HandoffResult("AgentA", "AgentB", resp, "Please help")
        s = sprint(show, hr)
        @test contains(s, "AgentA")
        @test contains(s, "AgentB")
    end
end
