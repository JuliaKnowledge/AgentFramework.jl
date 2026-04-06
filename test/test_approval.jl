using Test
using AgentFramework

# ─── Mock client for approval testing ─────────────────────────────────────

mutable struct ApprovalMockClient <: AbstractChatClient
    responses::Vector{Vector{Content}}
    call_count::Int
end
ApprovalMockClient(responses...) = ApprovalMockClient(collect(responses), 0)

function AgentFramework.get_response(client::ApprovalMockClient, messages::Vector{Message}, options=nothing)
    client.call_count += 1
    idx = min(client.call_count, length(client.responses))
    contents = client.responses[idx]
    msg = Message(:assistant, contents)
    return ChatResponse(messages=[msg], finish_reason=STOP)
end


@testset "Approval Framework" begin

    # ═══════════════════════════════════════════════════════════════════════
    #  1. Content Types
    # ═══════════════════════════════════════════════════════════════════════

    @testset "Approval content constructors" begin
        fc = function_call_content("call-1", "dangerous_tool", """{"x": 1}""")

        @testset "function_approval_request_content" begin
            req = function_approval_request_content("call-1", fc)
            @test req.type == AgentFramework.FUNCTION_APPROVAL_REQUEST
            @test req.id == "call-1"
            @test req.function_call === fc
            @test req.user_input_request == true
            @test is_approval_request(req)
            @test !is_approval_response(req)
        end

        @testset "function_approval_response_content — approved" begin
            resp = function_approval_response_content(true, "call-1", fc)
            @test resp.type == AgentFramework.FUNCTION_APPROVAL_RESPONSE
            @test resp.approved == true
            @test resp.id == "call-1"
            @test resp.function_call === fc
            @test is_approval_response(resp)
            @test !is_approval_request(resp)
        end

        @testset "function_approval_response_content — rejected" begin
            resp = function_approval_response_content(false, "call-1", fc)
            @test resp.approved == false
        end

        @testset "to_approval_response" begin
            req = function_approval_request_content("call-1", fc)
            resp = to_approval_response(req, true)
            @test resp.type == AgentFramework.FUNCTION_APPROVAL_RESPONSE
            @test resp.approved == true
            @test resp.id == "call-1"
            @test resp.function_call === fc

            rejected = to_approval_response(req, false)
            @test rejected.approved == false
        end

        @testset "to_approval_response errors on wrong type" begin
            text = text_content("hello")
            @test_throws ErrorException to_approval_response(text, true)
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  2. FunctionTool approval_mode
    # ═══════════════════════════════════════════════════════════════════════

    @testset "FunctionTool approval_mode" begin
        @testset "Default is :never_require" begin
            tool = FunctionTool(name="safe", description="safe", func=identity)
            @test tool.approval_mode == :never_require
        end

        @testset "Can set :always_require" begin
            tool = FunctionTool(name="danger", description="dangerous", func=identity,
                                approval_mode=:always_require)
            @test tool.approval_mode == :always_require
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  3. Approval in tool execution
    # ═══════════════════════════════════════════════════════════════════════

    @testset "Approval required — returns approval requests" begin
        danger_tool = FunctionTool(
            name="delete_file",
            description="Delete a file",
            func=(path) -> "deleted $path",
            approval_mode=:always_require,
        )

        # Simulate: LLM returns a function call to delete_file
        fc = function_call_content("c1", "delete_file", """{"path":"/important.txt"}""")
        tool_calls = [fc]

        # Create a minimal agent with the tool
        client = ApprovalMockClient([text_content("done")])
        agent = Agent(name="test", client=client, instructions="test",
                      tools=[danger_tool])

        results = AgentFramework._execute_tool_calls(agent, agent.tools, tool_calls)

        @test length(results) == 1
        @test is_approval_request(results[1])
        @test results[1].id == "c1"
        @test results[1].function_call === fc
    end

    @testset "No approval needed — executes normally" begin
        safe_tool = FunctionTool(
            name="get_time",
            description="Get current time",
            func=() -> "12:00",
        )

        fc = function_call_content("c1", "get_time", """{}""")
        client = ApprovalMockClient([text_content("done")])
        agent = Agent(name="test", client=client, instructions="test",
                      tools=[safe_tool])

        results = AgentFramework._execute_tool_calls(agent, agent.tools, [fc])
        @test length(results) == 1
        @test results[1].type == AgentFramework.FUNCTION_RESULT
        @test results[1].result == "12:00"
    end

    @testset "Mixed tools — approval triggers for batch" begin
        safe = FunctionTool(name="safe", description="safe", func=() -> "ok")
        danger = FunctionTool(name="danger", description="danger", func=() -> "boom",
                              approval_mode=:always_require)

        fc_safe = function_call_content("c1", "safe", """{}""")
        fc_danger = function_call_content("c2", "danger", """{}""")

        client = ApprovalMockClient([text_content("done")])
        agent = Agent(name="test", client=client, instructions="test",
                      tools=[safe, danger])

        results = AgentFramework._execute_tool_calls(agent, agent.tools, [fc_safe, fc_danger])

        # Both return as approval requests when any tool in the batch needs approval
        @test length(results) == 2
        @test all(is_approval_request, results)
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  4. MCPSpecificApproval
    # ═══════════════════════════════════════════════════════════════════════

    @testset "MCPSpecificApproval" begin
        @testset "Construction with lists" begin
            approval = MCPSpecificApproval(
                always_require_approval=["delete", "write"],
                never_require_approval=["read", "list"],
            )
            @test length(approval.always_require_approval) == 2
            @test "delete" in approval.always_require_approval
            @test "read" in approval.never_require_approval
        end

        @testset "Default is nothing" begin
            approval = MCPSpecificApproval()
            @test approval.always_require_approval === nothing
            @test approval.never_require_approval === nothing
        end
    end

end  # Approval Framework
