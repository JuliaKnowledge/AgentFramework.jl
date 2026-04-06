using AgentFramework
using AgentFramework.CodingAgents
using Test

function _cmd_args(cmd::Cmd)
    return collect(cmd)
end

@testset "CodingAgents.jl" begin
    @testset "GitHubCopilotChatClient captures session ids and resumes" begin
        commands = Vector{Vector{String}}()
        outputs = [
            """
            {"type":"assistant.message","data":{"messageId":"msg-1","content":"OK","toolRequests":[],"reasoningText":"Think","outputTokens":4}}
            {"type":"result","sessionId":"copilot-session-1","exitCode":0,"usage":{"premiumRequests":0,"totalApiDurationMs":12,"sessionDurationMs":14}}
            """,
            """
            {"type":"assistant.message","data":{"messageId":"msg-2","content":"Welcome back","toolRequests":[]}}
            {"type":"result","sessionId":"copilot-session-1","exitCode":0}
            """,
        ]
        idx = Ref(0)

        capture_runner = function(cmd)
            push!(commands, _cmd_args(cmd))
            idx[] += 1
            return (stdout = outputs[idx[]], stderr = "", exitcode = 0)
        end

        client = GitHubCopilotChatClient(capture_runner = capture_runner, stream_runner = (cmd, on_line) -> (stderr = "", exitcode = 0))
        agent = Agent(client = client)
        session = AgentSession(id = "agent-session-1")

        first = run_agent(agent, "Say hello"; session = session)
        @test first.text == "OK"
        @test session.thread_id == "copilot-session-1"
        @test first.conversation_id == "copilot-session-1"

        second = run_agent(agent, "Continue"; session = session)
        @test second.text == "Welcome back"
        @test any(arg -> arg == "--resume=copilot-session-1", commands[2])
        @test any(arg -> occursin("Continue", arg), commands[2])
        @test !any(arg -> occursin("Say hello", arg), commands[2])
    end

    @testset "GitHubCopilotChatClient streams JSON events" begin
        lines = [
            """{"type":"assistant.reasoning_delta","data":{"reasoningId":"r1","deltaContent":"Think"}}""",
            """{"type":"assistant.message_delta","data":{"messageId":"msg-1","deltaContent":"OK"}}""",
            """{"type":"result","sessionId":"copilot-session-2","exitCode":0}""",
        ]

        stream_runner = function(cmd, on_line)
            for line in lines
                on_line(line)
            end
            return (stderr = "", exitcode = 0)
        end

        client = GitHubCopilotChatClient(
            capture_runner = cmd -> (stdout = "", stderr = "", exitcode = 0),
            stream_runner = stream_runner,
        )

        updates = collect(get_response_streaming(client, [Message(:user, "Hello")], ChatOptions()))
        @test length(updates) == 3
        @test is_reasoning(only(updates[1].contents))
        @test get_text(only(updates[2].contents)) == "OK"
        @test updates[3].conversation_id == "copilot-session-2"
        @test updates[3].finish_reason == STOP
    end

    @testset "ClaudeCodeChatClient captures session ids and resumes" begin
        commands = Vector{Vector{String}}()
        outputs = [
            """{"type":"result","subtype":"success","is_error":false,"result":"OK","stop_reason":"end_turn","session_id":"claude-session-1","uuid":"resp-1","usage":{"input_tokens":3,"output_tokens":4}}""",
            """{"type":"result","subtype":"success","is_error":false,"result":"Welcome back","stop_reason":"end_turn","session_id":"claude-session-1","uuid":"resp-2","usage":{"input_tokens":2,"output_tokens":2}}""",
        ]
        idx = Ref(0)

        capture_runner = function(cmd)
            push!(commands, _cmd_args(cmd))
            idx[] += 1
            return (stdout = outputs[idx[]], stderr = "", exitcode = 0)
        end

        client = ClaudeCodeChatClient(capture_runner = capture_runner, stream_runner = (cmd, on_line) -> (stderr = "", exitcode = 0))
        agent = Agent(client = client)
        session = AgentSession(id = "claude-agent-session")

        first = run_agent(agent, "Say hello"; session = session)
        @test first.text == "OK"
        @test session.thread_id == "claude-session-1"
        @test first.conversation_id == "claude-session-1"

        second = run_agent(agent, "Continue"; session = session)
        @test second.text == "Welcome back"
        @test any(arg -> arg == "--resume", commands[2])
        @test any(arg -> arg == "claude-session-1", commands[2])
        @test any(arg -> occursin("Continue", arg), commands[2])
        @test !any(arg -> occursin("Say hello", arg), commands[2])
    end

    @testset "ClaudeCodeChatClient supports streaming and structured output" begin
        lines = [
            """{"type":"system","subtype":"init","session_id":"claude-stream-1"}""",
            """{"type":"stream_event","event":{"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-6"}},"session_id":"claude-stream-1"}""",
            """{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"Think"}},"session_id":"claude-stream-1"}""",
            """{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"OK"}},"session_id":"claude-stream-1"}""",
            """{"type":"result","subtype":"success","is_error":false,"result":"OK","stop_reason":"end_turn","session_id":"claude-stream-1","uuid":"resp-3","usage":{"input_tokens":3,"output_tokens":4}}""",
        ]
        captured = Vector{Vector{String}}()

        stream_runner = function(cmd, on_line)
            push!(captured, _cmd_args(cmd))
            for line in lines
                on_line(line)
            end
            return (stderr = "", exitcode = 0)
        end

        client = ClaudeCodeChatClient(
            capture_runner = cmd -> (stdout = "", stderr = "", exitcode = 0),
            stream_runner = stream_runner,
        )
        options = ChatOptions(response_format = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}("ok" => Dict{String, Any}("type" => "boolean"))))

        updates = collect(get_response_streaming(client, [Message(:user, "Hello")], options))
        @test length(updates) == 3
        @test is_reasoning(only(updates[1].contents))
        @test get_text(only(updates[2].contents)) == "OK"
        @test updates[3].conversation_id == "claude-stream-1"
        @test any(arg -> arg == "--json-schema", captured[1])
    end

    @testset "Managed coding clients reject Julia FunctionTools" begin
        tool = FunctionTool(name = "echo", description = "Echo", func = identity)
        options = ChatOptions(tools = [tool])
        copilot = GitHubCopilotChatClient(capture_runner = cmd -> (stdout = "", stderr = "", exitcode = 0))
        claude = ClaudeCodeChatClient(capture_runner = cmd -> (stdout = "", stderr = "", exitcode = 0))

        @test_throws ChatClientInvalidRequestError get_response(copilot, [Message(:user, "Hi")], options)
        @test_throws ChatClientInvalidRequestError get_response(claude, [Message(:user, "Hi")], options)
    end
end
