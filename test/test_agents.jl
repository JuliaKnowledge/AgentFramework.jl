using AgentFramework
using Test

# Mock chat client for testing
mutable struct MockChatClient <: AbstractChatClient
    responses::Vector{ChatResponse}
    call_count::Int
end

MockChatClient(responses::Vector{ChatResponse}) = MockChatClient(responses, 0)
MockChatClient(response::ChatResponse) = MockChatClient([response], 0)

function AgentFramework.get_response(client::MockChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count += 1
    idx = min(client.call_count, length(client.responses))
    return client.responses[idx]
end

function AgentFramework.get_response_streaming(client::MockChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    resp = AgentFramework.get_response(client, messages, options)
    ch = Channel{ChatResponseUpdate}(1)
    Threads.@spawn begin
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

mutable struct MetadataTrackingChatClient <: AbstractChatClient
    responses::Vector{ChatResponse}
    streaming_updates::Vector{Vector{ChatResponseUpdate}}
    seen_messages::Vector{Vector{Message}}
    seen_options::Vector{ChatOptions}
    call_count::Int
end

function MetadataTrackingChatClient(response::ChatResponse)
    MetadataTrackingChatClient([response], Vector{Vector{ChatResponseUpdate}}(), Vector{Vector{Message}}(), ChatOptions[], 0)
end

function MetadataTrackingChatClient(streaming_updates::Vector{ChatResponseUpdate})
    MetadataTrackingChatClient(ChatResponse[], [streaming_updates], Vector{Vector{Message}}(), ChatOptions[], 0)
end

function AgentFramework.get_response(client::MetadataTrackingChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count += 1
    push!(client.seen_messages, copy(messages))
    push!(client.seen_options, options)
    idx = min(client.call_count, length(client.responses))
    return client.responses[idx]
end

function AgentFramework.get_response_streaming(client::MetadataTrackingChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    client.call_count += 1
    push!(client.seen_messages, copy(messages))
    push!(client.seen_options, options)
    idx = min(client.call_count, length(client.streaming_updates))
    ch = Channel{ChatResponseUpdate}(max(length(client.streaming_updates[idx]), 1))
    Threads.@spawn begin
        for update in client.streaming_updates[idx]
            put!(ch, update)
        end
        close(ch)
    end
    return ch
end

Base.@kwdef mutable struct TrackingProvider <: BaseContextProvider
    source_id::String
    events::Vector{String}
end

function AgentFramework.before_run!(provider::TrackingProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    push!(provider.events, "before:" * provider.source_id)
    state["count"] = get(state, "count", 0) + 1
end

function AgentFramework.after_run!(provider::TrackingProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    push!(provider.events, "after:" * provider.source_id * ":" * string(get(state, "count", 0)))
end

mutable struct FragmentedStreamingChatClient <: AbstractChatClient
    call_count::Int
end

FragmentedStreamingChatClient() = FragmentedStreamingChatClient(0)

function AgentFramework.get_response(client::FragmentedStreamingChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    error("Non-streaming response is not used in this test client.")
end

function AgentFramework.get_response_streaming(client::FragmentedStreamingChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    client.call_count += 1
    ch = Channel{ChatResponseUpdate}(8)

    Threads.@spawn begin
        if client.call_count == 1
            put!(ch, ChatResponseUpdate(
                role = :assistant,
                contents = [function_call_content("call_1", "add", "")],
                raw_representation = Dict{String, Any}(
                    "id" => "resp-1",
                    "choices" => Any[
                        Dict{String, Any}(
                            "delta" => Dict{String, Any}(
                                "role" => "assistant",
                                "tool_calls" => Any[
                                    Dict{String, Any}(
                                        "index" => 0,
                                        "id" => "call_1",
                                        "function" => Dict{String, Any}("name" => "add"),
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            ))
            put!(ch, ChatResponseUpdate(
                contents = [function_call_content("call_1", "", "{\"a\": 3, ")],
                raw_representation = Dict{String, Any}(
                    "id" => "resp-1",
                    "choices" => Any[
                        Dict{String, Any}(
                            "delta" => Dict{String, Any}(
                                "tool_calls" => Any[
                                    Dict{String, Any}(
                                        "index" => 0,
                                        "function" => Dict{String, Any}("arguments" => "{\"a\": 3, "),
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            ))
            put!(ch, ChatResponseUpdate(
                contents = [function_call_content("call_1", "", "\"b\": 4}")],
                finish_reason = TOOL_CALLS,
                raw_representation = Dict{String, Any}(
                    "id" => "resp-1",
                    "choices" => Any[
                        Dict{String, Any}(
                            "delta" => Dict{String, Any}(
                                "tool_calls" => Any[
                                    Dict{String, Any}(
                                        "index" => 0,
                                        "function" => Dict{String, Any}("arguments" => "\"b\": 4}"),
                                    ),
                                ],
                            ),
                            "finish_reason" => "tool_calls",
                        ),
                    ],
                ),
            ))
        else
            put!(ch, ChatResponseUpdate(
                role = :assistant,
                contents = [text_content("The result is 7.")],
                finish_reason = STOP,
            ))
        end
        close(ch)
    end

    return ch
end

struct FailingStreamingChatClient <: AbstractChatClient end

function AgentFramework.get_response(client::FailingStreamingChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    error("Non-streaming response is not used in this test client.")
end

function AgentFramework.get_response_streaming(client::FailingStreamingChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    throw(ChatClientError("stream failed"))
end

@testset "Agents" begin
    @testset "Agent creation" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "Hello!")],
            finish_reason = STOP,
        ))
        agent = Agent(
            name = "TestBot",
            instructions = "Be helpful.",
            client = client,
        )
        @test agent.name == "TestBot"
        @test agent.instructions == "Be helpful."
        @test isempty(agent.tools)
    end

    @testset "run_agent basic" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "Hello there!")],
            finish_reason = STOP,
            model_id = "mock-model",
        ))
        agent = Agent(client=client)

        response = run_agent(agent, "Hi!")
        @test response.text == "Hello there!"
        @test response.finish_reason == STOP
        @test client.call_count == 1
    end

    @testset "run_agent with instructions" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "I am helpful")],
            finish_reason = STOP,
        ))
        agent = Agent(
            client = client,
            instructions = "You are a helpful assistant.",
        )

        run_agent(agent, "Hello")
        @test client.call_count == 1
    end

    @testset "run_agent with tool execution" begin
        # First response: tool call, second response: final answer
        tool_call_response = ChatResponse(
            messages = [Message(:assistant, [
                function_call_content("call_1", "add", """{"a": 3, "b": 4}"""),
            ])],
            finish_reason = TOOL_CALLS,
        )
        final_response = ChatResponse(
            messages = [Message(:assistant, "The result is 7.")],
            finish_reason = STOP,
        )

        client = MockChatClient([tool_call_response, final_response])

        add_tool = FunctionTool(
            name = "add",
            description = "Add two numbers",
            func = (a, b) -> a + b,
            parameters = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "a" => Dict{String, Any}("type" => "number"),
                    "b" => Dict{String, Any}("type" => "number"),
                ),
                "required" => ["a", "b"],
            ),
        )

        agent = Agent(client=client, tools=[add_tool])
        response = run_agent(agent, "What is 3 + 4?")

        @test response.text == "The result is 7."
        @test client.call_count == 2  # tool call + final
    end

    @testset "run_agent with session" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "Continuing...")],
            finish_reason = STOP,
        ))
        agent = Agent(client=client)
        session = AgentSession(id="test-session")
        session.state["counter"] = 1

        response = run_agent(agent, "Continue"; session=session)
        @test response.text == "Continuing..."
    end

    @testset "run_agent injects session metadata and syncs conversation ids" begin
        client = MetadataTrackingChatClient(ChatResponse(
            messages = [Message(:assistant, "Continuing...")],
            response_id = "resp-1",
            conversation_id = "thread-123",
            finish_reason = STOP,
        ))
        agent = Agent(client = client)
        session = AgentSession(id = "test-session", thread_id = "existing-thread")

        response = run_agent(agent, "Continue"; session = session)
        seen_options = only(client.seen_options)

        @test response.conversation_id == "thread-123"
        @test session.thread_id == "thread-123"
        @test seen_options.additional["_agentframework_session_id"] == "test-session"
        @test seen_options.additional["_agentframework_thread_id"] == "existing-thread"
        @test length(seen_options.additional["_agentframework_input_messages"]) == 1
        @test only(seen_options.additional["_agentframework_input_messages"]).role == :user
        @test get_text(only(seen_options.additional["_agentframework_input_messages"])) == "Continue"
    end

    @testset "run_agent with context provider" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "Got context")],
            finish_reason = STOP,
        ))

        # Use InMemoryHistoryProvider as context provider
        history = InMemoryHistoryProvider(source_id="test_history")
        save_messages!(history, "s1", [
            Message(:user, "previous question"),
            Message(:assistant, "previous answer"),
        ])

        agent = Agent(client=client, context_providers=[history])
        session = AgentSession(id="s1")
        response = run_agent(agent, "Follow up"; session=session)
        @test response.text == "Got context"
    end

    @testset "run_agent with middleware" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "Response")],
            finish_reason = STOP,
        ))

        middleware_called = Ref(false)
        mw = function(ctx::AgentContext, next)
            middleware_called[] = true
            return next(ctx)
        end

        agent = Agent(client=client, agent_middlewares=[mw])
        response = run_agent(agent, "Test")
        @test response.text == "Response"
        @test middleware_called[]
    end

    @testset "run_agent_streaming" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "Streamed response")],
            finish_reason = STOP,
        ))
        agent = Agent(client=client)

        stream = run_agent_streaming(agent, "Hello")
        collected_text = String[]
        for update in stream
            txt = get_text(update)
            if !isempty(txt)
                push!(collected_text, txt)
            end
        end
        @test !isempty(collected_text)
        @test join(collected_text) == "Streamed response"
    end

    @testset "run_agent_streaming syncs conversation ids" begin
        client = MetadataTrackingChatClient([
            ChatResponseUpdate(role = :assistant, contents = [text_content("Streamed ")]),
            ChatResponseUpdate(role = :assistant, contents = [text_content("response")]),
            ChatResponseUpdate(finish_reason = STOP, conversation_id = "stream-thread-1"),
        ])
        agent = Agent(client = client)
        session = AgentSession(id = "stream-session", thread_id = "existing-stream-thread")

        stream = run_agent_streaming(agent, "Hello"; session = session)
        for _ in stream
        end
        response = get_final_response(stream)
        seen_options = only(client.seen_options)

        @test response.text == "Streamed response"
        @test response.conversation_id == "stream-thread-1"
        @test session.thread_id == "stream-thread-1"
        @test seen_options.additional["_agentframework_session_id"] == "stream-session"
        @test seen_options.additional["_agentframework_thread_id"] == "existing-stream-thread"
        @test length(seen_options.additional["_agentframework_input_messages"]) == 1
        @test only(seen_options.additional["_agentframework_input_messages"]).role == :user
        @test get_text(only(seen_options.additional["_agentframework_input_messages"])) == "Hello"
    end

    @testset "run_agent_streaming uses middleware and provider finalization" begin
        client = MockChatClient(ChatResponse(
            messages = [Message(:assistant, "Streamed response")],
            finish_reason = STOP,
        ))

        provider_events = String[]
        p1 = TrackingProvider(source_id="provider_one", events=provider_events)
        p2 = TrackingProvider(source_id="provider_two", events=provider_events)

        agent_middleware_called = Ref(false)
        chat_middleware_called = Ref(false)

        agent_mw = function(ctx::AgentContext, next)
            agent_middleware_called[] = true
            return next(ctx)
        end
        chat_mw = function(ctx::ChatContext, next)
            chat_middleware_called[] = true
            return next(ctx)
        end

        session = AgentSession(id="stream-session")
        agent = Agent(
            client = client,
            context_providers = [p1, p2],
            agent_middlewares = [agent_mw],
            chat_middlewares = [chat_mw],
        )

        stream = run_agent_streaming(agent, "Hello"; session=session)
        for _ in stream
        end
        response = get_final_response(stream)

        @test response.text == "Streamed response"
        @test agent_middleware_called[]
        @test chat_middleware_called[]
        @test provider_events == [
            "before:provider_one",
            "before:provider_two",
            "after:provider_two:1",
            "after:provider_one:1",
        ]
        @test session.state["provider_one"]["count"] == 1
        @test session.state["provider_two"]["count"] == 1
    end

    @testset "run_agent_streaming reconstructs fragmented tool calls" begin
        client = FragmentedStreamingChatClient()

        add_tool = FunctionTool(
            name = "add",
            description = "Add two numbers",
            func = (a, b) -> a + b,
            parameters = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "a" => Dict{String, Any}("type" => "number"),
                    "b" => Dict{String, Any}("type" => "number"),
                ),
                "required" => ["a", "b"],
            ),
        )

        agent = Agent(client=client, tools=[add_tool])
        stream = run_agent_streaming(agent, "What is 3 + 4?")
        for _ in stream
        end
        response = get_final_response(stream)

        @test response.text == "The result is 7."
        @test client.call_count == 2
    end

    @testset "run_agent_streaming surfaces failures" begin
        agent = Agent(client=FailingStreamingChatClient())
        stream = run_agent_streaming(agent, "Hello")
        @test_throws ChatClientError begin
            for _ in stream
            end
        end
        @test_throws ChatClientError get_final_response(stream)
    end

    @testset "create_session" begin
        client = MockChatClient(ChatResponse())
        agent = Agent(client=client)
        session = create_session(agent)
        @test !isempty(session.id)

        session2 = create_session(agent; session_id="custom-id")
        @test session2.id == "custom-id"
    end

    @testset "Agent show" begin
        client = MockChatClient(ChatResponse())
        add_tool = FunctionTool(name="add", description="Add", func=identity)
        agent = Agent(name="Bot", client=client, tools=[add_tool])
        s = sprint(show, agent)
        @test contains(s, "Bot")
        @test contains(s, "1 tools")
    end
end
