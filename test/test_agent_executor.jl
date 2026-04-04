using AgentFramework
using Test

# ── Mock Chat Client ─────────────────────────────────────────────────────────

mutable struct AEMockChatClient <: AbstractChatClient
    responses::Vector{String}
    call_count::Ref{Int}
    received_messages::Vector{Vector{Message}}
end

AEMockChatClient(responses::Vector{String}) = AEMockChatClient(responses, Ref(0), Vector{Vector{Message}}())
AEMockChatClient(response::String) = AEMockChatClient([response])

function AgentFramework.get_response(client::AEMockChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count[] += 1
    push!(client.received_messages, copy(messages))
    idx = min(client.call_count[], length(client.responses))
    text = client.responses[idx]
    ChatResponse(
        messages = [Message(:assistant, text)],
        finish_reason = STOP,
        model_id = "mock-model",
    )
end

function AgentFramework.get_response_streaming(client::AEMockChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
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

# ── Helper to create a test agent ────────────────────────────────────────────

function make_test_agent(responses::Vector{String})
    client = AEMockChatClient(responses)
    agent = Agent(name="TestAgent", instructions="Be helpful.", client=client)
    return agent, client
end

function make_test_agent(response::String)
    make_test_agent([response])
end

# ── Tests ────────────────────────────────────────────────────────────────────

@testset "AgentExecutor" begin

    @testset "AgentExecutorRequest construction with defaults" begin
        req = AgentExecutorRequest()
        @test isempty(req.messages)
        @test req.should_respond == true
        @test isempty(req.metadata)

        req2 = AgentExecutorRequest(
            messages = [Message(:user, "Hello")],
            should_respond = false,
            metadata = Dict{String, Any}("key" => "value"),
        )
        @test length(req2.messages) == 1
        @test req2.should_respond == false
        @test req2.metadata["key"] == "value"
    end

    @testset "AgentExecutorResponse construction" begin
        ar = AgentResponse(
            messages = [Message(:assistant, "Hi")],
            finish_reason = STOP,
        )
        resp = AgentExecutorResponse(
            agent_response = ar,
            full_conversation = [Message(:user, "Hello"), Message(:assistant, "Hi")],
            executor_id = "exec-1",
        )
        @test resp.agent_response === ar
        @test length(resp.full_conversation) == 2
        @test resp.executor_id == "exec-1"
        @test isempty(resp.metadata)

        resp2 = AgentExecutorResponse(
            agent_response = ar,
            full_conversation = Message[],
            executor_id = "exec-2",
            metadata = Dict{String, Any}("run" => 1),
        )
        @test resp2.metadata["run"] == 1
    end

    @testset "AgentExecutor construction and defaults" begin
        agent, _ = make_test_agent("ok")
        ae = AgentExecutor("test-exec", agent)
        @test ae.id == "test-exec"
        @test ae.agent === agent
        @test ae.yield_response == false
        @test ae.forward_response == true
        @test ae.auto_respond == true
        @test isempty(ae._message_cache)
        @test isempty(ae._full_conversation)
        @test !isempty(ae.session.id)

        # With custom kwargs
        session = AgentSession(id="custom-session")
        ae2 = AgentExecutor("test-exec-2", agent;
            session=session, yield_response=true, forward_response=false, auto_respond=false)
        @test ae2.session.id == "custom-session"
        @test ae2.yield_response == true
        @test ae2.forward_response == false
        @test ae2.auto_respond == false
    end

    @testset "to_executor_spec creates valid ExecutorSpec" begin
        agent, _ = make_test_agent("ok")
        ae = AgentExecutor("spec-test", agent; yield_response=true)
        spec = to_executor_spec(ae)

        @test spec.id == "spec-test"
        @test contains(spec.description, "TestAgent")
        @test String ∈ spec.input_types
        @test Message ∈ spec.input_types
        @test AgentExecutorRequest ∈ spec.input_types
        @test AgentExecutorResponse ∈ spec.input_types
        @test AgentExecutorResponse ∈ spec.output_types
        @test AgentExecutorResponse ∈ spec.yield_types

        # Without yield
        ae2 = AgentExecutor("spec-test-2", agent; yield_response=false)
        spec2 = to_executor_spec(ae2)
        @test isempty(spec2.yield_types)
    end

    @testset "handle_string! processes string input" begin
        agent, client = make_test_agent("Hello back!")
        ae = AgentExecutor("str-test", agent; forward_response=false, yield_response=false)
        ctx = WorkflowContext(executor_id="str-test")

        handle_string!(ae, "Hello", ctx)

        @test client.call_count[] == 1
        @test length(ae._full_conversation) == 2  # user + assistant
        @test ae._full_conversation[1].role == :user
        @test get_text(ae._full_conversation[1]) == "Hello"
        @test ae._full_conversation[2].role == :assistant
        @test get_text(ae._full_conversation[2]) == "Hello back!"
        @test isempty(ae._message_cache)  # cleared after run
    end

    @testset "handle_message! processes Message input" begin
        agent, client = make_test_agent("Got it")
        ae = AgentExecutor("msg-test", agent; forward_response=false, yield_response=false)
        ctx = WorkflowContext(executor_id="msg-test")

        msg = Message(:user, "Custom message")
        handle_message!(ae, msg, ctx)

        @test client.call_count[] == 1
        @test length(ae._full_conversation) == 2
        @test ae._full_conversation[1].role == :user
        @test get_text(ae._full_conversation[1]) == "Custom message"
    end

    @testset "handle_request! with should_respond=true triggers agent run" begin
        agent, client = make_test_agent("Responding")
        ae = AgentExecutor("req-true", agent; forward_response=false)
        ctx = WorkflowContext(executor_id="req-true")

        req = AgentExecutorRequest(
            messages = [Message(:user, "Please respond")],
            should_respond = true,
        )
        handle_request!(ae, req, ctx)

        @test client.call_count[] == 1
        @test length(ae._full_conversation) == 2
    end

    @testset "handle_request! with should_respond=false only caches" begin
        agent, client = make_test_agent("Should not see")
        ae = AgentExecutor("req-false", agent; forward_response=false)
        ctx = WorkflowContext(executor_id="req-false")

        req = AgentExecutorRequest(
            messages = [Message(:user, "Just cache this")],
            should_respond = false,
        )
        handle_request!(ae, req, ctx)

        @test client.call_count[] == 0
        @test length(ae._message_cache) == 1  # cached but not consumed
        @test isempty(ae._full_conversation)
    end

    @testset "handle_response! chains from prior AgentExecutorResponse" begin
        agent, client = make_test_agent("Chained response")
        ae = AgentExecutor("chain-test", agent; forward_response=false)
        ctx = WorkflowContext(executor_id="chain-test")

        prior_conv = [Message(:user, "Hi"), Message(:assistant, "Hello")]
        prior_resp = AgentExecutorResponse(
            agent_response = AgentResponse(messages=[Message(:assistant, "Hello")], finish_reason=STOP),
            full_conversation = prior_conv,
            executor_id = "other-exec",
        )

        handle_response!(ae, prior_resp, ctx)

        @test client.call_count[] == 1
        # Full conversation should have prior conversation + new response
        @test length(ae._full_conversation) >= 3
    end

    @testset "Session persistence across calls" begin
        agent, client = make_test_agent(["First", "Second"])
        session = AgentSession(id="persistent-session")
        ae = AgentExecutor("session-test", agent; session=session, forward_response=false)

        ctx1 = WorkflowContext(executor_id="session-test")
        handle_string!(ae, "Call 1", ctx1)
        session_id_after_first = ae.session.id

        ctx2 = WorkflowContext(executor_id="session-test")
        handle_string!(ae, "Call 2", ctx2)
        session_id_after_second = ae.session.id

        @test session_id_after_first == "persistent-session"
        @test session_id_after_second == "persistent-session"
        @test ae.session === session
    end

    @testset "Message cache accumulates across calls" begin
        agent, client = make_test_agent("ok")
        ae = AgentExecutor("cache-test", agent; forward_response=false)
        ctx = WorkflowContext(executor_id="cache-test")

        # Add messages without responding
        req1 = AgentExecutorRequest(messages=[Message(:user, "msg1")], should_respond=false)
        req2 = AgentExecutorRequest(messages=[Message(:user, "msg2")], should_respond=false)
        handle_request!(ae, req1, ctx)
        handle_request!(ae, req2, ctx)

        @test length(ae._message_cache) == 2

        # Now trigger respond
        req3 = AgentExecutorRequest(messages=[Message(:user, "msg3")], should_respond=true)
        handle_request!(ae, req3, ctx)

        @test client.call_count[] == 1
        # Agent received all 3 messages
        @test length(client.received_messages[1]) >= 3  # includes system instructions
        @test isempty(ae._message_cache)  # cleared after run
    end

    @testset "reset! clears cache and conversation" begin
        agent, client = make_test_agent("ok")
        ae = AgentExecutor("reset-test", agent; forward_response=false)
        ctx = WorkflowContext(executor_id="reset-test")

        handle_string!(ae, "Hello", ctx)
        @test !isempty(ae._full_conversation)
        old_session_id = ae.session.id

        reset!(ae)
        @test isempty(ae._message_cache)
        @test isempty(ae._full_conversation)
        @test ae.session.id != old_session_id  # new session
    end

    @testset "get_conversation returns full history" begin
        agent, _ = make_test_agent(["Reply 1", "Reply 2"])
        ae = AgentExecutor("conv-test", agent; forward_response=false)

        ctx1 = WorkflowContext(executor_id="conv-test")
        handle_string!(ae, "First", ctx1)

        ctx2 = WorkflowContext(executor_id="conv-test")
        handle_string!(ae, "Second", ctx2)

        conv = get_conversation(ae)
        @test length(conv) == 4  # user1, assistant1, user2, assistant2
        @test conv[1].role == :user
        @test conv[2].role == :assistant
        @test conv[3].role == :user
        @test conv[4].role == :assistant

        # Returns a copy
        push!(conv, Message(:user, "extra"))
        @test length(get_conversation(ae)) == 4
    end

    @testset "forward_response=true sends message to workflow context" begin
        agent, _ = make_test_agent("Forwarded")
        ae = AgentExecutor("fwd-test", agent; forward_response=true, yield_response=false)
        ctx = WorkflowContext(executor_id="fwd-test")

        handle_string!(ae, "Hello", ctx)

        @test length(ctx._sent_messages) == 1
        sent = ctx._sent_messages[1]
        @test sent.data isa AgentExecutorResponse
        @test sent.data.executor_id == "fwd-test"
        @test get_text(sent.data.agent_response) == "Forwarded"
        @test isempty(ctx._yielded_outputs)
    end

    @testset "yield_response=true yields output to workflow context" begin
        agent, _ = make_test_agent("Yielded")
        ae = AgentExecutor("yield-test", agent; forward_response=false, yield_response=true)
        ctx = WorkflowContext(executor_id="yield-test")

        handle_string!(ae, "Hello", ctx)

        @test length(ctx._yielded_outputs) == 1
        yielded = ctx._yielded_outputs[1]
        @test yielded isa AgentExecutorResponse
        @test yielded.executor_id == "yield-test"
        @test get_text(yielded.agent_response) == "Yielded"
        @test isempty(ctx._sent_messages)
    end

    @testset "to_executor_spec handler dispatches correctly" begin
        agent, client = make_test_agent(["str-reply", "msg-reply", "req-reply", "resp-reply"])
        ae = AgentExecutor("dispatch-test", agent; forward_response=false)
        spec = to_executor_spec(ae)

        # String dispatch
        ctx1 = WorkflowContext(executor_id="dispatch-test")
        spec.handler("hello string", ctx1)
        @test client.call_count[] == 1

        # Message dispatch
        ctx2 = WorkflowContext(executor_id="dispatch-test")
        spec.handler(Message(:user, "hello message"), ctx2)
        @test client.call_count[] == 2

        # AgentExecutorRequest dispatch
        ctx3 = WorkflowContext(executor_id="dispatch-test")
        spec.handler(AgentExecutorRequest(messages=[Message(:user, "hello req")]), ctx3)
        @test client.call_count[] == 3

        # AgentExecutorResponse dispatch
        ctx4 = WorkflowContext(executor_id="dispatch-test")
        prior = AgentExecutorResponse(
            agent_response = AgentResponse(messages=[Message(:assistant, "prior")], finish_reason=STOP),
            full_conversation = [Message(:user, "x"), Message(:assistant, "prior")],
            executor_id = "other",
        )
        spec.handler(prior, ctx4)
        @test client.call_count[] == 4
    end

    @testset "Integration with WorkflowBuilder" begin
        agent, client = make_test_agent("workflow result")
        ae = AgentExecutor("agent-node", agent; yield_response=true, forward_response=false)
        spec = to_executor_spec(ae)

        workflow = WorkflowBuilder(name="AEWorkflow", start=spec) |>
            b -> add_output(b, "agent-node") |>
            build

        result = run_workflow(workflow, "Test input")
        @test result.state == WF_IDLE
        outputs = get_outputs(result)
        @test length(outputs) == 1
        @test outputs[1] isa AgentExecutorResponse
        @test get_text(outputs[1].agent_response) == "workflow result"
    end

end
