using AgentFramework
using AgentFramework.Hosting
using HTTP
using JSON3
using Test

mutable struct HostingMockChatClient <: AbstractChatClient
    responses::Vector{String}
    call_count::Int
    received_messages::Vector{Vector{Message}}
end

HostingMockChatClient(responses::Vector{String}) = HostingMockChatClient(responses, 0, Vector{Vector{Message}}())
HostingMockChatClient(response::String) = HostingMockChatClient([response])

function AgentFramework.get_response(client::HostingMockChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count += 1
    push!(client.received_messages, deepcopy(messages))
    idx = min(client.call_count, length(client.responses))
    return ChatResponse(
        messages = [Message(:assistant, client.responses[idx])],
        finish_reason = STOP,
        model_id = "hosting-mock",
    )
end

function AgentFramework.get_response_streaming(client::HostingMockChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
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

function make_hosted_agent(name::String, responses::Vector{String})
    client = HostingMockChatClient(responses)
    agent = Agent(name = name, instructions = "Be concise.", client = client)
    return agent, client
end

make_hosted_agent(name::String, response::String) = make_hosted_agent(name, [response])

function make_hil_workflow(name::String)
    asker = ExecutorSpec(id = "asker", handler = (msg, ctx) -> begin
        if msg isa Dict && haskey(msg, "answer")
            yield_output(ctx, "Got: $(msg["answer"])")
        else
            request_info(ctx, "What is your name?"; request_id = "name_req")
        end
    end)

    builder = WorkflowBuilder(name = name, start = asker)
    add_output(builder, "asker")
    return build(builder)
end

parse_json_response(response::HTTP.Response) = Dict{String, Any}(Hosting._materialize_json(JSON3.read(String(response.body))))

@testset "Hosting.jl" begin
    @testset "Hosted agent runtime persists sessions and history" begin
        runtime = HostedRuntime()
        agent, _ = make_hosted_agent("assistant", ["Hello there.", "Welcome back."])
        register_agent!(runtime, agent)

        first = run_agent!(runtime, "assistant", "Hello")
        @test AgentFramework.get_text(first.response) == "Hello there."
        @test first.session.metadata["agent_name"] == "assistant"
        @test length(first.history) == 2

        stored_agent = runtime.agents["assistant"]
        second = run_agent!(runtime, "assistant", "Again"; session_id = first.session.id)
        @test AgentFramework.get_text(second.response) == "Welcome back."
        @test length(stored_agent.client.received_messages) == 2
        @test length(stored_agent.client.received_messages[2]) == 4
        @test stored_agent.client.received_messages[2][1].role == :system
        @test get_text(stored_agent.client.received_messages[2][1]) == "Be concise."
        @test get_text(stored_agent.client.received_messages[2][2]) == "Hello"
        @test get_text(stored_agent.client.received_messages[2][3]) == "Hello there."
        @test get_text(stored_agent.client.received_messages[2][4]) == "Again"

        session_info = get_agent_session(runtime, "assistant", first.session.id)
        @test length(session_info.history) == 4
        @test get_text(session_info.history[end]) == "Welcome back."

        sessions = list_agent_sessions(runtime, "assistant")
        @test length(sessions) == 1
        @test sessions[1].id == first.session.id

        @test delete_agent_session!(runtime, "assistant", first.session.id)
        @test list_agent_sessions(runtime, "assistant") == AgentSession[]
    end

    @testset "Hosted workflow runs track checkpoints and resume" begin
        runtime = HostedRuntime()
        workflow = make_hil_workflow("hil")
        register_workflow!(runtime, workflow)

        first = start_workflow_run!(runtime, "hil", "start")
        @test first.state == WF_IDLE_WITH_PENDING_REQUESTS
        @test first.checkpoint_id !== nothing
        @test length(first.pending_requests) == 1
        @test first.pending_requests[1]["request_id"] == "name_req"

        stored = get_workflow_run(runtime, "hil", first.id)
        @test stored.internal_workflow_name == "hil::" * first.id

        resumed = resume_workflow_run!(runtime, "hil", first.id, Dict("name_req" => Dict("answer" => "Alice")))
        @test resumed.state == WF_IDLE
        @test resumed.outputs == Any["Got: Alice"]
        @test isempty(resumed.pending_requests)

        runs = list_workflow_runs(runtime, "hil")
        @test length(runs) == 1
        @test runs[1].outputs == Any["Got: Alice"]
    end

    @testset "File-backed runtime restores sessions, history, and workflow records" begin
        mktempdir() do directory
            runtime = HostedRuntime(directory)
            agent, _ = make_hosted_agent("assistant", "Persisted reply.")
            workflow = make_hil_workflow("hil")
            register_agent!(runtime, agent)
            register_workflow!(runtime, workflow)

            run = run_agent!(runtime, "assistant", "Persist me")
            wf_run = start_workflow_run!(runtime, "hil", "start")

            restored = HostedRuntime(directory)
            register_agent!(restored, make_hosted_agent("assistant", "Restored reply.")[1])
            register_workflow!(restored, make_hil_workflow("hil"))

            session_info = get_agent_session(restored, "assistant", run.session.id)
            @test length(session_info.history) == 2
            @test get_text(session_info.history[1]) == "Persist me"

            stored_run = get_workflow_run(restored, "hil", wf_run.id)
            @test stored_run.state == WF_IDLE_WITH_PENDING_REQUESTS
            @test stored_run.checkpoint_id == wf_run.checkpoint_id
        end
    end

    @testset "HTTP handler exposes hosted runtime routes" begin
        runtime = HostedRuntime()
        agent, _ = make_hosted_agent("assistant", ["Hello from HTTP.", "Resumed HTTP."])
        workflow = make_hil_workflow("hil")
        register_agent!(runtime, agent)
        register_workflow!(runtime, workflow)

        health = handle_request(runtime, HTTP.Request("GET", "/health"))
        @test health.status == 200
        @test parse_json_response(health)["status"] == "ok"

        run_request = HTTP.Request(
            "POST",
            "/agents/assistant/run",
            ["Content-Type" => "application/json"],
            JSON3.write(Dict("message" => "Hello HTTP")),
        )
        run_response = handle_request(runtime, run_request)
        @test run_response.status == 200
        run_payload = parse_json_response(run_response)
        @test run_payload["response"]["text"] == "Hello from HTTP."
        session_id = run_payload["session"]["id"]

        session_response = handle_request(runtime, HTTP.Request("GET", "/agents/assistant/sessions/" * session_id))
        @test session_response.status == 200
        session_payload = parse_json_response(session_response)
        @test length(session_payload["history"]) == 2

        workflow_start = handle_request(
            runtime,
            HTTP.Request(
                "POST",
                "/workflows/hil/runs",
                ["Content-Type" => "application/json"],
                JSON3.write(Dict("input" => "start")),
            ),
        )
        @test workflow_start.status == 200
        workflow_payload = parse_json_response(workflow_start)
        @test workflow_payload["state"] == "WF_IDLE_WITH_PENDING_REQUESTS"

        resume_response = handle_request(
            runtime,
            HTTP.Request(
                "POST",
                "/workflows/hil/runs/" * workflow_payload["id"] * "/resume",
                ["Content-Type" => "application/json"],
                JSON3.write(Dict("responses" => Dict("name_req" => Dict("answer" => "Alice")))),
            ),
        )
        @test resume_response.status == 200
        resume_payload = parse_json_response(resume_response)
        @test resume_payload["outputs"] == Any["Got: Alice"]
    end
end
