@testset "A2ARemoteAgent" begin
    @testset "run_agent reuses session context and reference task ids" begin
        requests = Channel{Dict{String, Any}}(2)
        turns = Ref(0)
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            payload = JSON3.read(String(request.body), Dict{String, Any})
            put!(requests, payload)
            @test payload["method"] == "message/send"

            turns[] += 1
            if turns[] == 1
                result = Dict{String, Any}(
                    "kind" => "task",
                    "id" => "task-1",
                    "contextId" => "ctx-1",
                    "status" => Dict{String, Any}("state" => "completed"),
                    "artifacts" => Any[
                        Dict{String, Any}(
                            "artifactId" => "art-1",
                            "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "First result")],
                        ),
                    ],
                )
            else
                result = Dict{String, Any}(
                    "kind" => "message",
                    "role" => "agent",
                    "messageId" => "msg-2",
                    "contextId" => "ctx-1",
                    "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "Follow-up response")],
                )
            end

            return HTTP.Response(
                200,
                ["Content-Type" => "application/json"],
                JSON3.write(Dict{String, Any}("jsonrpc" => "2.0", "id" => payload["id"], "result" => result)),
            )
        end

        try
            agent = A2ARemoteAgent(url = "http://127.0.0.1:$(HTTP.port(server))", name = "Remote")
            session = create_session(agent)

            first = run_agent(agent, "Start"; session = session)
            second = run_agent(agent, "Follow up"; session = session)

            first_request = take!(requests)
            second_request = take!(requests)

            @test first.text == "First result"
            @test second.text == "Follow-up response"
            @test session.state["__a2a_context_id__"] == "ctx-1"
            @test session.state["__a2a_task_id__"] == "task-1"
            @test !haskey(first_request["params"]["message"], "referenceTaskIds")
            @test second_request["params"]["message"]["contextId"] == "ctx-1"
            @test second_request["params"]["message"]["referenceTaskIds"] == Any["task-1"]
        finally
            close(server)
        end
    end

    @testset "run_agent_streaming polls background tasks" begin
        polls = Ref(0)
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            payload = JSON3.read(String(request.body), Dict{String, Any})
            if payload["method"] == "message/send"
                result = Dict{String, Any}(
                    "kind" => "task",
                    "id" => "task-stream",
                    "contextId" => "ctx-stream",
                    "status" => Dict{String, Any}(
                        "state" => "working",
                        "message" => Dict{String, Any}(
                            "kind" => "message",
                            "role" => "agent",
                            "messageId" => "msg-working",
                            "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "Working...")],
                        ),
                    ),
                )
            else
                polls[] += 1
                result = Dict{String, Any}(
                    "kind" => "task",
                    "id" => "task-stream",
                    "contextId" => "ctx-stream",
                    "status" => Dict{String, Any}("state" => "completed"),
                    "artifacts" => Any[
                        Dict{String, Any}(
                            "artifactId" => "art-stream",
                            "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "Done streaming")],
                        ),
                    ],
                )
            end

            return HTTP.Response(
                200,
                ["Content-Type" => "application/json"],
                JSON3.write(Dict{String, Any}("jsonrpc" => "2.0", "id" => payload["id"], "result" => result)),
            )
        end

        try
            agent = A2ARemoteAgent(url = "http://127.0.0.1:$(HTTP.port(server))", name = "Remote", poll_interval = 0.01, max_polls = 5)
            stream = run_agent_streaming(agent, "Stream")
            updates = AgentResponseUpdate[]
            for update in stream
                push!(updates, update)
            end
            final = get_final_response(stream)

            @test length(updates) == 2
            @test updates[1].text == "Working..."
            @test updates[1].continuation_token isa A2AContinuationToken
            @test updates[2].text == "Done streaming"
            @test updates[2].finish_reason == STOP
            @test final.text == "Done streaming"
            @test final.finish_reason == STOP
        finally
            close(server)
        end
    end
end
