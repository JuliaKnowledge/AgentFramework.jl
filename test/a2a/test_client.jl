@testset "A2AClient" begin
    @testset "get_agent_card loads agent discovery document" begin
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            @test String(request.target) == "/.well-known/agent.json"
            return HTTP.Response(
                200,
                ["Content-Type" => "application/json"],
                JSON3.write(
                    Dict{String, Any}(
                        "name" => "Remote Helper",
                        "description" => "A remote A2A agent",
                        "url" => "http://127.0.0.1:$(HTTP.port(server))",
                        "defaultInputModes" => ["text/plain"],
                        "defaultOutputModes" => ["text/plain"],
                    ),
                ),
            )
        end

        try
            client = A2AClient(base_url = "http://127.0.0.1:$(HTTP.port(server))")
            card = get_agent_card(client)
            @test card.name == "Remote Helper"
            @test card.description == "A remote A2A agent"
            @test card.url == "http://127.0.0.1:$(HTTP.port(server))"
            @test card.default_input_modes == ["text/plain"]
        finally
            close(server)
        end
    end

    @testset "send_message returns immediate message responses" begin
        requests = Channel{Dict{String, Any}}(1)
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            payload = JSON3.read(String(request.body), Dict{String, Any})
            put!(requests, payload)
            return HTTP.Response(
                200,
                ["Content-Type" => "application/json"],
                JSON3.write(
                    Dict{String, Any}(
                        "jsonrpc" => "2.0",
                        "id" => payload["id"],
                        "result" => Dict{String, Any}(
                            "kind" => "message",
                            "role" => "agent",
                            "messageId" => "msg-123",
                            "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "Hello from A2A")],
                        ),
                    ),
                ),
            )
        end

        try
            client = A2AClient(base_url = "http://127.0.0.1:$(HTTP.port(server))")
            response = A2A.send_message(client, Message(:user, "Hello"))
            request = take!(requests)
            @test request["method"] == "message/send"
            @test request["params"]["message"]["parts"][1]["text"] == "Hello"
            @test response.response_id == "msg-123"
            @test response.finish_reason == STOP
            @test response.text == "Hello from A2A"
        finally
            close(server)
        end
    end

    @testset "background tasks can be polled to completion" begin
        polls = Ref(0)
        listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
        server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
            payload = JSON3.read(String(request.body), Dict{String, Any})
            if payload["method"] == "message/send"
                return HTTP.Response(
                    200,
                    ["Content-Type" => "application/json"],
                    JSON3.write(
                        Dict{String, Any}(
                            "jsonrpc" => "2.0",
                            "id" => payload["id"],
                            "result" => Dict{String, Any}(
                                "kind" => "task",
                                "id" => "task-1",
                                "contextId" => "ctx-1",
                                "status" => Dict{String, Any}(
                                    "state" => "working",
                                    "message" => Dict{String, Any}(
                                        "kind" => "message",
                                        "role" => "agent",
                                        "messageId" => "work-1",
                                        "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "Working...")],
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            end

            polls[] += 1
            @test payload["method"] == "tasks/get"
            state = polls[] == 1 ? "working" : "completed"
            result = if state == "working"
                Dict{String, Any}(
                    "kind" => "task",
                    "id" => "task-1",
                    "contextId" => "ctx-1",
                    "status" => Dict{String, Any}("state" => "working"),
                )
            else
                Dict{String, Any}(
                    "kind" => "task",
                    "id" => "task-1",
                    "contextId" => "ctx-1",
                    "status" => Dict{String, Any}("state" => "completed"),
                    "artifacts" => Any[
                        Dict{String, Any}(
                            "artifactId" => "art-1",
                            "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "Done")],
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
            client = A2AClient(base_url = "http://127.0.0.1:$(HTTP.port(server))", poll_interval = 0.01)
            response = A2A.send_message(client, Message(:user, "Start"); background = true)
            @test response.continuation_token isa A2AContinuationToken
            @test response.response_id == "task-1"
            @test response.text == "Working..."

            final = A2A.wait_for_completion(client, response.continuation_token; poll_interval = 0.01, max_polls = 5)
            @test final.continuation_token === nothing
            @test final.finish_reason == STOP
            @test final.text == "Done"
        finally
            close(server)
        end
    end
end
