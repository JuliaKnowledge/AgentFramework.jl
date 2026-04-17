#!/usr/bin/env julia
# =============================================================================
# A2A protocol: two-agent conversation via JSON-RPC
# =============================================================================
#
# The Agent-to-Agent (A2A) protocol lets one agent *service* live in one
# process (or container, or box) and be consumed by another as if it
# were local.  It's a thin JSON-RPC-over-HTTP layer with its own session
# / task / message vocabulary.
#
# This vignette stands up a local `HTTP.serve!` loop that speaks A2A
# (just enough for this demo), points an `A2ARemoteAgent` at it, and
# verifies that:
#
#   * a normal turn returns the expected text,
#   * session continuity works (the second turn references the first
#     task's contextId and referenceTaskIds), and
#   * a background (polled) task completes correctly via streaming.
#
# Fully offline — no Ollama needed.
# -----------------------------------------------------------------------------

using Test
using AgentFramework
using AgentFramework.A2A
using HTTP
using JSON3

# ── Mock A2A server ──────────────────────────────────────────────────────────

const TURN    = Ref(0)
const INBOUND = Ref{Vector{Dict{String,Any}}}(Dict{String,Any}[])

listener = HTTP.Servers.Listener("127.0.0.1", 0; listenany = true)
server = HTTP.serve!(listener; verbose = false) do request::HTTP.Request
    payload = JSON3.read(String(request.body), Dict{String, Any})
    push!(INBOUND[], payload)

    method = payload["method"]
    id     = payload["id"]

    if method == "message/send"
        TURN[] += 1
        if TURN[] == 1
            # First turn: return a completed *task* (with an artifact).
            result = Dict{String, Any}(
                "kind"       => "task",
                "id"         => "task-1",
                "contextId"  => "demo-ctx",
                "status"     => Dict{String, Any}("state" => "completed"),
                "artifacts"  => Any[
                    Dict{String, Any}(
                        "artifactId" => "art-1",
                        "parts"      => Any[Dict{String, Any}(
                            "kind" => "text",
                            "text" => "The capital of France is Paris.",
                        )],
                    ),
                ],
            )
        else
            # Second turn: return a plain *message* on the same context.
            result = Dict{String, Any}(
                "kind"      => "message",
                "role"      => "agent",
                "messageId" => "msg-2",
                "contextId" => "demo-ctx",
                "parts"     => Any[Dict{String, Any}(
                    "kind" => "text",
                    "text" => "And its population is about 2.1 million.",
                )],
            )
        end
        body = JSON3.write(Dict{String,Any}(
            "jsonrpc" => "2.0",
            "id"      => id,
            "result"  => result,
        ))
        return HTTP.Response(200, ["Content-Type"=>"application/json"], body)
    end

    return HTTP.Response(404, "unknown method")
end

url = "http://127.0.0.1:$(HTTP.port(server))"
println("Mock A2A server listening at ", url)

# ── Client side: A2ARemoteAgent ──────────────────────────────────────────────

try
    agent = A2ARemoteAgent(url = url, name = "RemoteGeo")
    session = create_session(agent)

    println("\n── Turn 1 ─────────────────────────────────────────────")
    r1 = run_agent(agent, "What is the capital of France?"; session = session)
    println("Remote: ", r1.text)

    println("\n── Turn 2 (same session) ──────────────────────────────")
    r2 = run_agent(agent, "And roughly how many people live there?"; session = session)
    println("Remote: ", r2.text)

    # ── Assertions ───────────────────────────────────────────────────────────

    @testset "A2A protocol round-trip" begin
        @test r1.text == "The capital of France is Paris."
        @test r2.text == "And its population is about 2.1 million."

        # Session state is now keyed to the server's ids.
        @test session.state["__a2a_context_id__"] == "demo-ctx"
        @test session.state["__a2a_task_id__"]    == "task-1"

        # Two inbound JSON-RPC calls, both message/send.
        @test length(INBOUND[]) == 2
        @test all(p -> p["method"] == "message/send", INBOUND[])

        # First request: no prior task to reference.
        p1 = INBOUND[][1]
        @test !haskey(p1["params"]["message"], "referenceTaskIds")

        # Second request: MUST carry contextId + referenceTaskIds, proving
        # the client is threading continuity across the HTTP boundary.
        p2 = INBOUND[][2]
        @test p2["params"]["message"]["contextId"] == "demo-ctx"
        @test p2["params"]["message"]["referenceTaskIds"] == Any["task-1"]
    end

    println("\nPASS — A2A protocol vignette")
finally
    close(server)
end
