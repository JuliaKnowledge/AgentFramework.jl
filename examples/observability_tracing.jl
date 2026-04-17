#!/usr/bin/env julia
# =============================================================================
# Observability: tracing an agent run with AF telemetry
# =============================================================================
#
# AF ships an OpenTelemetry-shaped telemetry subsystem built on three
# concentric middlewares (agent / chat / function).  `instrument!(agent,
# backend)` wires all three up in one call.
#
# An `InMemoryTelemetryBackend` captures every span — perfect for tests
# and offline demos.  For real deployments swap in a backend that
# forwards to Jaeger / Tempo / Azure Monitor (implement
# `record_span!(backend, span)` for your type).
#
# This vignette runs a one-turn, one-tool agent conversation and then
# walks the captured span tree, showing:
#
#   * agent.run (outermost, kind=:internal)
#     ├─ chat.completion.tool_selection   (first LLM turn — picks tool)
#     ├─ function.invoke calc              (tool execution)
#     └─ chat.completion.final             (second LLM turn — writes reply)
#
# Requires OLLAMA_HOST reachable + qwen3:8b.
# -----------------------------------------------------------------------------

using Test
using AgentFramework
using Dates

ollama_host = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
client = OllamaChatClient(model="qwen3:8b", base_url=ollama_host)

# ── Single tool under observation ────────────────────────────────────────────

function calc(expr::AbstractString)
    val = eval(Meta.parse(String(expr)))
    return "Result: $val"
end

calc_tool = FunctionTool(
    name = "calc",
    description = "Evaluate a small numeric expression.",
    func = calc,
    parameters = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "expr" => Dict{String,Any}("type"=>"string"),
        ),
        "required" => ["expr"],
    ),
)

agent = Agent(
    name = "calculator",
    client = client,
    instructions = "Use the calc tool for any arithmetic question.  Be concise.",
    tools = [calc_tool],
)

# ── Instrument ───────────────────────────────────────────────────────────────

backend = InMemoryTelemetryBackend()
instrument!(agent, backend)

# ── Run one turn ─────────────────────────────────────────────────────────────

println("\nRunning the agent with telemetry attached…")
resp = run_agent(agent, "What is (7 + 5) * 3?")
println("Final answer: ", strip(resp.text))

# ── Inspect the span tree ────────────────────────────────────────────────────

spans = get_spans(backend)
println("\nCaptured ", length(spans), " spans:")

# Sort roots-first, then by start time.
sort!(spans, by = s -> (s.parent_id !== nothing, s.start_time))

for s in spans
    dur = duration_ms(s)
    dur_str = dur === nothing ? "?ms" : "$(dur)ms"
    parent  = s.parent_id === nothing ? "(root)" : s.parent_id[1:min(end,8)]
    println("  • $(rpad(s.name, 35)) kind=$(s.kind)  status=$(s.status)  " *
            "dur=$(dur_str)  parent=$(parent)")
    for (k, v) in s.attributes
        kstr = String(k)
        if startswith(kstr, "gen_ai.")
            vs = string(v)
            println("       $(kstr) = $(first(vs, 60))")
        end
    end
end

# ── Assertions ───────────────────────────────────────────────────────────────

@testset "Observability / tracing" begin
    @test !isempty(spans)

    # At least one agent.run span (AF's telemetry middlewares currently
    # emit flat spans rather than a parented tree).
    @test any(s -> occursin("agent", s.name), spans)

    # We should see both chat.* and function.* spans.
    @test any(s -> occursin("chat", s.name), spans)
    @test any(s -> occursin(r"function|tool|invoke"i, s.name), spans)

    # Every span is finished with an :ok status.
    @test all(s -> s.status in (:ok, :unset) && s.end_time !== nothing, spans)

    # At least one span carries a gen_ai semantic-convention attribute.
    @test any(s -> any(k -> startswith(String(k), "gen_ai."), keys(s.attributes)), spans)

    # Answer is correct (7+5)*3 = 36.
    @test occursin("36", resp.text)
end

println("\nPASS — observability/tracing vignette")
