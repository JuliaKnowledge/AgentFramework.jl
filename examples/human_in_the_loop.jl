#!/usr/bin/env julia
# =============================================================================
# Human-in-the-loop: function-middleware tool approvals
# =============================================================================
#
# AgentFramework has a three-layer middleware pipeline:
#
#     Agent → Chat → Function(Tool)
#
# The innermost (function) layer wraps *every* tool invocation and can:
#   * inspect arguments before the tool runs
#   * short-circuit with `terminate_pipeline(result)`  (the tool never runs)
#   * mutate the result after the tool runs
#
# This vignette uses a function middleware to gate a **dangerous** tool
# (`delete_file`) behind a human approval prompt.  The harmless tool
# (`list_files`) runs without interruption.
#
# The approval callback is pluggable — here we inject a scripted
# "approver" (so the test is deterministic), but a real app could swap
# in `readline()` on stdin, a webhook, a Slack round-trip, etc.
#
# Requires OLLAMA_HOST reachable + qwen3:8b.
# -----------------------------------------------------------------------------

using Test
using AgentFramework

ollama_host = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
client = OllamaChatClient(model="qwen3:8b", base_url=ollama_host)

# ── State: fake filesystem + audit log ───────────────────────────────────────

const FS = Ref{Vector{String}}(["report.md", "notes.txt", "archive/old.log"])
const AUDIT = Ref{Vector{String}}(String[])

function list_files_tool()
    push!(AUDIT[], "list_files()")
    return "Files: " * join(FS[], ", ")
end

function delete_file_tool(path::AbstractString)
    p = String(path)
    push!(AUDIT[], "delete_file($p)")
    filter!(!=(p), FS[])
    return "Deleted $p. Remaining: " * join(FS[], ", ")
end

list_tool = FunctionTool(
    name = "list_files",
    description = "List all files in the project.",
    func = list_files_tool,
    parameters = Dict{String,Any}("type"=>"object","properties"=>Dict{String,Any}()),
)

delete_tool = FunctionTool(
    name = "delete_file",
    description = "Permanently delete a file by path.",
    func = delete_file_tool,
    parameters = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "path" => Dict{String,Any}("type"=>"string","description"=>"File path to delete."),
        ),
        "required" => ["path"],
    ),
)

# ── The approval middleware ──────────────────────────────────────────────────
#
# `gated_tools` is the set of tool names that require approval.  The
# `approver` callback returns `true` to allow, `false` to deny.
# Denied calls short-circuit with `terminate_pipeline`, which returns a
# user-visible rejection string *as if* it were the tool's output — the
# LLM then sees the rejection on its next turn and can respond to it.

function approval_middleware(gated_tools::Set{String}, approver)
    return function(ctx, call_next)
        name = ctx.tool === nothing ? "" : ctx.tool.name
        if name in gated_tools
            ok = approver(name, ctx.arguments)
            push!(AUDIT[], "approval($name,$(ctx.arguments))=$ok")
            if !ok
                terminate_pipeline(
                    "DENIED: the human operator rejected the call to $name.";
                    message = "user rejected $name",
                )
            end
        end
        return call_next(ctx)
    end
end

# Scripted approver: approve archive/*, deny anything else.
function scripted_approver(name, args)
    path = get(args, "path", "")
    return startswith(path, "archive/")
end

agent = Agent(
    client = client,
    instructions = "You are a file-management assistant.  Use the tools to " *
                   "help the user.  If a deletion is denied, acknowledge it " *
                   "politely and suggest alternatives.",
    tools = [list_tool, delete_tool],
    function_middlewares = Function[
        approval_middleware(Set(["delete_file"]), scripted_approver),
    ],
)

# ── Scenario 1: deletion that the approver ALLOWS (archive/*) ────────────────

println("\n── Scenario 1: approved deletion (archive/*) ─────────────────")
resp1 = run_agent(agent, "Please delete archive/old.log for me.")
println("Assistant: ", strip(resp1.text))

# ── Scenario 2: deletion that the approver DENIES ────────────────────────────

println("\n── Scenario 2: rejected deletion (report.md) ─────────────────")
resp2 = run_agent(agent, "Please delete report.md.")
println("Assistant: ", strip(resp2.text))

# ── Assertions ───────────────────────────────────────────────────────────────

@testset "Tool approval middleware" begin
    @test "archive/old.log" ∉ FS[]
    @test "report.md" in FS[]

    @test any(occursin("approval(delete_file", s) && endswith(s, "=true")  for s in AUDIT[])
    @test any(occursin("approval(delete_file", s) && endswith(s, "=false") for s in AUDIT[])

    @test any(startswith(s, "delete_file(archive/") for s in AUDIT[])
    @test !any(startswith(s, "delete_file(report.md)") for s in AUDIT[])

    @test occursin(r"deni|reject|cannot|can't|unable|not allowed"i, resp2.text)
end

println("\nAudit log:")
foreach(e -> println("  • ", e), AUDIT[])
println("\nPASS — human-in-the-loop tool approval vignette")
