#!/usr/bin/env julia
# =============================================================================
# LLMWiki-as-tool: exposing a Julia-powered wiki as an AgentFramework tool
# =============================================================================
#
# LLMWiki.jl maintains a version-controlled, LLM-compiled knowledge base out
# of source notes.  Its `search_wiki` function runs a BM25 index over the
# compiled concept pages and returns ranked `SearchResult`s (slug, title,
# score, snippet).
#
# This vignette:
#   1. Seeds a tiny wiki with two hand-written concept pages (so we skip the
#      LLM-powered compile step and keep the example offline-except-for-chat).
#   2. Wraps `search_wiki` in an AgentFramework `FunctionTool`.
#   3. Builds an agent and asks it a question whose answer only lives in the
#      wiki — the agent must call the tool.
#   4. Asserts the tool was called AND the answer contains the wiki fact.
#
# Requires:
#   * OLLAMA_HOST reachable (default http://localhost:11434)
#   * qwen3:8b (tool-capable)
# -----------------------------------------------------------------------------

using Test
using AgentFramework
using LLMWiki

# ── 1.  Set up a tiny wiki ───────────────────────────────────────────────────

wiki_root = mktempdir()
config = default_config(wiki_root)
init_wiki(config)

# Seed two concept pages directly (bypassing the LLM compile pipeline).
write(joinpath(config.concepts_dir, "hermes-shell.md"), """---
title: "Hermes Shell"
page_type: concept
---
# Hermes Shell

The **Hermes Shell** is the project's internal command-line runtime.  It was
introduced in v2.3 and replaced the legacy Bash wrapper.  Hermes Shell uses
S-expressions for its configuration syntax and supports hot-reloading of
plugins without restarting the host process.
""")

write(joinpath(config.concepts_dir, "quorum-protocol.md"), """---
title: "Quorum Protocol"
page_type: concept
---
# Quorum Protocol

The **Quorum Protocol** is the distributed consensus algorithm used by the
storage layer.  It requires at least five voting nodes; the default
quorum-size is three, meaning an operation commits as soon as three of the
five nodes acknowledge it.  Quorum was benchmarked at roughly 12,000 ops/s.
""")

# Sanity check BM25 retrieval works standalone.
let hits = search_wiki(config, "hermes shell configuration"; method=:bm25, top_k=3)
    @info "Direct BM25 probe" n=length(hits) top=(isempty(hits) ? nothing : hits[1].slug)
    @assert !isempty(hits) "BM25 should find at least one page"
end

# ── 2.  Wrap search_wiki as a FunctionTool ───────────────────────────────────

const TOOL_CALL_LOG = Ref{Vector{String}}(String[])

function wiki_search_tool(query::AbstractString; top_k::Int=3)
    push!(TOOL_CALL_LOG[], String(query))
    hits = search_wiki(config, String(query); method=:bm25, top_k=top_k)
    if isempty(hits)
        return "No matching wiki pages."
    end
    buf = IOBuffer()
    for (i, h) in enumerate(hits)
        println(buf, "[$(i)] $(h.title) (slug=$(h.slug), score=$(round(h.score, digits=3)))")
        # SearchResult.snippet is empty for BM25 — load the concept page body.
        page_path = joinpath(config.concepts_dir, h.slug * ".md")
        if isfile(page_path)
            body = read(page_path, String)
            # Strip the YAML frontmatter so the LLM sees only prose.
            body = replace(body, r"^---\n.*?\n---\n"s => "")
            body = strip(body)
            println(buf, "    ", first(body, 600))
        end
    end
    String(take!(buf))
end

tool = FunctionTool(
    name = "wiki_search",
    description = "Search the project wiki for facts about internal systems. " *
                  "Call this whenever the user asks about named project concepts " *
                  "(e.g. Hermes Shell, Quorum Protocol). Returns ranked snippets.",
    func = wiki_search_tool,
    parameters = Dict{String, Any}(
        "type" => "object",
        "properties" => Dict{String, Any}(
            "query" => Dict{String, Any}(
                "type" => "string",
                "description" => "Search query — a few keywords or a short question.",
            ),
            "top_k" => Dict{String, Any}(
                "type" => "integer",
                "description" => "Maximum number of results to return (default 3).",
            ),
        ),
        "required" => ["query"],
    ),
)

# ── 3.  Build the agent ──────────────────────────────────────────────────────

ollama_host = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
client = OllamaChatClient(model = "qwen3:8b", base_url = ollama_host)

agent = Agent(
    client = client,
    instructions = """
        You are a project engineer.  When asked about internal project
        concepts, ALWAYS call the wiki_search tool first and ground your
        answer in the returned snippets.  Quote specific numbers and names
        verbatim from the wiki when they are present.
        """,
    tools = [tool],
)

# ── 4.  Ask something only answerable from the wiki ──────────────────────────

question = "What is the default quorum-size of the Quorum Protocol, " *
           "and roughly how many ops per second has it been benchmarked at?"
println("\nUser: ", question, "\n")

resp = run_agent(agent, question)
answer = resp.text
println("Assistant:\n", answer, "\n")

# ── 5.  Assertions ───────────────────────────────────────────────────────────

@testset "LLMWiki-as-tool" begin
    @test !isempty(TOOL_CALL_LOG[])
    @test any(q -> occursin(r"quorum"i, q), TOOL_CALL_LOG[])
    @test occursin(r"\bthree\b|\b3\b", lowercase(answer))
    @test occursin("12", answer) || occursin("12,000", answer) || occursin("12000", answer)
    @test !isempty(answer)
end

println("\nTool call log: ", TOOL_CALL_LOG[])
println("PASS — LLMWiki-as-tool vignette")
