#!/usr/bin/env julia
# =============================================================================
# Azure / Ollama parity: one agent, three interchangeable providers
# =============================================================================
#
# AgentFramework's `AbstractChatClient` contract is what makes the whole
# stack provider-agnostic.  Any struct that implements it plugs into
# `Agent` the same way.  In practice we use three flavours:
#
#   * `OllamaChatClient`            — Ollama's native /api/chat endpoint.
#   * `OpenAIChatClient`            — OpenAI-compatible /v1/chat/completions.
#                                      (Works against Ollama's own /v1 shim,
#                                      and against OpenAI, vLLM, LM Studio, …)
#   * `AzureOpenAIChatClient`       — Azure OpenAI resource, with API-key
#                                      OR AAD bearer token (via AzureIdentity).
#
# This vignette builds the agent **once**, runs it against the first two
# live against Ollama (same question, same expected answer), and then
# constructs an `AzureOpenAIChatClient` with the same config shape — NOT
# calling it, since the demo box doesn't have Azure credentials, but
# proving the code path is identical.
#
# Requires OLLAMA_HOST reachable + qwen3:8b.
# -----------------------------------------------------------------------------

using Test
using AgentFramework

ollama_host = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
question = "In one short sentence: what is 2+2?"

function run_with(client::AbstractChatClient)
    agent = Agent(
        client = client,
        instructions = "Answer in one short sentence, numerically accurate.",
    )
    resp = run_agent(agent, question)
    # Strip qwen3 <think> blocks.
    return strip(replace(resp.text, r"<think>.*?</think>"s => ""))
end

# ── 1. Ollama native ─────────────────────────────────────────────────────────

println("\n── Provider 1: OllamaChatClient (native /api/chat) ──")
client_native = OllamaChatClient(model = "qwen3:8b", base_url = ollama_host)
ans_native = run_with(client_native)
println("Answer: ", ans_native)

# ── 2. OpenAI-compatible shim (Ollama /v1) ───────────────────────────────────
#
# Ollama ships an OpenAI-compatible endpoint at /v1.  Pointing
# OpenAIChatClient at it demonstrates that AF's OpenAI driver works
# against any OpenAI-protocol server — including OpenAI itself, vLLM,
# LM Studio, Together, etc.

println("\n── Provider 2: OpenAIChatClient → Ollama /v1 shim ──")
client_openai = OpenAIChatClient(
    model    = "qwen3:8b",
    base_url = ollama_host * "/v1",
    api_key  = "ollama",         # dummy; Ollama ignores the key
)
ans_openai = run_with(client_openai)
println("Answer: ", ans_openai)

# ── 3. Azure OpenAI (constructed only — no live call) ────────────────────────
#
# Azure OpenAI deployments support two auth modes:
#   (a) API key              — `api_key = "…"` on the client
#   (b) Entra ID bearer      — pass an `AzureIdentity` credential via
#                              `credential = DefaultAzureCredential()`
#
# Both come from the same `AzureOpenAIChatClient` type.  Constructing it
# is the ONLY line you change when moving an agent from local to Azure.

println("\n── Provider 3: AzureOpenAIChatClient (constructed only) ──")
client_azure = AzureOpenAIChatClient(
    model = "gpt-4o-mini",                            # Azure deployment name
    endpoint = "https://contoso-openai.openai.azure.com",
    api_key = "demo-key-not-real",
    api_version = "2024-06-01",
)
println("Constructed: ", client_azure)
println("(skipping live call — Azure credentials not in scope for this vignette)")

# ── Assertions ───────────────────────────────────────────────────────────────

@testset "Azure / Ollama provider parity" begin
    # Both live providers must give a grounded answer containing "4".
    @test occursin("4", ans_native)
    @test occursin("4", ans_openai)

    # All three clients subtype the common abstraction.
    @test client_native isa AbstractChatClient
    @test client_openai isa AbstractChatClient
    @test client_azure  isa AbstractChatClient

    # Fluent surface matches (same constructor keywords work on all three).
    @test client_azure.model == "gpt-4o-mini"
    @test client_azure.endpoint == "https://contoso-openai.openai.azure.com"
    @test !isempty(client_azure.api_version)
end

println("\nPASS — Azure/Ollama parity vignette")
