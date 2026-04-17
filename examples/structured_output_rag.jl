# Structured-output RAG: typed Answer with citations from MemPalace
#
# Narrative: knowledge workers want grounded answers with explicit source
# attribution. We combine three AF features:
#
#   1. A MemPalace palace seeded with mini "kb articles" (verbatim).
#   2. A MemPalaceContextProvider that retrieves the top drawers per
#      query and injects them as context.
#   3. AF's `response_format` + `parse_structured` so the LLM returns a
#      typed `Answer` struct with a citations array we can programmatically
#      check.
#
# The final assertion proves the model grounded its answer by verifying
# each returned citation_id exists in the palace.
#
# Run:  julia --project=. examples/structured_output_rag.jl
#       (Ollama at $OLLAMA_HOST with qwen3:8b + nomic-embed-text)

using AgentFramework, MemPalace, JSON3
using Test

const OLLAMA_URL = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
const LLM_MODEL  = get(ENV, "ECO_LLM",   "qwen3:8b")
const EMB_MODEL  = get(ENV, "ECO_EMBED", "nomic-embed-text:latest")

# ── Seed a tiny knowledge base in MemPalace. `source_file` gives each
#    drawer a stable id we can cite back. ─────────────────────────────────
emb = OllamaEmbedder(model = EMB_MODEL, base_url = OLLAMA_URL, dim = 768)
palace = Palace(embedder = emb, chunk_chars = 400, chunk_overlap = 40)

const KB = [
    ("KB-101", "[KB-101] The parser module lives in src/parser.jl and handles " *
               "tokenization, AST construction, and error recovery. It was rewritten in v0.4.0."),
    ("KB-102", "[KB-102] The CI pipeline runs unit tests on Linux, macOS, and Windows. " *
               "A failed test on any platform blocks the merge."),
    ("KB-103", "[KB-103] Deployments to staging happen nightly at 02:00 UTC. Production " *
               "deploys are gated on a green CI pipeline and a manual approval."),
]
for (id, text) in KB
    mine_text!(palace, text; wing = "kb", room = "docs", source_file = id)
end

Ext = Base.get_extension(MemPalace, :MemPalaceAgentFrameworkExt)
provider = Ext.MemPalaceContextProvider(palace;
    wing      = "kb",
    room      = "docs",
    n_results = 3,
    store     = false,
)

# ── Declare the typed Answer shape. AgentFramework will emit a JSON-Schema
#    `response_format` from this struct. ──────────────────────────────────
Base.@kwdef struct Answer
    answer::String
    citation_ids::Vector{String}
    confidence::Float64   # 0.0 – 1.0
end

fmt = response_format_for(Answer)
@info "response_format" fmt

# ── Wire the agent. ──────────────────────────────────────────────────────
agent = Agent(
    name         = "kb-rag",
    client       = OllamaChatClient(model = LLM_MODEL, base_url = OLLAMA_URL),
    instructions = """
        You are a KB assistant. Answer ONLY from the supplied memory
        context. Each memory snippet is tagged with its KB id inline as
        "[KB-NNN]" at the start of the text. Copy those exact ids
        (e.g. "KB-103") into the citation_ids field — one id per snippet
        you used. If the context does not contain the answer, say so and
        set confidence < 0.3.
        """,
    context_providers = [provider],
)

# ── Ask a question answerable from exactly one KB article. ──────────────
question = "What is the schedule for staging deployments?"
resp = AgentFramework.run_agent(agent, question;
                                 options = ChatOptions(response_format = fmt))

# Strip any code-fence noise some models wrap around JSON.
function _clean_json(text::AbstractString)
    t = String(text)
    t = replace(t, r"```json\s*" => "")
    t = replace(t, r"```\s*"     => "")
    return strip(t)
end

println("\nRaw LLM text:\n", resp.text)
parsed = parse_structured(Answer, _clean_json(resp.text))
ans    = parsed.value
println("\nParsed Answer:")
println("  answer       = ", ans.answer)
println("  citation_ids = ", ans.citation_ids)
println("  confidence   = ", ans.confidence)

# ── Assertions: answer is grounded AND cites real drawers ───────────────
@testset "structured RAG answer is grounded in the KB" begin
    @test ans isa Answer
    @test !isempty(ans.answer)
    @test !isempty(ans.citation_ids)
    @test occursin("02:00", ans.answer) || occursin("nightly", lowercase(ans.answer))

    # Every citation must resolve to a real drawer (by source_file id).
    all_drawers = collect(values(palace.backend.drawers))
    valid_ids = Set(d.source_file for d in all_drawers)
    for cid in ans.citation_ids
        @test cid in valid_ids
    end
    # And at least one of the cited ids is KB-103 (the deployment doc).
    @test "KB-103" in ans.citation_ids

    # Confidence should be high since the context contains the answer.
    @test ans.confidence > 0.3
end

println("\nPASS — structured-output RAG vignette")
