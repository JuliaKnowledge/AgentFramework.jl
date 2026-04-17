# Memory backends compared: Mem0 vs Graphiti vs MemPalace
#
# Narrative: same seeded facts, same question, three different memory
# backends wired into an AgentFramework Agent. Shows how AF's pluggable
# memory layer lets you pick a backend for the task — short-term vector
# recall (Mem0), temporal knowledge graph (Graphiti), or verbatim
# wings/rooms palace (MemPalace) — without rewriting the agent.
#
# Run:  julia --project=. docs/vignettes/memory_backends_compared.jl
#       (Ollama at $OLLAMA_HOST, qwen3:8b + nomic-embed-text:latest.)

using AgentFramework, Mem0, Graphiti, MemPalace
using Dates: now, UTC
using Test

const OLLAMA_URL = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
const OLLAMA_V1  = OLLAMA_URL * "/v1"
const LLM_MODEL  = get(ENV, "ECO_LLM",   "qwen3:8b")
const EMB_MODEL  = get(ENV, "ECO_EMBED", "nomic-embed-text:latest")

# The shared corpus: three facts about Alice.
const FACTS = [
    "Alice leads the parser team at Acme Corp.",
    "Alice prefers dark-mode UIs across all tools.",
    "Alice shipped v0.4.1 to staging on 2025-02-12.",
]
const QUESTION = "What does Alice lead?"

# The AF chat client is the same in every run.
af_chat() = OllamaChatClient(model = LLM_MODEL, base_url = OLLAMA_URL)

function make_agent(provider; name="MemAgent")
    return Agent(
        name              = name,
        instructions      = "Answer briefly based on the provided memories.",
        client            = af_chat(),
        context_providers = [provider],
    )
end

# ── Mem0: vector store + optional LLM fact extraction ──────────────────
function mem0_provider()
    cfg = MemoryConfig(
        llm = LlmConfig(provider = "ollama",
                        config = Dict{String, Any}(
                            "model" => LLM_MODEL,
                            "base_url" => OLLAMA_URL,
                            "temperature" => 0.0)),
        embedder = EmbedderConfig(provider = "ollama",
                                   config = Dict{String, Any}(
                                       "model" => EMB_MODEL,
                                       "base_url" => OLLAMA_URL,
                                       "embedding_dims" => 768)),
        vector_store = VectorStoreConfig(provider = "in_memory",
                                          config = Dict{String, Any}(
                                              "collection_name" => "compare_mem0",
                                              "embedding_model_dims" => 768)),
    )
    mem = Mem0.Memory(config = cfg)
    for f in FACTS
        add(mem, f; user_id = "alice", infer = false)
    end

    Mem0Ext = Base.get_extension(AgentFramework, :AgentFrameworkMem0Ext)
    return Mem0Ext.LocalMem0ContextProvider(mem; user_id = "alice", top_k = 3)
end

# ── Graphiti: temporal knowledge graph via Ollama (OpenAI-compatible) ──
function graphiti_provider()
    driver = MemoryDriver()
    llm = OpenAILLMClient(api_key="ollama", base_url=OLLAMA_V1,
                          model=LLM_MODEL, temperature=0.0, timeout=180)
    emb = OpenAIEmbedder(api_key="ollama", base_url=OLLAMA_V1, model=EMB_MODEL)
    client = GraphitiClient(driver, llm, emb)
    for (i, f) in enumerate(FACTS)
        add_episode(client, "ep$i", f;
                    source = Graphiti.TEXT,
                    group_id = "alice",
                    reference_time = now(UTC))
    end
    GExt = Base.get_extension(Graphiti, :GraphitiAgentFrameworkExt)
    return GExt.GraphitiContextProvider(client; group_id = "alice")
end

# ── MemPalace: verbatim wings/rooms palace ──────────────────────────────
function mempalace_provider()
    emb = OllamaEmbedder(model = EMB_MODEL, base_url = OLLAMA_URL, dim = 768)
    p = Palace(embedder = emb, chunk_chars = 300, chunk_overlap = 30)
    for (i, f) in enumerate(FACTS)
        mine_text!(p, f; wing = "alice", room = "profile",
                    source_file = "fact$i.txt")
    end
    Ext = Base.get_extension(MemPalace, :MemPalaceAgentFrameworkExt)
    return Ext.MemPalaceContextProvider(p;
        wing = "alice", room = "profile",
        n_results = 3, store = false)
end

# ── Run each agent against the same question ───────────────────────────
function run_one(label, provider_factory)
    println("\n── ", label, " ──")
    provider = provider_factory()
    agent = make_agent(provider; name = label)
    sess = AgentSession(user_id = "alice")
    r = AgentFramework.run_agent(agent, QUESTION; session = sess)
    println("Q: ", QUESTION)
    println("A: ", first(r.text, 240))
    return r
end

r_mem0     = run_one("Mem0",      mem0_provider)
r_graphiti = run_one("Graphiti",  graphiti_provider)
r_palace   = run_one("MemPalace", mempalace_provider)

@testset "all three backends retrieved a grounded answer" begin
    for (label, r) in [("Mem0", r_mem0), ("Graphiti", r_graphiti),
                        ("MemPalace", r_palace)]
        @test !isempty(r.text)
        low = lowercase(r.text)
        grounded = occursin("parser", low) || occursin("acme", low)
        @test grounded
        println("  ✓ ", label, " grounded = ", grounded)
    end
end

println("\nPASS — three memory backends answered through identical agent code.")
