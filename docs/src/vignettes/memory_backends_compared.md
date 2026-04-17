# Memory backends compared: Mem0 vs Graphiti vs MemPalace

## Narrative

AgentFramework.jl has a **modular memory stack**. Any
`BaseContextProvider` can be slotted into an Agent's
`context_providers` list without touching the rest of the code. That
means the same agent code can be driven by very different memory
backends: a vector store (Mem0), a temporal knowledge graph (Graphiti),
or a verbatim wings/rooms palace (MemPalace). This vignette seeds the
same three facts into each backend and asks the same question, so you
can see them side-by-side.

> **Runnable companion:**
> [`examples/memory_backends_compared.jl`](https://github.com/JuliaKnowledge/AgentFramework.jl/blob/main/examples/memory_backends_compared.jl)
> (requires Ollama at `$OLLAMA_HOST` with `qwen3:8b` +
> `nomic-embed-text:latest`)

## When to pick which

| Backend    | Retrieval model             | Best for                                                      |
|------------|-----------------------------|---------------------------------------------------------------|
| Mem0       | Vector + optional LLM facts | Short-term / session memory; conversational preference capture |
| Graphiti   | Temporal knowledge graph    | Long-term relationships, "who knew what when" queries         |
| MemPalace  | Verbatim wings/rooms + BM25 | Scope-isolated note corpora where exact text matters (audit, compliance) |

## Shared setup

```julia
using AgentFramework, Mem0, Graphiti, MemPalace

const FACTS = [
    "Alice leads the parser team at Acme Corp.",
    "Alice prefers dark-mode UIs across all tools.",
    "Alice shipped v0.4.1 to staging on 2025-02-12.",
]
const QUESTION = "What does Alice lead?"

af_chat() = OllamaChatClient(model = "qwen3:8b",
                             base_url = "http://localhost:11434")

function make_agent(provider; name="MemAgent")
    Agent(name = name,
          instructions = "Answer briefly based on the provided memories.",
          client = af_chat(),
          context_providers = [provider])
end
```

## Backend 1 — Mem0 (vector store)

```julia
cfg = MemoryConfig(
    llm      = LlmConfig(provider = "ollama", config = Dict("model" => "qwen3:8b", ...)),
    embedder = EmbedderConfig(provider = "ollama", config = Dict("model" => "nomic-embed-text", ...)),
    vector_store = VectorStoreConfig(provider = "in_memory", ...),
)
mem = Mem0.Memory(config = cfg)
for f in FACTS; add(mem, f; user_id = "alice", infer = false); end

Mem0Ext = Base.get_extension(AgentFramework, :AgentFrameworkMem0Ext)
provider = Mem0Ext.LocalMem0ContextProvider(mem; user_id = "alice", top_k = 3)
```

## Backend 2 — Graphiti (temporal graph)

```julia
driver = MemoryDriver()
llm    = OpenAILLMClient(api_key="ollama", base_url="http://localhost:11434/v1",
                          model="qwen3:8b")
emb    = OpenAIEmbedder(api_key="ollama", base_url="http://localhost:11434/v1",
                         model="nomic-embed-text")
client = GraphitiClient(driver, llm, emb)
for (i, f) in enumerate(FACTS)
    add_episode(client, "ep\$i", f; source = Graphiti.TEXT,
                group_id = "alice", reference_time = now(UTC))
end

GExt = Base.get_extension(Graphiti, :GraphitiAgentFrameworkExt)
provider = GExt.GraphitiContextProvider(client; group_id = "alice")
```

## Backend 3 — MemPalace (verbatim palace)

```julia
emb = OllamaEmbedder(model = "nomic-embed-text",
                     base_url = "http://localhost:11434", dim = 768)
p = Palace(embedder = emb, chunk_chars = 300, chunk_overlap = 30)
for (i, f) in enumerate(FACTS)
    mine_text!(p, f; wing = "alice", room = "profile",
                source_file = "fact\$i.txt")
end

Ext = Base.get_extension(MemPalace, :MemPalaceAgentFrameworkExt)
provider = Ext.MemPalaceContextProvider(p;
    wing = "alice", room = "profile", n_results = 3, store = false)
```

## Result

All three agents answer the question grounded in the seeded facts:

```
── Mem0 ──
A: Alice leads the parser team at Acme Corp.

── Graphiti ──
A: Alice leads the parser team.

── MemPalace ──
A: Alice leads the parser team at Acme Corp.
```

The agent code, the LLM, and the embedder model are all identical — only
the memory backend changes. Pick the one whose retrieval model matches
your data.

## See also

* [`Mem0Integration`](@ref) — the built-in AF integration using Mem0's
  cloud API or self-hosted server.
* [`MemoryContextProvider`](@ref) — the generic vector-memory provider
  that works with **any** `AbstractMemoryStore` (including
  `MemPalaceMemoryStore`).
* [Graphiti.jl](https://github.com/JuliaKnowledge/Graphiti.jl) — temporal
  knowledge graph.
* [MemPalace.jl](https://github.com/JuliaKnowledge/MemPalace.jl) —
  wings/rooms/drawers/closets.
