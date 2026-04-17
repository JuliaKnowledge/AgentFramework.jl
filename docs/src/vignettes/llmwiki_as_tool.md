# LLMWiki as a tool: giving an agent a knowledge base

## Narrative

[LLMWiki.jl](https://github.com/JuliaKnowledge/LLMWiki.jl) maintains a
version-controlled, LLM-compiled knowledge base out of your source
notes. It exposes a BM25 index over the compiled concept pages via
`search_wiki(config, query; method=:bm25, top_k=K)`.

Exposing that function as an AgentFramework `FunctionTool` is a few
lines — and gives the agent a surgical way to ground answers in your
verified, slowly-changing knowledge, separately from any dynamic memory
provider.

> **Runnable companion:**
> [`examples/llmwiki_as_tool.jl`](https://github.com/JuliaKnowledge/AgentFramework.jl/blob/main/examples/llmwiki_as_tool.jl)

## Key snippet

```julia
using LLMWiki, AgentFramework

config = default_config(mktempdir())
init_wiki(config)
# ... write concept pages under config.concepts_dir ...

function wiki_search_tool(query::AbstractString; top_k::Int = 3)
    hits = search_wiki(config, String(query); method = :bm25, top_k = top_k)
    buf = IOBuffer()
    for (i, h) in enumerate(hits)
        println(buf, "[$i] $(h.title) (slug=$(h.slug))")
        body = read(joinpath(config.concepts_dir, h.slug * ".md"), String)
        body = replace(body, r"^---\n.*?\n---\n"s => "")
        println(buf, "    ", first(strip(body), 600))
    end
    String(take!(buf))
end

tool = FunctionTool(
    name = "wiki_search",
    description = "Search the project wiki for facts about internal systems.",
    func = wiki_search_tool,
    parameters = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "query" => Dict{String,Any}("type" => "string"),
            "top_k" => Dict{String,Any}("type" => "integer"),
        ),
        "required" => ["query"],
    ),
)

agent = Agent(
    client = OllamaChatClient(model = "qwen3:8b"),
    instructions = "When asked about internal project concepts, ALWAYS call wiki_search first.",
    tools = [tool],
)

resp = run_agent(agent, "What is the default quorum-size of the Quorum Protocol?")
```

## Result

```
Tool call log: ["Quorum Protocol default quorum-size",
                "Quorum Protocol benchmark ops per second"]
```

```
Assistant:
The Quorum Protocol has a default quorum-size of three, meaning an
operation commits once three of the five required voting nodes
acknowledge it. It has been benchmarked at roughly 12,000 ops/s.
```

The agent made two tool calls without being told to; both facts came
straight from the wiki pages.

## Why this pattern matters

* **Slow knowledge vs fast memory.** Wiki pages change via PR review.
  Memory providers (Mem0, MemPalace, Graphiti) change per-turn. Giving
  the agent both lets you separate **stable canonical facts** from
  **personal/dynamic context**.
* **Citations for free.** `SearchResult.slug` is a natural citation
  anchor that links back to the source file in your repo.
* **Human-writable.** The concept pages are plain Markdown with YAML
  frontmatter — no rebuild step needed when an engineer edits one.

## See also

* [`LLMWiki.jl`](https://github.com/JuliaKnowledge/LLMWiki.jl) —
  `compile!`, `query_wiki`, `search_wiki`, `build_bm25_index`.
* [Memory backends compared](memory_backends_compared.md) — for
  per-user dynamic context.
* [Structured output RAG](structured_output_rag.md) — combine the two
  patterns: wiki-as-tool + typed Answer struct with citations.
