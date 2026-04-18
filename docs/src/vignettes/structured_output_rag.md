# Structured output RAG: typed Answer with citations

## Narrative

Knowledge workers don't just want an answer — they want to know where
the answer came from. This vignette combines three AF features into
a single flow:

1. A **MemPalace palace** seeded with three tiny knowledge-base articles.
2. A **MemPalaceContextProvider** that retrieves the top-k drawers per
   query and injects them as context.
3. **Structured output** via `response_format_for(Answer)` +
   `parse_structured(Answer, text)` so the model must return a typed
   struct — `answer::String`, `citation_ids::Vector{String}`,
   `confidence::Float64` — which we can programmatically validate.

The key assertion: every returned `citation_id` resolves to a real
drawer's `source_file`, and at least one of the cited ids is the KB
article that contains the answer.

> **Runnable companion:**
> [`examples/structured_output_rag.jl`](https://github.com/JuliaKnowledge/AgentFramework.jl/blob/main/examples/structured_output_rag.jl)

## Key snippet

```julia
Base.@kwdef struct Answer
    answer::String
    citation_ids::Vector{String}
    confidence::Float64   # 0.0 – 1.0
end

fmt = response_format_for(Answer)

agent = Agent(
    client = OllamaChatClient(model = "qwen3:8b"),
    instructions = """
        You are a KB assistant. Answer ONLY from the supplied memory
        context. Each memory snippet is tagged inline as [KB-NNN]; copy
        those exact ids into citation_ids.
        """,
    context_providers = [mempalace_provider],
)

resp = run_agent(agent, "What is the schedule for staging deployments?";
                  options = ChatOptions(response_format = fmt))
parsed = parse_structured(Answer, clean_json(resp.text))

@assert "KB-103" in parsed.value.citation_ids
@assert parsed.value.confidence > 0.3
```

## Result

```
Parsed Answer:
  answer       = Deployments to staging happen nightly at 02:00 UTC.
  citation_ids = ["KB-103"]
  confidence   = 0.9
```

Every citation resolves to a real drawer; the answer is grounded in a
verifiable source.

## Why this pattern matters

* **Programmatic verification.** Downstream code can check citation
  validity before surfacing the answer to a user.
* **Type safety at the LLM boundary.** `parse_structured` uses the
  Julia struct to enforce types; malformed JSON fails loudly.
* **Confidence signal.** The `confidence::Float64` field lets you gate
  display — show the answer when high, fall back to "I don't know" when
  low.

## See also

* [`MemoryContextProvider`](@ref) — generic vector memory provider that
  works with any `AbstractMemoryStore`.
* [MemPalace.jl](https://github.com/JuliaKnowledge/MemPalace.jl) — the
  verbatim palace used here.
* `Structured Output` guide — `response_format_for`,
  `schema_from_type`, `parse_structured`.
