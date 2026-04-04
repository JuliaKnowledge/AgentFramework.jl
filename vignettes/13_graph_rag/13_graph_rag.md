# Graph RAG with RDFLib.jl
AgentFramework.jl

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Knowledge Domain: Solar System](#knowledge-domain-solar-system)
- [Step 1: Setting Up the Knowledge
  Graph](#step-1-setting-up-the-knowledge-graph)
- [Step 2: LLM-Based Entity and Relationship
  Extraction](#step-2-llm-based-entity-and-relationship-extraction)
- [Step 3: Populating the RDF Knowledge
  Graph](#step-3-populating-the-rdf-knowledge-graph)
- [Step 4: SPARQL Query Tools for the
  Agent](#step-4-sparql-query-tools-for-the-agent)
- [Step 5: Programmatic GraphRAG](#step-5-programmatic-graphrag)
- [Step 6: Asking Questions](#step-6-asking-questions)
  - [Simple entity lookup](#simple-entity-lookup)
  - [Comparative question](#comparative-question)
  - [Multi-entity query](#multi-entity-query)
- [Step 7: GraphRAG vs Plain LLM](#step-7-graphrag-vs-plain-llm)
- [Step 8: Inspecting the Knowledge
  Graph](#step-8-inspecting-the-knowledge-graph)
- [Summary](#summary)
  - [Key Takeaways](#key-takeaways)

## Overview

Graph-based Retrieval-Augmented Generation (GraphRAG) enhances LLM
responses by grounding them in structured knowledge graphs rather than
flat text chunks. By extracting entities and relationships into an RDF
knowledge graph and querying it with SPARQL, an agent can reason over
interconnected facts, follow multi-hop relationships, and provide more
accurate, explainable answers.

In this vignette, you will learn how to:

- Build an RDF knowledge graph from unstructured text using an LLM
- Store entities and relationships as RDF triples with RDFLib.jl
- Create agent tools that query the knowledge graph via SPARQL
- Implement a full GraphRAG pipeline: ingest → extract → store → query →
  answer
- Compare GraphRAG answers against plain LLM responses

## Prerequisites

- **Ollama** running locally with a model:

  ``` bash
  ollama pull gemma3
  ```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using AgentFramework
using RDFLib
using JSON3
```

## Knowledge Domain: Solar System

We use a short corpus about the solar system. In production, this text
could come from documents, web pages, or databases.

``` julia
corpus = [
    """The Sun is the star at the center of the Solar System. It is a G-type
    main-sequence star (G2V). The Sun's diameter is about 1,392,700 km, roughly
    109 times that of Earth. It contains 99.86% of the total mass of the Solar System.""",

    """Mercury is the smallest planet in the Solar System and the closest to
    the Sun. It has no atmosphere and its surface temperature ranges from
    -180°C to 430°C. Mercury's orbital period is about 88 Earth days.""",

    """Venus is the second planet from the Sun. It is similar in size to Earth
    and is sometimes called Earth's sister planet. Venus has a thick atmosphere
    of carbon dioxide with sulfuric acid clouds. Its surface temperature is
    about 465°C, making it the hottest planet.""",

    """Earth is the third planet from the Sun and the only known planet to
    harbor life. It has one natural satellite, the Moon. Earth's diameter is
    12,742 km and its orbital period is 365.25 days.""",

    """Mars is the fourth planet from the Sun, often called the Red Planet due
    to iron oxide on its surface. Mars has two small moons: Phobos and Deimos.
    It has a thin atmosphere of mostly carbon dioxide. Mars's diameter is about
    6,779 km.""",

    """Jupiter is the fifth planet from the Sun and the largest in the Solar
    System. It is a gas giant with a mass more than twice that of all other
    planets combined. Jupiter has at least 95 known moons, including the four
    large Galilean moons: Io, Europa, Ganymede, and Callisto.""",

    """Saturn is the sixth planet from the Sun, famous for its ring system. It
    is the second-largest planet in the Solar System. Saturn has at least 146
    known moons, with Titan being the largest. Titan has a thick nitrogen
    atmosphere.""",
]
nothing
```

## Step 1: Setting Up the Knowledge Graph

First, we create an RDF graph with a namespace for our domain:

``` julia
kg = RDFGraph()
astro = Namespace("http://example.org/astro/")

function to_uri(name::String)
    slug = replace(lowercase(strip(name)), r"[^a-z0-9]+" => "_")
    return slug
end

println("Knowledge graph initialized: ", length(kg), " triples")
```

    Knowledge graph initialized: 0 triples

## Step 2: LLM-Based Entity and Relationship Extraction

The key innovation in GraphRAG is using an LLM to extract structured
knowledge from unstructured text. We prompt the model to return JSON
triples that we then load into the RDF graph.

``` julia
extraction_client = OllamaChatClient(model="gemma3:latest")

extraction_prompt = """You are a knowledge graph extraction system.
Given a text passage, extract entities and relationships as JSON triples.

Return ONLY a JSON array of objects, each with "subject", "predicate", and "object" fields.
Use simple, clear names (e.g., "Earth", "type_of", "Planet").
Extract factual relationships like: type_of, part_of, has_moon, has_diameter,
has_temperature, has_atmosphere, orbits, orbital_period, etc.
For numeric values, include the units (e.g., "12742 km", "365.25 days").

Do NOT include any text outside the JSON array. Do NOT use markdown code blocks.

Example output:
[{"subject": "Earth", "predicate": "type_of", "object": "Planet"},
 {"subject": "Earth", "predicate": "has_moon", "object": "Moon"},
 {"subject": "Earth", "predicate": "orbits", "object": "Sun"}]
"""
nothing
```

``` julia
extraction_agent = Agent(
    name="Extractor",
    instructions=extraction_prompt,
    client=extraction_client,
    options=ChatOptions(temperature=0.1),
)

all_extracted = Dict{String,String}[]

for (i, passage) in enumerate(corpus)
    println("Extracting from passage $i/$(length(corpus))...")

    response = run_agent(extraction_agent, "Extract knowledge triples from:\n\n$passage")
    raw_text = get_text(response)

    # Parse JSON from the response (strip any markdown fencing)
    cleaned = replace(raw_text, r"```json\s*" => "", r"```\s*" => "")
    cleaned = strip(cleaned)

    # Find the JSON array — parse flexibly since LLMs sometimes return
    # arrays as object values (e.g., "object": ["Phobos", "Deimos"])
    start_idx = findfirst('[', cleaned)
    end_idx = findlast(']', cleaned)
    if !isnothing(start_idx) && !isnothing(end_idx)
        json_str = cleaned[start_idx:end_idx]
        try
            raw = JSON3.read(json_str)
            for item in raw
                s = string(get(item, :subject, ""))
                p = string(get(item, :predicate, ""))
                obj = get(item, :object, "")
                # If the object is an array, create one triple per element
                objs = obj isa AbstractVector ? [string(o) for o in obj] : [string(obj)]
                for o in objs
                    if !isempty(s) && !isempty(p) && !isempty(o)
                        push!(all_extracted, Dict("subject" => s, "predicate" => p, "object" => o))
                    end
                end
            end
            println("  → Extracted triples (total so far: $(length(all_extracted)))")
        catch e
            println("  ⚠ Parse error: ", e)
        end
    else
        println("  ⚠ No JSON array found")
    end
end

println("\nTotal extracted triples: ", length(all_extracted))
```

    Extracting from passage 1/7...
      → Extracted triples (total so far: 8)
    Extracting from passage 2/7...
      → Extracted triples (total so far: 14)
    Extracting from passage 3/7...
      → Extracted triples (total so far: 22)
    Extracting from passage 4/7...
      → Extracted triples (total so far: 27)
    Extracting from passage 5/7...
      → Extracted triples (total so far: 35)
    Extracting from passage 6/7...
      → Extracted triples (total so far: 45)
    Extracting from passage 7/7...
      → Extracted triples (total so far: 53)

    Total extracted triples: 53

## Step 3: Populating the RDF Knowledge Graph

Now we convert the extracted triples into proper RDF and add them to the
graph:

``` julia
for triple_dict in all_extracted
    s = astro(to_uri(triple_dict["subject"]))
    p = astro(to_uri(triple_dict["predicate"]))

    obj_str = triple_dict["object"]
    # Use Literal for values that look like measurements
    if any(c -> isdigit(c), obj_str) && any(u -> occursin(u, obj_str), ["km", "°C", "days", "%", "kg"])
        o = Literal(obj_str)
    else
        o = astro(to_uri(obj_str))
    end

    add!(kg, Triple(s, p, o))
end

# Add RDF type triples
for triple_dict in all_extracted
    if triple_dict["predicate"] in ["type_of", "is_a"]
        s = astro(to_uri(triple_dict["subject"]))
        o = astro(to_uri(triple_dict["object"]))
        add!(kg, Triple(s, RDF.type, o))
    end
end

println("Knowledge graph now contains $(length(kg)) triples")
```

    Knowledge graph now contains 58 triples

Let’s inspect the graph by serializing to Turtle format:

``` julia
println(serialize(kg, TurtleFormat()))
```

    @prefix ns1: <http://example.org/astro/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ns1:atmosphere ns1:of ns1:carbon_dioxide .

    ns1:earth a ns1:planet ;
        ns1:distance_from ns1:sun ;
        ns1:has_diameter "1,392,700 km",
            "12742 km" ;
        ns1:has_moon ns1:moon ;
        ns1:has_orbital_period "365.25 days" ;
        ns1:type_of ns1:planet .

    ns1:jupiter a ns1:gas_giant ;
        ns1:has_mass ns1:twice_that_of_all_other_planets ;
        ns1:has_moons ns1:95 ;
        ns1:includes ns1:io,
            ns1:europa,
            ns1:ganymede,
            ns1:callisto ;
        ns1:is_from ns1:solar_system ;
        ns1:orbits ns1:sun ;
        ns1:rank ns1:fifth ;
        ns1:type_of ns1:gas_giant .

    ns1:mars a ns1:planet ;
        ns1:has_atmosphere ns1:carbon_dioxide ;
        ns1:has_diameter "6,779 km" ;
        ns1:has_moon ns1:phobos,
            ns1:deimos ;
        ns1:has_surface_feature ns1:iron_oxide ;
        ns1:is_called ns1:red_planet ;
        ns1:orbits ns1:sun ;
        ns1:type_of ns1:planet .

    ns1:mercury a ns1:planet ;
        ns1:closest_to ns1:sun ;
        ns1:has_atmosphere ns1:no_atmosphere ;
        ns1:has_orbital_period "88 Earth days" ;
        ns1:has_temperature_max "430°C" ;
        ns1:has_temperature_min "-180°C" ;
        ns1:type_of ns1:planet .

    ns1:saturn ns1:famous_for ns1:ring_system ;
        ns1:from ns1:sun ;
        ns1:has_moon ns1:146,
            ns1:titan ;
        ns1:is ns1:planet,
            ns1:sixth_planet,
            ns1:second_largest_planet .

    ns1:solar_system ns1:has_mass "99.86%" .

    ns1:sun a ns1:star ;
        ns1:at_center_of ns1:solar_system ;
        ns1:diameter_relative_to ns1:earth ;
        ns1:has_diameter "1,392,700 km" ;
        ns1:has_mass_proportion "99.86%" ;
        ns1:has_spectral_type ns1:g2v ;
        ns1:type_of ns1:star .

    ns1:titan ns1:has_atmosphere ns1:nitrogen_atmosphere .

    ns1:venus ns1:has ns1:thick_atmosphere,
            ns1:sulfuric_acid_clouds ;
        ns1:has_temperature "465°C" ;
        ns1:is ns1:second_planet_from_the_sun ;
        ns1:is_called ns1:earth_s_sister_planet ;
        ns1:is_hottest ns1:planet ;
        ns1:is_similar_in_size_to ns1:earth .

## Step 4: SPARQL Query Tools for the Agent

We create query functions to retrieve facts from the knowledge graph —
the “retrieval” component of GraphRAG.

``` julia
"""Find all known facts about a specific entity."""
function query_entity(entity_name::String)
    uri = astro(to_uri(entity_name))
    results = String[]

    for t in triples(kg, (uri, nothing, nothing))
        pred = split(string(t.predicate), "/")[end]
        obj = t.object isa Literal ? string(t.object) : split(string(t.object), "/")[end]
        push!(results, "$pred: $obj")
    end

    for t in triples(kg, (nothing, nothing, uri))
        subj = split(string(t.subject), "/")[end]
        pred = split(string(t.predicate), "/")[end]
        push!(results, "$subj $pred this entity")
    end

    isempty(results) && return "No facts found for '$entity_name'. Try a different name."
    return "Facts about $entity_name:\n" * join(results, "\n")
end

"""Execute a SPARQL query against the knowledge graph."""
function run_sparql(query::String)
    try
        results = sparql_query(kg, query)
        isempty(results) && return "Query returned no results."
        lines = String[]
        for row in results
            parts = ["$k=$(v isa Literal ? string(v) : split(string(v), "/")[end])" for (k, v) in row]
            push!(lines, join(parts, ", "))
        end
        return "Results ($(length(results)) rows):\n" * join(lines, "\n")
    catch e
        return "SPARQL error: $(sprint(showerror, e))"
    end
end

"""List all entities of a given type in the knowledge graph."""
function list_entities_of_type(type_name::String)
    type_uri = astro(to_uri(type_name))
    entities = String[]

    for t in triples(kg, (nothing, RDF.type, type_uri))
        push!(entities, split(string(t.subject), "/")[end])
    end

    type_pred = astro("type_of")
    for t in triples(kg, (nothing, type_pred, type_uri))
        name = split(string(t.subject), "/")[end]
        name ∉ entities && push!(entities, name)
    end

    isempty(entities) && return "No entities found of type '$type_name'."
    return "Entities of type $type_name: " * join(entities, ", ")
end
nothing
```

Let’s verify the retrieval functions work:

``` julia
println(query_entity("Earth"))
println()
println(query_entity("Jupiter"))
```

    Facts about Earth:
    has_diameter: 1,392,700 km
    type_of: planet
    distance_from: sun
    has_moon: moon
    has_diameter: 12742 km
    has_orbital_period: 365.25 days
    22-rdf-syntax-ns#type: planet
    venus is_similar_in_size_to this entity
    sun diameter_relative_to this entity

    Facts about Jupiter:
    type_of: gas_giant
    orbits: sun
    is_from: solar_system
    rank: fifth
    has_mass: twice_that_of_all_other_planets
    has_moons: 95
    includes: io
    includes: europa
    includes: ganymede
    includes: callisto
    22-rdf-syntax-ns#type: gas_giant

``` julia
println(list_entities_of_type("Planet"))
```

    Entities of type Planet: mercury, earth, mars

## Step 5: Programmatic GraphRAG

The core of GraphRAG is **retrieve then generate**: first query the
knowledge graph for relevant facts, then pass those facts as context to
the LLM. We define a helper that implements this pipeline:

``` julia
"""Ask a question using GraphRAG: retrieve from KG, then generate with LLM."""
function ask_graph_rag(question::String; entities::Vector{String}=String[])
    # Step 1: Retrieve — gather facts from the knowledge graph
    context_parts = String[]
    for entity in entities
        facts = query_entity(entity)
        push!(context_parts, facts)
    end
    context = join(context_parts, "\n\n")

    # Step 2: Generate — ask the LLM to answer using only the retrieved context
    rag_agent = Agent(
        name="GraphRAG",
        instructions="""You are an astronomy expert. Answer the question using ONLY the
knowledge graph facts provided below. Do not use your own knowledge.
If the facts don't contain enough information, say so.

Knowledge Graph Facts:
$context""",
        client=extraction_client,
        options=ChatOptions(temperature=0.1),
    )
    response = run_agent(rag_agent, question)
    return get_text(response)
end
nothing
```

## Step 6: Asking Questions

Let’s ask questions and see how the GraphRAG pipeline retrieves facts
and generates grounded answers:

### Simple entity lookup

``` julia
answer = ask_graph_rag("What do we know about Mars?", entities=["Mars"])
println(answer)
```

    Based on the knowledge graph facts, we know the following about Mars:

    *   It is a planet.
    *   It orbits the sun.
    *   It is called the red planet.
    *   It has a surface feature of iron oxide.
    *   It has two moons: Phobos and Deimos.
    *   It has an atmosphere composed of carbon dioxide.
    *   Its diameter is 6,779 km.

### Comparative question

``` julia
answer = ask_graph_rag(
    "Which planet is larger, Earth or Mars? Give their diameters.",
    entities=["Earth", "Mars"],
)
println(answer)
```

    Earth has a diameter of 1,392,700 km and Mars has a diameter of 6,779 km. Therefore, Earth is larger than Mars.

### Multi-entity query

``` julia
answer = ask_graph_rag(
    "What moons does Jupiter have?",
    entities=["Jupiter"],
)
println(answer)
```

    Jupiter has the following moons: io, europa, ganymede, and callisto.

## Step 7: GraphRAG vs Plain LLM

Let’s compare the GraphRAG pipeline (grounded in the knowledge graph)
with a plain LLM that has no graph context:

``` julia
plain_agent = Agent(
    name="PlainAssistant",
    instructions="You are a helpful assistant. Answer questions concisely.",
    client=extraction_client,
    options=ChatOptions(temperature=0.1),
)

question = "What moons does Jupiter have?"

println("=" ^ 60)
println("Question: $question")
println("=" ^ 60)

println("\n--- GraphRAG (grounded in knowledge graph) ---")
answer_rag = ask_graph_rag(question, entities=["Jupiter"])
println(answer_rag)

println("\n--- Plain LLM (no knowledge graph) ---")
response_plain = run_agent(plain_agent, question)
println(get_text(response_plain))
```

    ============================================================
    Question: What moons does Jupiter have?
    ============================================================

    --- GraphRAG (grounded in knowledge graph) ---
    Jupiter has the following moons: io, europa, ganymede, and callisto.

    --- Plain LLM (no knowledge graph) ---
    Jupiter has 95 confirmed moons. Here’s a breakdown:

    *   **Galilean Moons:** Io, Europa, Ganymede, Callisto
    *   **Small Moons:** (Over 70 others)

The GraphRAG answer is constrained to facts extracted into the knowledge
graph, while the plain LLM draws on its training data — which may be
more comprehensive but is also less traceable and potentially less
accurate for domain-specific queries.

## Step 8: Inspecting the Knowledge Graph

One advantage of GraphRAG is transparency — we can inspect exactly what
knowledge the agent has access to:

``` julia
subjects_set = Set(string(t.subject) for t in triples(kg))
predicates_set = Set(string(t.predicate) for t in triples(kg))

println("Knowledge Graph Statistics:")
println("  Total triples: ", length(kg))
println("  Unique entities (subjects): ", length(subjects_set))
println("  Unique predicates: ", length(predicates_set))
println()
println("Predicates used:")
for p in sort(collect(predicates_set))
    name = split(p, "/")[end]
    count = length(collect(triples(kg, (nothing, URIRef(p), nothing))))
    println("  $name: $count triples")
end
```

    Knowledge Graph Statistics:
      Total triples: 58
      Unique entities (subjects): 10
      Unique predicates: 30

    Predicates used:
      at_center_of: 1 triples
      closest_to: 1 triples
      diameter_relative_to: 1 triples
      distance_from: 1 triples
      famous_for: 1 triples
      from: 1 triples
      has: 2 triples
      has_atmosphere: 3 triples
      has_diameter: 4 triples
      has_mass: 2 triples
      has_mass_proportion: 1 triples
      has_moon: 5 triples
      has_moons: 1 triples
      has_orbital_period: 2 triples
      has_spectral_type: 1 triples
      has_surface_feature: 1 triples
      has_temperature: 1 triples
      has_temperature_max: 1 triples
      has_temperature_min: 1 triples
      includes: 4 triples
      is: 4 triples
      is_called: 2 triples
      is_from: 1 triples
      is_hottest: 1 triples
      is_similar_in_size_to: 1 triples
      of: 1 triples
      orbits: 2 triples
      rank: 1 triples
      type_of: 5 triples
      22-rdf-syntax-ns#type: 5 triples

## Summary

| Component            | Purpose                                           |
|----------------------|---------------------------------------------------|
| `RDFGraph`           | In-memory RDF knowledge graph store               |
| `Namespace`          | URI minting for entities and predicates           |
| `Triple` / `Literal` | RDF triple representation                         |
| `sparql_query`       | SPARQL query execution                            |
| `Agent` (extraction) | LLM converts text → structured triples            |
| `Agent` (generation) | LLM answers questions grounded in retrieved facts |
| Retrieval functions  | Query the graph for relevant context              |

### Key Takeaways

1.  **GraphRAG = Knowledge Graph + LLM**: Extract structured knowledge
    from text, store as RDF, query with SPARQL.
2.  **Retrieve then generate**: Programmatically query the graph for
    relevant facts, then pass them as context to the LLM.
3.  **Transparency**: Every answer can be traced back to specific
    triples in the knowledge graph.
4.  **RDFLib.jl**: Provides a full RDF stack (graphs, SPARQL,
    serialization) natively in Julia.
5.  **Extensibility**: The knowledge graph can be enriched with RDFS/OWL
    reasoning, external SPARQL endpoints, or additional data sources.
