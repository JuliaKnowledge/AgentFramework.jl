# Neo4j Integration

The `AgentFramework.Neo4jIntegration` submodule connects [Neo4j](https://neo4j.com)
knowledge graphs to AgentFramework.jl agents. It provides a low-level Cypher client,
GraphRAG-style helpers for entities and bi-temporal relationships, and a context
provider that injects a k-hop neighbourhood of a seed node into the agent's
conversation.

## Overview

The integration is built directly on Neo4j's HTTP transactional endpoint — no
external driver package is required. It works through three layers:

1. **`Neo4jClient`** — HTTP client for the transactional Cypher endpoint
   (`POST /db/{database}/tx/commit`). Supports HTTP Basic and Bearer auth.
2. **GraphRAG helpers** — `Neo4jEntity`, `Neo4jRelationship`, and operations
   (`add_entity!`, `add_relationship!`, `find_entities`, `find_related`,
   `expire_relationships!`) that model entities and bi-temporally-tracked
   relationships in the spirit of [Graphiti](https://github.com/getzep/graphiti).
3. **`Neo4jContextProvider`** — a [`BaseContextProvider`](@ref) that hooks into
   the agent lifecycle to retrieve and inject graph context before each run.

## Quick Start

```julia
using AgentFramework
using AgentFramework.Neo4jIntegration

client = Neo4jClient(
    base_url = "http://localhost:7474",
    database = "neo4j",
    user = "neo4j",
    password = "password",
)

# Seed a few entities and a relationship
alice = Neo4jEntity(uuid = "alice-1", label = "Person", name = "Alice")
bob = Neo4jEntity(uuid = "bob-1", label = "Person", name = "Bob")
add_entity!(client, alice)
add_entity!(client, bob)

add_relationship!(client, Neo4jRelationship(
    source_uuid = "alice-1",
    target_uuid = "bob-1",
    relationship_type = "KNOWS",
    properties = Dict("since" => "2024"),
))

# Inject graph context into an agent
provider = Neo4jContextProvider(
    client = client,
    seed_uuid_fn = (agent, session, ctx) -> "alice-1",
    depth = 2,
    limit = 25,
)

agent = Agent(
    name = "kg-agent",
    instructions = "You are an assistant that reasons over a knowledge graph.",
    client = OpenAIChatClient(model = "gpt-4o-mini"),
    context_providers = [provider],
)
```

## Neo4jClient

`Neo4jClient` communicates with the Neo4j HTTP transactional endpoint. Configuration
can come from constructor arguments or environment variables (`NEO4J_URL`,
`NEO4J_DATABASE`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_TOKEN`).

```julia
client = Neo4jClient(
    base_url = "http://localhost:7474",   # or NEO4J_URL
    database = "neo4j",                    # or NEO4J_DATABASE
    user = "neo4j",                        # or NEO4J_USERNAME
    password = "password",                 # or NEO4J_PASSWORD
    token = nothing,                       # Bearer token alternative to basic auth
)

# Run a read query
rows = cypher_query(client, "MATCH (n:Person) RETURN n.name AS name LIMIT 10")

# Run a write query
cypher_write(client, "CREATE (:Person {name: \$name})", Dict("name" => "Carol"))

# Health check
ping(client)
```

Errors from the Cypher transactional endpoint (which can return HTTP 200 with
`errors` in the body) are raised as `Neo4jError`.

## GraphRAG helpers

`Neo4jEntity` and `Neo4jRelationship` are thin structs that model nodes and
edges with bi-temporal metadata.

```julia
struct Neo4jEntity
    uuid::String
    label::String
    name::String
    properties::Dict{String, Any}
end

struct Neo4jRelationship
    source_uuid::String
    target_uuid::String
    relationship_type::String
    valid_at::DateTime                        # default: now(UTC)
    expired_at::Union{Nothing, DateTime}      # nothing → still valid
    properties::Dict{String, Any}
end
```

### Operations

| Function | Purpose |
|----------|---------|
| `add_entity!(client, entity)` | `MERGE` an entity by UUID, writing properties |
| `add_relationship!(client, rel)` | `MERGE` a typed relationship between two UUIDs |
| `find_entities(client; label, filter, limit)` | Scan entities by label and property filter |
| `find_related(client, seed_uuid; depth, relationship_type, limit, include_expired)` | k-hop traversal; excludes edges with non-null `expired_at` by default |
| `expire_relationships!(client, source, target; relationship_type, expired_at)` | Mark edges expired (bi-temporal edit) |
| `clear_entities!(client; label)` | Delete all entities, optionally filtered by label |

Labels and relationship types cannot be bound as Cypher parameters, so the
helpers validate them against `^[A-Za-z_][A-Za-z0-9_]*$` to prevent
Cypher injection. All user values are bound as parameters.

## Neo4jContextProvider

`Neo4jContextProvider` injects graph context into agent runs.

### Constructor

```julia
provider = Neo4jContextProvider(
    client = client,                                # Neo4jClient

    # Required: resolve the seed node UUID for each run
    seed_uuid_fn = (agent, session, ctx) -> "user-123",

    # Traversal
    depth = 2,                                       # k-hop neighbourhood
    limit = 25,                                      # max related nodes to fetch
    relationship_type = nothing,                     # optional filter
    include_expired = false,                         # include expired edges?

    # Optional: persist knowledge after a run
    writer_fn = nothing,                             # (client, agent, session, ctx, state) -> Any
)
```

### How it works

**Before each agent run (`before_run!`):**

1. Calls `seed_uuid_fn` to resolve the seed node UUID.
2. Calls `find_related` to fetch entities within `depth` hops.
3. Formats the neighbourhood as a markdown list (entity name, label, UUID,
   inbound relationships).
4. Injects the formatted text as a user message via `extend_messages!`.

**After each agent run (`after_run!`):**

If `writer_fn` is set, it's invoked with the client, agent, session, context,
and per-run `state` — useful for writing new entities or relationships derived
from the conversation back into the graph.

### Example: session-scoped seed

```julia
provider = Neo4jContextProvider(
    client = client,
    seed_uuid_fn = (agent, session, ctx) -> session.user_id,
    depth = 3,
    limit = 40,
)

session = AgentSession(user_id = "alice-1")
response = run_agent(agent, "Who does Alice know?"; session = session)
```

## API Reference

```@autodocs
Modules = [AgentFramework.Neo4jIntegration]
Order = [:module, :type, :function]
```
