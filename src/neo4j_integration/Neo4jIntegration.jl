"""
    AgentFramework.Neo4jIntegration

Neo4j knowledge-graph integration for AgentFramework.jl.

Provides `Neo4jClient` for HTTP-transport Cypher access, GraphRAG-style
entity/relationship helpers with bi-temporal edges, and a
`Neo4jContextProvider` that injects a k-hop neighbourhood of a seed node
into the agent's message context.
"""
module Neo4jIntegration

using ..AgentFramework
using Base64: base64encode
using Dates
using HTTP
using JSON3

import ..AgentFramework:
    after_run!,
    before_run!,
    extend_messages!,
    AgentSession,
    BaseContextProvider,
    Message,
    SessionContext,
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM

include("exceptions.jl")
include("client.jl")
include("cypher.jl")
include("graph_ops.jl")
include("context_provider.jl")

export Neo4jError
export Neo4jClient, cypher_query, cypher_write, ping
export Neo4jEntity, Neo4jRelationship
export add_entity!, add_relationship!, find_entities, find_related
export expire_relationships!, clear_entities!
export Neo4jContextProvider

end # module Neo4jIntegration
