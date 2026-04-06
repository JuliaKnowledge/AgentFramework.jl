"""
    AgentFramework.Mem0Integration

Mem0 semantic memory integration for AgentFramework.jl.
Provides `Mem0Client` for communicating with Mem0 API and `Mem0ContextProvider`
for injecting memory context into agent conversations.
"""
module Mem0Integration

using ..AgentFramework
using HTTP
using JSON3

import ..AgentFramework: after_run!, before_run!, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER

include("exceptions.jl")
include("client.jl")
include("context_provider.jl")

export MEM0_PLATFORM, MEM0_OSS
export Mem0Error, Mem0Client, Mem0ContextProvider

end # module Mem0Integration
