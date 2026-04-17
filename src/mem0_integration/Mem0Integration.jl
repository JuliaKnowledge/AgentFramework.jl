"""
    AgentFramework.Mem0Integration

Integration with the **Mem0 cloud / SaaS** REST API
(`https://api.mem0.ai`, and `MEM0_OSS` for self-hosted OSS deployments).
Provides `Mem0Client` as the HTTP client and `Mem0ContextProvider` as a
`BaseContextProvider` that retrieves and persists memories against the
remote Mem0 service.

!!! note "Which Mem0 do I want?"
    - **`Mem0Integration`** (this module) — talks to the Mem0 SaaS or
      a self-hosted Mem0 OSS deployment over HTTP. Use this if you
      already have a Mem0 account or a running Mem0 server.
    - **`LocalMem0ContextProvider`** (defined in the
      `AgentFrameworkMem0Ext` package extension, loaded when both
      AgentFramework and the `Mem0.jl` package are imported) — talks
      directly to a `Mem0.Memory` struct from the Julia `Mem0.jl`
      package, which runs entirely in-process with a SQLite vector
      store. Use this for fully-offline, self-hosted agent memory.

    The two paths are complementary and can be used together in the
    same agent.
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

"""
    AgentFramework.Mem0Cloud

Alias for [`Mem0Integration`](@ref) — the module name that most
accurately describes what it does (talk to the Mem0 cloud / SaaS
REST API). Prefer `Mem0Cloud` in new code for clarity when the local
`AgentFrameworkMem0Ext` extension is also in play.
"""
const Mem0Cloud = Mem0Integration
