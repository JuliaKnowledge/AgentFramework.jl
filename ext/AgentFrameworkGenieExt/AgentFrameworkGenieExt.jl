"""
    AgentFrameworkGenieExt

Development web interface for AgentFramework.jl, built with Genie.jl.
Loaded automatically when both AgentFramework and Genie are loaded.

Provides a chat UI for interacting with agents and workflows during development.

## Quick Start

```julia
using AgentFramework, Genie

# DevUI functions are now available
serve_devui(entities=[agent])
```
"""
module AgentFrameworkGenieExt

using AgentFramework
using Genie, Genie.Router, Genie.Requests
using JSON3
using UUIDs
using Dates
using HTTP
using Logging

# ─── Configuration ────────────────────────────────────────────────────────────

"""
    DevUIConfig

Configuration for the development UI server.
"""
Base.@kwdef mutable struct DevUIConfig
    title::String = "AgentFramework DevUI"
    port::Int = 8080
    host::String = "127.0.0.1"
    dev_mode::Bool = true
end

# ─── Sub-modules ──────────────────────────────────────────────────────────────

include("entities.jl")
include("conversations.jl")
include("ag_ui_protocol.jl")
include("streaming.jl")
include("api.jl")
include("server.jl")

# ─── Entry Point ──────────────────────────────────────────────────────────────

"""
    AgentFramework.serve_devui(; entities, port, host, auto_open, title, dev_mode)

Start the AgentFramework DevUI server.

# Keyword Arguments
- `entities`: Vector of `Agent` or `Workflow` objects to register.
- `port::Int = 8080`: Port to listen on.
- `host::String = "127.0.0.1"`: Host to bind to.
- `auto_open::Bool = true`: Whether to open the browser automatically.
- `title::String = "AgentFramework DevUI"`: Title displayed in the UI.
- `dev_mode::Bool = true`: Enable development mode (verbose logging).

# Example
```julia
using AgentFramework, Genie

agent = Agent(name="Helper", client=OllamaChatClient("llama3"))
serve_devui(entities=[agent], port=8080)
```
"""
function AgentFramework.serve_devui(;
    entities = [],
    port::Int = 8080,
    host::String = "127.0.0.1",
    auto_open::Bool = true,
    title::String = "AgentFramework DevUI",
    dev_mode::Bool = true,
)
    config = DevUIConfig(title=title, port=port, host=host, dev_mode=dev_mode)
    registry = EntityRegistry()
    conv_store = ConversationStore()

    # Register provided entities
    for entity in entities
        info = register_entity!(registry, entity)
        @info "Registered entity" name=info.name type=info.type id=info.id
    end

    # Setup Genie app and routes
    Genie.config.run_as_server = true
    Genie.config.server_host = host
    Genie.config.server_port = port

    setup_routes!(registry, conv_store, config)

    url = "http://$(host):$(port)"
    @info "🤖 AgentFramework DevUI starting" url=url entities=length(list_entities(registry))
    println()
    println("  ╔══════════════════════════════════════════╗")
    println("  ║   🤖 AgentFramework DevUI               ║")
    println("  ║   $(rpad(url, 37))║")
    println("  ╚══════════════════════════════════════════╝")
    println()

    Genie.up(port, host; open_browser=auto_open)
end

end # module AgentFrameworkGenieExt
