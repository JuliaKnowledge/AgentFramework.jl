# AgentFramework.jl

AgentFramework.jl is an idiomatic Julia framework for building, orchestrating, and testing AI agents. It provides chat clients, tools, middleware, workflows, MCP integration, structured output, telemetry, and session storage in a Julia-first API.

Planned repository: <https://github.com/JuliaKnowledge/AgentFramework.jl>

## Installation

Until the package is registered:

```julia
using Pkg
Pkg.develop(path="path/to/AgentFramework.jl")
```

If you want the optional Azure OpenAI credential extension before registration, add AzureIdentity.jl separately:

```julia
using Pkg
Pkg.develop(path="path/to/AzureIdentity.jl")
Pkg.develop(path="path/to/AgentFramework.jl")
```

After registration:

```julia
using Pkg
Pkg.add("AgentFramework")
```

## Highlights

- Agent, tool, and middleware primitives for iterative agent workflows
- OpenAI, Azure OpenAI, Ollama, and Anthropic chat clients
- Workflow graphs with checkpointing, fan-out, fan-in, and human-in-the-loop support
- MCP client integration, skills, structured output, and telemetry hooks
- Optional Azure OpenAI credential integration through AzureIdentity.jl

## Notes

- The Graph RAG vignettes also use RDFLib.jl, but RDFLib.jl is not required to load or test AgentFramework.jl itself.
- AzureIdentity.jl is developed alongside this package and should be registered before the optional Azure extension is published.

## License

MIT License. Copyright (c) 2026 Simon Frost.
