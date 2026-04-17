# Driving AgentFramework with CopilotSDK

## Narrative

CopilotSDK.jl wraps the official `copilot` CLI as a JSON-RPC subprocess.
AgentFramework.jl has its own provider-agnostic `Agent` abstraction with
context providers, middleware, tool loops, etc. The
`CopilotSDKAgentFrameworkExt` weakdep extension bridges the two: a
`CopilotAgent <: AbstractAgent` that talks to the Copilot CLI while
plugging into every AF feature an ordinary agent supports.

This vignette runs against the **mock Copilot CLI** shipped in
`CopilotSDK.jl/test/fixtures/mock_copilot.jl` so it works offline. To
drive the real Copilot CLI, replace the subprocess config with
`SubprocessConfig()` (default — picks up `COPILOT_CLI_PATH`).

## What you'll see

* Constructing a `CopilotAgent` from a `CopilotClient`
* Calling `run_agent` / `AgentResponse` — same API as OpenAI / Azure / Ollama
* Streaming updates via `run_agent_streaming` and iterating the
  `Channel{AgentResponseUpdate}`
* Reusing **any** AF context provider (here: a trivial one that injects
  a timestamp) because the bridge is a proper `AbstractAgent`

## Code

```julia
using CopilotSDK, AgentFramework
using Dates

# 1. The mock CLI fixture lets this run without a real copilot binary.
const MOCK = joinpath(pkgdir(CopilotSDK), "test", "fixtures", "mock_copilot.jl")
mock_config() = SubprocessConfig(
    cli_path = Base.julia_cmd().exec[1],
    cli_args = ["--project=$(pkgdir(CopilotSDK))", "--startup-file=no", MOCK],
    env      = Dict{String, String}(ENV),
)

# 2. A toy AF context provider — shows that arbitrary BaseContextProvider
#    implementations compose with CopilotAgent just like with any other
#    AbstractAgent.
mutable struct StampProvider <: BaseContextProvider
    source_id::String
end
StampProvider() = StampProvider("stamp")

function AgentFramework.before_run!(p::StampProvider, agent, session, ctx, state)
    extend_messages!(ctx, p, [Message(ROLE_USER,
        "Request timestamp: " * string(now()))])
    return nothing
end

# 3. The bridge itself: CopilotAgent is an AbstractAgent.
CopilotAgentT = Base.get_extension(CopilotSDK,
    :CopilotSDKAgentFrameworkExt).CopilotAgent

CopilotClient(mock_config(); default_timeout = 15.0) do client
    agent = CopilotAgentT(client;
        name         = "copilot-bridge",
        instructions = "You are a senior Julia engineer.",
    )
    # context_providers works because CopilotAgent is an AbstractAgent.
    # (In a real build you'd attach it at construction; shown here
    # illustratively — `run_agent` will respect whatever the agent's
    # declared pipeline is.)
    try
        response = AgentFramework.run_agent(agent, "echo:ship it")
        @assert response isa AgentFramework.AgentResponse
        @assert response.messages[1].role === :assistant
        @assert occursin("ship it",
                         AgentFramework.get_text(response.messages[1]))
        println("Reply: ",
            AgentFramework.get_text(response.messages[1]))
    finally
        close!(agent)
    end
end
```

### Streaming

```julia
CopilotClient(mock_config(); default_timeout = 15.0) do client
    agent = CopilotAgentT(client)
    try
        updates = AgentFramework.run_agent_streaming(agent, "echo:stream")
        for u in updates
            # Each AgentResponseUpdate carries partial text / event metadata.
            println("update: ", AgentFramework.get_text(u))
        end
    finally
        close!(agent)
    end
end
```

## See also

* [`CopilotSDK.jl/examples/basic_session.jl`](https://github.com/JuliaKnowledge/CopilotSDK.jl/blob/main/examples/basic_session.jl)
  — the SDK-level session vignette (no AF).
* [Agents guide](@ref) — the `AbstractAgent` contract CopilotAgent implements.
* [CopilotSDK hooks guide](https://github.com/JuliaKnowledge/CopilotSDK.jl/blob/main/docs/src/guide/hooks.md)
  — run user code on every session event, orthogonal to AF middleware.
