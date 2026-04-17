# Runnable companion to docs/src/vignettes/copilot_with_agentframework.md
#
# Drives CopilotSDK's CopilotAgent through AgentFramework's Agent API
# against the mock Copilot CLI so the vignette works offline.
#
# Run:  julia --project=. examples/copilot_with_agentframework.jl

using CopilotSDK, AgentFramework, Test

const MOCK = joinpath(pkgdir(CopilotSDK), "test", "fixtures", "mock_copilot.jl")

mock_config() = SubprocessConfig(
    cli_path = Base.julia_cmd().exec[1],
    cli_args = ["--project=$(pkgdir(CopilotSDK))", "--startup-file=no", MOCK],
    env      = Dict{String, String}(ENV),
)

const Ext = Base.get_extension(CopilotSDK, :CopilotSDKAgentFrameworkExt)
@assert Ext !== nothing "CopilotSDKAgentFrameworkExt not loaded"
const CopilotAgentT = Ext.CopilotAgent

@testset "CopilotAgent via AF run_agent" begin
    CopilotClient(mock_config(); default_timeout = 15.0) do client
        agent = CopilotAgentT(client;
            name         = "copilot-bridge",
            instructions = "You are a senior Julia engineer.",
        )
        try
            resp = AgentFramework.run_agent(agent, "echo:ship it")
            @test resp isa AgentFramework.AgentResponse
            @test length(resp.messages) == 1
            @test resp.messages[1].role === :assistant
            text = AgentFramework.get_text(resp.messages[1])
            @info "Copilot reply via AF" text
            @test occursin("ship it", text)
            @test resp.conversation_id == session_id(agent.session)
        finally
            close!(agent)
        end
    end
end

@testset "CopilotAgent streaming via AF" begin
    CopilotClient(mock_config(); default_timeout = 15.0) do client
        agent = CopilotAgentT(client)
        try
            updates = AgentFramework.run_agent_streaming(agent, "echo:stream")
            collected = String[]
            for u in updates
                t = AgentFramework.get_text(u)
                isempty(t) && continue
                push!(collected, t)
            end
            @info "streamed chunks" n=length(collected)
            @test any(occursin("stream", c) for c in collected)
        finally
            close!(agent)
        end
    end
end

println("\nPASS — Copilot × AgentFramework vignette")
