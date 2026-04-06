using Documenter
using AgentFramework

makedocs(
    sitename = "AgentFramework.jl",
    modules = [AgentFramework],
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://sdwfrost.github.io/AgentFramework.jl",
        assets = String[],
        size_threshold = 500_000,  # some API pages are large
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Agents" => "guide/agents.md",
            "Tools" => "guide/tools.md",
            "Middleware" => "guide/middleware.md",
            "Sessions & Memory" => "guide/sessions.md",
            "Workflows" => "guide/workflows.md",
            "Providers" => "guide/providers.md",
            "Streaming" => "guide/streaming.md",
            "Advanced Topics" => "guide/advanced.md",
        ],
        "API Reference" => [
            "Agents" => "api/agents.md",
            "Content" => "api/content.md",
            "Messages" => "api/messages.md",
            "Tools" => "api/tools.md",
            "Sessions & Context" => "api/sessions.md",
            "Chat Client" => "api/chat_client.md",
            "Middleware" => "api/middleware.md",
            "Capabilities" => "api/capabilities.md",
            "Resilience" => "api/resilience.md",
            "Structured Output" => "api/structured_output.md",
            "Workflows" => "api/workflows.md",
            "Declarative" => "api/declarative.md",
            "Multimodal" => "api/multimodal.md",
            "Compaction" => "api/compaction.md",
            "MCP" => "api/mcp.md",
            "Evaluation" => "api/evaluation.md",
            "Skills" => "api/skills.md",
            "Handoffs" => "api/handoffs.md",
            "Telemetry" => "api/telemetry.md",
            "Serialization" => "api/serialization.md",
            "Exceptions" => "api/exceptions.md",
        ],
        "Submodules" => [
            "A2A Protocol" => "submodules/a2a.md",
            "Hosting" => "submodules/hosting.md",
            "Mem0 Integration" => "submodules/mem0.md",
            "Bedrock" => "submodules/bedrock.md",
            "Coding Agents" => "submodules/coding_agents.md",
        ],
    ],
    warnonly = [:missing_docs, :cross_references, :docs_block],
)

deploydocs(
    repo = "github.com/sdwfrost/AgentFramework.jl.git",
    push_preview = true,
)
