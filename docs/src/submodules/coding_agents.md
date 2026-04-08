# Coding Agents

The `AgentFramework.CodingAgents` submodule provides CLI-backed chat clients for **GitHub Copilot** and **Claude Code**. These clients invoke the respective command-line tools as subprocesses, allowing AgentFramework.jl agents to leverage powerful coding assistants with code interpretation, file search, and web search capabilities.

## Overview

This module exports two [`AbstractChatClient`](@ref) implementations:

- **`GitHubCopilotChatClient`** — wraps the `copilot` CLI
- **`ClaudeCodeChatClient`** — wraps the `claude` CLI

Both clients:

- Format AgentFramework messages into a prompt and invoke the CLI
- Parse JSON output into standard [`ChatResponse`](@ref) / `ChatResponseUpdate` types
- Support streaming via `get_response_streaming`
- Support session resumption via thread/conversation IDs
- Declare rich capabilities (code interpretation, file search, web search)

## Quick Start

### GitHub Copilot

```julia
using AgentFramework
using AgentFramework.CodingAgents

agent = Agent(
    name = "copilot-agent",
    instructions = "You are a coding assistant powered by GitHub Copilot.",
    client = GitHubCopilotChatClient(),
)

response = run_agent(agent, "Write a Julia function to compute fibonacci numbers")
println(get_text(response))
```

### Claude Code

```julia
using AgentFramework
using AgentFramework.CodingAgents

agent = Agent(
    name = "claude-agent",
    instructions = "You are a coding assistant powered by Claude.",
    client = ClaudeCodeChatClient(model = "claude-sonnet-4-20250514"),
)

response = run_agent(agent, "Review this code for bugs")
println(get_text(response))
```

## GitHubCopilotChatClient

```julia
Base.@kwdef mutable struct GitHubCopilotChatClient <: AbstractChatClient
    model::String = ""                          # or set GITHUB_COPILOT_MODEL env var
    cli_path::String = "copilot"                # or set GITHUB_COPILOT_CLI_PATH env var
    cwd::Union{Nothing, String} = nothing       # working directory for the CLI
    agent::Union{Nothing, String} = nothing      # named agent to use
    reasoning_effort::Union{Nothing, String} = nothing
    allow_all_tools::Bool = true
    allow_all_paths::Bool = false
    allow_all_urls::Bool = false
    no_ask_user::Bool = true
    autopilot::Bool = false
    add_dirs::Vector{String} = String[]
    available_tools::Vector{String} = String[]
    excluded_tools::Vector{String} = String[]
    allow_tools::Vector{String} = String[]
    deny_tools::Vector{String} = String[]
    allow_urls::Vector{String} = String[]
    deny_urls::Vector{String} = String[]
    additional_mcp_config::Vector{String} = String[]
    github_mcp_tools::Vector{String} = String[]
    github_mcp_toolsets::Vector{String} = String[]
    enable_all_github_mcp_tools::Bool = false
    config_dir::Union{Nothing, String} = nothing
    log_dir::Union{Nothing, String} = nothing
    log_level::Union{Nothing, String} = nothing
    cli_args::Vector{String} = String[]
    env::Dict{String, String} = Dict{String, String}()
    capture_runner::Function                     # command execution function
    stream_runner::Function                      # streaming execution function
end
```

### Capabilities

| Capability | Supported |
|------------|-----------|
| Streaming | ✅ |
| Code Interpreter | ✅ |
| File Search | ✅ |
| Web Search | ✅ |
| Tool Calling | ❌ (tools are handled natively by the CLI) |
| Structured Output | ❌ |

### Configuration

**Model:** Set via `model` field, `GITHUB_COPILOT_MODEL` env var, or `ChatOptions.model`.

**CLI Path:** Set via `cli_path` field or `GITHUB_COPILOT_CLI_PATH` env var. Defaults to `"copilot"`.

**Per-request overrides:** Additional settings can be passed via `ChatOptions.additional["github_copilot"]`:

```julia
options = ChatOptions(
    model = "gpt-4o",
    additional = Dict(
        "github_copilot" => Dict(
            "reasoning_effort" => "high",
            "allow_all_paths" => true,
            "add_dirs" => ["/path/to/extra/dir"],
            "env" => Dict("CUSTOM_VAR" => "value"),
        ),
    ),
)
```

### Tool and Permission Control

| Field | CLI Flag | Description |
|-------|----------|-------------|
| `allow_all_tools` | `--allow-all-tools` | Allow all tool usage (default: `true`) |
| `allow_all_paths` | `--allow-all-paths` | Allow file access to all paths |
| `allow_all_urls` | `--allow-all-urls` | Allow network access to all URLs |
| `no_ask_user` | `--no-ask-user` | Don't prompt for user confirmation |
| `autopilot` | `--autopilot` | Enable autopilot mode |
| `available_tools` | `--available-tools` | Limit available tools |
| `excluded_tools` | `--excluded-tools` | Exclude specific tools |
| `allow_tools` | `--allow-tool` | Explicitly allow specific tools |
| `deny_tools` | `--deny-tool` | Deny specific tools |

### MCP Integration

```julia
client = GitHubCopilotChatClient(
    additional_mcp_config = ["/path/to/mcp-config.json"],
    github_mcp_tools = ["issues", "pull_requests"],
    github_mcp_toolsets = ["code_review"],
    # or enable everything:
    enable_all_github_mcp_tools = true,
)
```

### Session Resumption

The client supports resuming previous conversations using the session/thread ID returned in the response:

```julia
response = run_agent(agent, "Start a task"; session = session)
# response.conversation_id contains the Copilot session ID

# Resume later using the conversation_id through ChatOptions
```

## ClaudeCodeChatClient

```julia
Base.@kwdef mutable struct ClaudeCodeChatClient <: AbstractChatClient
    model::String = ""                          # or set CLAUDE_AGENT_MODEL env var
    cli_path::String = "claude"                  # or set CLAUDE_AGENT_CLI_PATH env var
    cwd::Union{Nothing, String} = nothing
    agent::Union{Nothing, String} = nothing
    permission_mode::Union{Nothing, String} = nothing
    max_turns::Union{Nothing, Int} = nothing
    max_budget_usd::Union{Nothing, Float64} = nothing
    effort::Union{Nothing, String} = nothing
    add_dirs::Vector{String} = String[]
    available_tools::Vector{String} = String[]
    allowed_tools::Vector{String} = String[]
    disallowed_tools::Vector{String} = String[]
    mcp_config::Vector{String} = String[]
    settings::Union{Nothing, String} = nothing
    append_system_prompt::Union{Nothing, String} = nothing
    cli_args::Vector{String} = String[]
    env::Dict{String, String} = Dict{String, String}()
    capture_runner::Function
    stream_runner::Function
end
```

### Capabilities

| Capability | Supported |
|------------|-----------|
| Streaming | ✅ |
| Code Interpreter | ✅ |
| File Search | ✅ |
| Web Search | ✅ |
| Tool Calling | ❌ (tools are handled natively by the CLI) |
| Structured Output | ✅ (via `--json-schema`) |

### Configuration

**Model:** Set via `model` field, `CLAUDE_AGENT_MODEL` env var, or `ChatOptions.model`.

**CLI Path:** Set via `cli_path` field or `CLAUDE_AGENT_CLI_PATH` env var. Defaults to `"claude"`.

**Per-request overrides** via `ChatOptions.additional["claude_code"]`:

```julia
options = ChatOptions(
    additional = Dict(
        "claude_code" => Dict(
            "max_turns" => 5,
            "max_budget_usd" => 1.0,
            "effort" => "high",
            "permission_mode" => "auto",
            "env" => Dict("CUSTOM_VAR" => "value"),
        ),
    ),
)
```

### Cost and Turn Limits

| Field | CLI Flag | Description |
|-------|----------|-------------|
| `max_turns` | `--max-turns` | Maximum number of agentic turns |
| `max_budget_usd` | `--max-budget-usd` | Maximum spend in USD |
| `effort` | `--effort` | Effort level (`"low"`, `"medium"`, `"high"`) |

### Structured Output

Claude Code supports structured output through JSON schema:

```julia
agent = Agent(
    name = "structured-agent",
    client = ClaudeCodeChatClient(),
)

options = ChatOptions(
    response_format = Dict(
        "type" => "object",
        "properties" => Dict(
            "summary" => Dict("type" => "string"),
            "score" => Dict("type" => "number"),
        ),
    ),
)
```

### Tool Control

| Field | CLI Flag | Description |
|-------|----------|-------------|
| `available_tools` | `--tools` | Limit available tools |
| `allowed_tools` | `--allowed-tools` | Explicitly allow tools |
| `disallowed_tools` | `--disallowed-tools` | Explicitly disallow tools |
| `mcp_config` | `--mcp-config` | MCP configuration files |
| `permission_mode` | `--permission-mode` | Permission mode for tool usage |

## Streaming

Both clients support streaming via `get_response_streaming`, which returns a `Channel{ChatResponseUpdate}`:

```julia
using AgentFramework
using AgentFramework.CodingAgents

agent = Agent(
    name = "streaming-agent",
    client = GitHubCopilotChatClient(),
)

stream = run_agent_streaming(agent, "Explain quicksort step by step")
for update in stream
    print(get_text(update))
end
println()
```

For Claude Code, streaming uses the `--output-format stream-json` flag with `--verbose` and `--include-partial-messages`, emitting `content_block_delta` events for incremental text and thinking content.

For GitHub Copilot, streaming parses newline-delimited JSON events including `assistant.message_delta`, `assistant.reasoning_delta`, and `result` events.

## Usage Details

Both clients report usage information in the response:

### GitHub Copilot

```julia
response = run_agent(agent, "Hello")
usage = response.usage_details
# usage.output_tokens — token count
# usage.additional["premium_requests"] — premium request count
# usage.additional["total_api_duration_ms"] — API duration
```

### Claude Code

```julia
response = run_agent(agent, "Hello")
usage = response.usage_details
# usage.input_tokens, usage.output_tokens
# usage.additional["cache_creation_input_tokens"]
# usage.additional["cache_read_input_tokens"]
```

## Error Handling

Both clients raise standard AgentFramework chat client exceptions:

- `ChatClientError` — CLI exited with a non-zero exit code
- `ChatClientInvalidResponseError` — CLI returned unparseable output
- `ChatClientInvalidRequestError` — invalid options (e.g., unsupported temperature overrides)

```julia
try
    response = run_agent(agent, "Hello")
catch e
    if e isa ChatClientError
        println("CLI error: ", e.message)
    else
        rethrow()
    end
end
```

!!! note "CLI Prerequisites"
    Both clients require their respective CLI tools to be installed and authenticated:
    - **GitHub Copilot:** Install the `copilot` CLI and run `copilot auth login`
    - **Claude Code:** Install the `claude` CLI and authenticate with your Anthropic account
