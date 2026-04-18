# Advanced Topics

This guide covers advanced features of AgentFramework.jl for power users and complex applications.

## Declarative Agents and Workflows

Define agents and workflows from YAML or JSON configuration files, enabling no-code agent definitions and version-controlled configurations.

### Declarative Agents

```julia
using AgentFramework
using Base64

# From YAML
agent = agent_from_yaml("""
name: ResearchAssistant
instructions: You are a research assistant. Provide well-sourced answers.
client:
  type: OllamaChatClient
  model: qwen3:8b
  base_url: http://localhost:11434
tools:
  - search_web
  - summarize
""")

# From JSON
agent = agent_from_json(
    """{"name":"Helper","instructions":"Be helpful.","client":{"type":"OllamaChatClient","model":"qwen3:8b","base_url":"http://localhost:11434"}}""",
)

# Serialize and round-trip through a file
agent_path = joinpath(storage_dir, "researcher.yaml")
agent_to_file(agent, agent_path)
agent = agent_from_file(agent_path)

yaml_str = agent_to_yaml(agent)
```

### Declarative Workflows

```julia
# Register handlers before loading
register_handler!("analyze_handler", (data, ctx) -> send_message(ctx, "Analysis: " * string(data)))
register_handler!("review_handler", (data, ctx) -> yield_output(ctx, "Review: " * string(data)))

workflow = workflow_from_yaml("""
name: ReviewPipeline
executors:
  - id: analyzer
    handler: analyze_handler
  - id: reviewer
    handler: review_handler
edges:
  - source: analyzer
    target: reviewer
start: analyzer
outputs: [reviewer]
""")
```

Use [`register_handler!`](@ref), [`register_tool!`](@ref), [`register_client!`](@ref), and [`register_context_provider!`](@ref) to make components available for declarative loading.

## Message Compaction

For long conversations that exceed context window limits, AgentFramework.jl provides several compaction strategies:

```julia
using AgentFramework

config = CompactionConfig(
    strategy = SLIDING_WINDOW,    # Keep only recent messages
    max_tokens = 4096,            # Token budget
    keep_recent = 20,             # Number of recent messages to keep
)

# Or use a compaction pipeline with multiple strategies
pipeline = CompactionPipeline([
    CompactionConfig(strategy=SELECTIVE_TOOL_CALL),   # First: compact tool calls
    CompactionConfig(strategy=SLIDING_WINDOW, keep_recent=30),  # Then: sliding window
])

# Check if compaction is needed
if needs_compaction(messages, config)
    compacted = compact_messages(messages, config)
end
```

### Available Strategies

| Strategy | Description |
|:---------|:------------|
| `NO_COMPACTION` | No compaction (default) |
| `SLIDING_WINDOW` | Keep only the N most recent messages |
| `DROP_OLDEST` | Drop the oldest messages to fit token budget |
| `SUMMARIZE_OLDEST` | Summarize older messages into a single message |
| `TRUNCATE` | Truncate individual message content |
| `SELECTIVE_TOOL_CALL` | Compact verbose tool call/result pairs |
| `TOOL_RESULT_ONLY` | Replace tool calls with just their results |

### Token Estimation

```julia
tokens = estimate_tokens("Hello, world!")
msg_tokens = estimate_message_tokens(message)
total = estimate_messages_tokens(messages)
```

## Content Filtering

Parse and handle content filter results from providers:

```julia
using AgentFramework

# Check filter results
results = ContentFilterResults(
    results = [
        ContentFilterResult(
            category = FILTER_VIOLENCE,
            severity = FILTER_HIGH,
            filtered = true,
        ),
    ],
    blocked = true,
)
if is_blocked(results)
    categories = get_filtered_categories(results)
    severity = max_severity(results)
    @warn "Content blocked" categories severity
end
```

### Severity Levels

| Level | Constant |
|:------|:---------|
| Safe | `FILTER_SAFE` |
| Low | `FILTER_LOW` |
| Medium | `FILTER_MEDIUM` |
| High | `FILTER_HIGH` |

### Filter Categories

`FILTER_HATE`, `FILTER_SELF_HARM`, `FILTER_SEXUAL`, `FILTER_VIOLENCE`, `FILTER_PROFANITY`, `FILTER_JAILBREAK`, `FILTER_PROTECTED_MATERIAL`, `FILTER_CUSTOM`

## Multimodal Content

Send images and audio to models that support them:

```julia
using AgentFramework

# Image from file
img = image_content(photo_path)

# Image from URL
img_url = image_url_content("https://example.com/photo.jpg"; media_type="image/jpeg")

# Image from base64
img_b64 = image_content(base64decode(base64_data); media_type="image/png")

# Audio
audio = audio_content(recording_path)

# Include in a message
msg = Message(:user, [text_content("What's in this image?"), img])
response = run_agent(agent, msg)
```

Helper functions:

| Function | Description |
|:---------|:------------|
| [`image_content`](@ref) | Create image content from file or base64 |
| [`image_url_content`](@ref) | Reference an image by URL |
| [`audio_content`](@ref) | Create audio content from file or data |
| [`file_content`](@ref) | Generic file content |
| [`detect_mime_type`](@ref) | Detect MIME type from file path |

## Scoped State Management

Scoped state provides hierarchical state management for workflows:

```julia
using AgentFramework

store = ScopedStateStore()

# Local state (per-executor)
set_local!(store, "executor_1", "counter", 0)
val = get_local(store, "executor_1", "counter")

# Broadcast state (visible to all executors)
set_broadcast!(store, "executor_1", "shared_config", Dict("mode" => "fast"))
config = get_broadcast(store, "shared_config")

# Workflow-level state
set_workflow_state!(store, "phase", "planning")
phase = get_workflow_state(store, "phase")
```

### State Scopes

| Scope | Constant | Visibility |
|:------|:---------|:-----------|
| Local | `SCOPE_LOCAL` | Single executor only |
| Broadcast | `SCOPE_BROADCAST` | All executors |
| Workflow | `SCOPE_WORKFLOW` | Workflow-level state |

## Skills Framework

Skills are reusable bundles of instructions, tools, and resources:

```julia
using AgentFramework

# Load skills from a directory
source = DirectorySkillSource(directory=skills_dir)
skills = get_skills(source)

# Create a skills provider for agents
provider = SkillsProvider()
load_skills!(provider, source)
agent = Agent(
    client = client,
    context_providers = [provider],
)
```

### Skill Structure

Skills are defined in Markdown files:

```markdown
# Web Search

Search the web for information.

## Instructions

When the user asks a question, search the web first.

## Tools

- search_web
- summarize_results
```

### Skill Sources

| Source | Description |
|:------|:------------|
| [`StaticSkillSource`](@ref) | Fixed list of skills |
| [`DirectorySkillSource`](@ref) | Load from a directory |
| [`DeduplicatingSkillSource`](@ref) | Deduplicate across sources |
| [`FilteringSkillSource`](@ref) | Filter skills by criteria |
| [`AggregatingSkillSource`](@ref) | Combine multiple sources |

## MCP (Model Context Protocol) Integration

Connect to external tool servers via the Model Context Protocol:

```julia
using AgentFramework

# Connect via stdio
mcp_client = StdioMCPClient(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem"])
connect!(mcp_client)

# List available tools
tools = list_tools(mcp_client)

# Convert MCP tools to FunctionTools for use with agents
function_tools = load_mcp_tools(mcp_client)

agent = Agent(
    client = client,
    tools = function_tools,
)

# Or use the convenience helper
agent = with_mcp_client(HTTPMCPClient(url="http://localhost:3000/mcp")) do connected
    Agent(client = client, tools = load_mcp_tools(connected))
end

# Clean up
close_mcp!(mcp_client)
```

### HTTP MCP Client

```julia
mcp_client = HTTPMCPClient(url="http://localhost:3000/mcp")
connect!(mcp_client)
```

### MCP Resources and Prompts

```julia
# List and read resources
resources = list_resources(mcp_client)
content = read_resource(mcp_client, resource_uri)

# List and use prompts
prompts = list_prompts(mcp_client)
prompt = get_prompt(mcp_client, "my_prompt", Dict{String, Any}("arg" => "value"))
```

## Evaluation Framework

Test agent quality with assertion-based evaluation:

```julia
using AgentFramework

# Build a local evaluator from one or more checks
evaluator = LocalEvaluator(keyword_check("Docs example response"))

# Evaluate an agent
results = evaluate_agent(
    agent = agent,
    queries = ["What is 2+2?"],
    evaluators = evaluator,
)
summary = only(results)

# Check results
println("Passed: $(eval_passed(summary)) / $(eval_total(summary))")
if !all_passed(summary)
    for item_result in summary.items
        if is_failed(item_result)
            println("Failed: ", item_result.input)
        end
    end
end
```

### Available Checks

| Check | Description |
|:------|:------------|
| [`keyword_check`](@ref) | Response contains expected keywords |
| [`tool_called_check`](@ref) | A specific tool was called |
| [`tool_calls_present`](@ref) | Any tool calls were made |
| [`tool_call_args_match`](@ref) | Tool call arguments match expectations |

### Evaluating Workflows

```julia
workflow_results = evaluate_workflow(
    workflow = workflow,
    queries = ["Draft a launch announcement"],
    evaluators = LocalEvaluator(keyword_check("Review")),
)
```

## Handoffs Between Agents

Handoffs enable one agent to transfer control to another:

```julia
using AgentFramework

transfer_tool = HandoffTool(
    name = "transfer_to_specialist",
    description = "Transfer to the specialist for technical questions.",
    target = specialist_agent,
)

main_agent = Agent(
    client = client,
    instructions = "Route technical questions to the specialist.",
    tools = [transfer_tool],
)

# normalize_agent_tools converts HandoffTools to FunctionTools for the LLM
tools = normalize_agent_tools(main_agent.tools)
```

## Telemetry and Observability

AgentFramework.jl provides OpenTelemetry-aligned telemetry:

```julia
using AgentFramework

# Use the in-memory backend for testing
backend = InMemoryTelemetryBackend()

# Instrument an agent
instrument!(agent, backend)

# Or add telemetry middleware manually
agent = Agent(
    client = client,
    agent_middlewares = [telemetry_agent_middleware(backend)],
    chat_middlewares = [telemetry_chat_middleware(backend)],
    function_middlewares = [telemetry_function_middleware(backend)],
)

# After running the agent, inspect spans
spans = get_spans(backend)
for span in spans
    println("$(span.name): $(duration_ms(span))ms")
end

# Clean up
clear_spans!(backend)
```

### Telemetry Spans

Each [`TelemetrySpan`](@ref) records:
- Operation name and duration
- Events and attributes
- Parent-child relationships

## Serialization and Persistence

Serialize agents, messages, and sessions for storage or transmission:

```julia
using AgentFramework

# Messages
json = serialize_to_json(message)
message = deserialize_from_json(json)

# Sessions
dict = serialize_to_dict(session)
session = deserialize_from_dict(dict)

# Register custom types for serialization
struct MyCustomType
    name::String
end

struct MyStateType
    count::Int
end

register_type!(MyCustomType)
register_state_type!(MyStateType)
```

## Settings Management

Load configuration from environment variables, `.env` files, or TOML:

```julia
using AgentFramework

settings = Settings()
load_from_env!(settings)
load_from_dotenv!(settings, dotenv_path)
load_from_toml!(settings, config_toml_path)

api_key = get_secret(settings, "OPENAI_API_KEY")
model = get_setting(settings, "MODEL", "qwen3:8b")
```

`SecretString` wraps sensitive values to prevent accidental logging.

## Structured Output

Parse LLM responses into typed structs:

```julia
using AgentFramework

Base.@kwdef struct Recipe
    name::String
    ingredients::Vector{String}
    steps::Vector{String}
    prep_time_minutes::Int
end

# Generate JSON Schema from type
schema = schema_from_type(Recipe)

# Parse agent response into struct
result = run_agent(Recipe, agent, "Give me a recipe for chocolate cake")
recipe = result.value
println(recipe.name)
println(recipe.ingredients)
```

## Next Steps

- [API Reference](../api/agents.md) — Complete API documentation for all types and functions
- [Workflows](workflows.md) — Multi-agent orchestration patterns
- [Providers](@ref) — Provider-specific advanced features
