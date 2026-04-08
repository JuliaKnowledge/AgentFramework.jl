# Middleware

Middleware provides a composable way to intercept and modify behavior at three levels of the agent pipeline. Each middleware layer follows the same pattern: receive a context and a `next` function, optionally modify the context, call `next` to proceed, and optionally modify the result.

## Three Middleware Layers

AgentFramework.jl has three middleware layers, each wrapping a different part of the pipeline:

```
┌─ Agent Middleware ──────────────────────────────────────────────┐
│  Wraps the entire run_agent() call                             │
│                                                                │
│  ┌─ Chat Middleware ────────────────────────────────────────┐   │
│  │  Wraps each LLM service call (may run multiple times    │   │
│  │  in a tool loop)                                        │   │
│  │                                                         │   │
│  │  ┌─ Function Middleware ─────────────────────────────┐   │   │
│  │  │  Wraps each individual tool invocation            │   │   │
│  │  └──────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

| Layer | Context Type | Wraps | Use Cases |
|:------|:-------------|:------|:----------|
| Agent | [`AgentContext`](@ref) | Entire `run_agent()` call | Logging, telemetry, auth, caching |
| Chat | [`ChatContext`](@ref) | Each LLM service call | Retry, rate limiting, request/response transforms |
| Function | [`FunctionInvocationContext`](@ref) | Each tool invocation | Validation, sandboxing, result caching |

## Context Types

### AgentContext

[`AgentContext`](@ref) is passed through the agent middleware pipeline:

```julia
Base.@kwdef mutable struct AgentContext
    agent::Any                              # The agent being invoked
    messages::Vector{Message}               # Input messages
    session::Union{Nothing, AgentSession}   # Current session
    options::Dict{String, Any}              # Agent run options
    stream::Bool                            # Whether this is streaming
    metadata::Dict{String, Any}            # Shared data between middleware
    result::Any                             # Set after execution
end
```

### ChatContext

[`ChatContext`](@ref) is passed through the chat middleware pipeline:

```julia
Base.@kwdef mutable struct ChatContext
    messages::Vector{Message}               # Messages sent to the LLM
    options::ChatOptions                    # Chat options for this request
    stream::Bool                            # Whether this is streaming
    metadata::Dict{String, Any}            # Shared data between middleware
    result::Any                             # Chat response (set after next())
end
```

### FunctionInvocationContext

[`FunctionInvocationContext`](@ref) is passed through the function middleware pipeline:

```julia
Base.@kwdef mutable struct FunctionInvocationContext
    tool::Union{Nothing, FunctionTool}      # The tool being invoked
    arguments::Dict{String, Any}            # Parsed function arguments
    call_id::String                         # Tool call ID from the LLM
    metadata::Dict{String, Any}            # Shared data between middleware
    result::Any                             # Function result (set after next())
end
```

## Creating Middleware Functions

A middleware function takes `(context, next)` and must call `next(context)` to continue the pipeline:

### Agent Middleware

```julia
function logging_agent_middleware(ctx::AgentContext, next::Function)
    @info "Agent invoked" agent=ctx.agent.name messages=length(ctx.messages)
    start = time()

    result = next(ctx)  # Continue the pipeline

    elapsed = time() - start
    @info "Agent completed" duration=elapsed
    return result
end

agent = Agent(
    client = client,
    agent_middlewares = [logging_agent_middleware],
)
```

### Chat Middleware

```julia
function token_logging_middleware(ctx::ChatContext, next::Function)
    @info "Sending $(length(ctx.messages)) messages to LLM"

    result = next(ctx)  # Call the LLM

    if result isa ChatResponse && result.usage_details !== nothing
        @info "Token usage" input=result.usage_details.input_tokens output=result.usage_details.output_tokens
    end
    return result
end

agent = Agent(
    client = client,
    chat_middlewares = [token_logging_middleware],
)
```

### Function Middleware

```julia
function tool_logging_middleware(ctx::FunctionInvocationContext, next::Function)
    @info "Calling tool" name=ctx.tool.name arguments=ctx.arguments

    result = next(ctx)  # Execute the tool

    @info "Tool returned" name=ctx.tool.name result=result
    return result
end

agent = Agent(
    client = client,
    function_middlewares = [tool_logging_middleware],
)
```

## The @middleware Macro

The [`@middleware`](@ref) macro provides syntactic sugar for defining middleware:

```julia
@middleware :chat my_middleware function(ctx, next)
    @info "Before"
    result = next(ctx)
    @info "After"
    return result
end
```

## Middleware Pipeline Ordering

Middleware executes in the order they appear in the vector. The first middleware is the outermost wrapper:

```julia
agent = Agent(
    client = client,
    chat_middlewares = [
        auth_middleware,      # Runs first (outermost)
        retry_middleware,     # Runs second
        logging_middleware,   # Runs third (innermost, closest to the handler)
    ],
)
```

Execution flow:

```
Request → auth → retry → logging → [LLM call] → logging → retry → auth → Response
```

## Common Patterns

### Retry with Exponential Backoff

AgentFramework.jl includes a built-in retry middleware via [`retry_chat_middleware`](@ref):

```julia
config = RetryConfig(
    max_retries = 3,
    initial_delay = 1.0,
    max_delay = 60.0,
    multiplier = 2.0,
    jitter = 0.1,
    retryable_status_codes = [429, 500, 502, 503, 504],
)

agent = Agent(
    client = client,
    chat_middlewares = [retry_chat_middleware(config)],
)
```

Or use the convenience builder:

```julia
agent = Agent(client=client)
with_retry!(agent)  # Adds retry middleware with defaults
```

### Rate Limiting

Use [`TokenBucketRateLimiter`](@ref) with [`rate_limit_chat_middleware`](@ref):

```julia
limiter = TokenBucketRateLimiter(requests_per_second=1.0, burst=10)

agent = Agent(
    client = client,
    chat_middlewares = [rate_limit_chat_middleware(limiter)],
)
```

Or with the convenience builder:

```julia
agent = Agent(client=client)
limiter = TokenBucketRateLimiter(requests_per_second=1.0, burst=10)
with_rate_limit!(agent, limiter)
```

### Content Filtering

```julia
function content_filter_middleware(ctx::ChatContext, next::Function)
    # Check input messages for blocked content
    for msg in ctx.messages
        text = get_text(msg)
        if occursin("forbidden", text)
            return ChatResponse(
                messages = [Message(:assistant, "I cannot process that request.")],
                finish_reason = CONTENT_FILTER,
            )
        end
    end
    return next(ctx)
end
```

### Request/Response Transformation

```julia
function add_system_context(ctx::ChatContext, next::Function)
    # Prepend additional context to messages
    timestamp = string(Dates.now())
    system_msg = Message(:system, "Current time: $(timestamp)")
    pushfirst!(ctx.messages, system_msg)
    return next(ctx)
end
```

### Tool Argument Validation

```julia
function validate_args_middleware(ctx::FunctionInvocationContext, next::Function)
    # Validate arguments before tool execution
    if ctx.tool.name == "delete_file" && !haskey(ctx.arguments, "confirm")
        ctx.arguments["confirm"] = false
        @warn "Auto-injected confirm=false for delete_file"
    end
    return next(ctx)
end
```

## MiddlewareTermination

Use [`terminate_pipeline`](@ref) to short-circuit the middleware chain and return a result immediately, skipping all remaining middleware and the handler:

```julia
function cache_middleware(ctx::ChatContext, next::Function)
    cache_key = hash(ctx.messages)
    cached = get(RESPONSE_CACHE, cache_key, nothing)
    if cached !== nothing
        # Skip the LLM call entirely
        terminate_pipeline(cached, message="Cache hit")
    end

    result = next(ctx)

    # Cache the response
    RESPONSE_CACHE[cache_key] = result
    return result
end
```

[`terminate_pipeline`](@ref) throws a [`MiddlewareTermination`](@ref) exception that is caught by the pipeline executor. The `result` field becomes the return value of the pipeline.

## Built-in Middleware

| Middleware | Layer | Description |
|:-----------|:------|:------------|
| [`retry_chat_middleware`](@ref) | Chat | Retry failed LLM calls with exponential backoff |
| [`rate_limit_chat_middleware`](@ref) | Chat | Token bucket rate limiting |
| [`telemetry_agent_middleware`](@ref) | Agent | OpenTelemetry-aligned span tracking |
| [`telemetry_chat_middleware`](@ref) | Chat | LLM call telemetry |
| [`telemetry_function_middleware`](@ref) | Function | Tool invocation telemetry |

## Applying Middleware Programmatically

The framework uses [`apply_agent_middleware`](@ref), [`apply_chat_middleware`](@ref), and [`apply_function_middleware`](@ref) internally, but you can use them directly:

```julia
ctx = ChatContext(messages=msgs, options=ChatOptions())
handler = (c) -> get_response(client, c.messages, c.options)
result = apply_chat_middleware([mw1, mw2], ctx, handler)
```

## Next Steps

- [Sessions & Memory](@ref) — Context providers that work alongside middleware
- [Agents](@ref) — How middleware integrates into the agent execution pipeline
- [Streaming](@ref) — Middleware behavior during streaming
