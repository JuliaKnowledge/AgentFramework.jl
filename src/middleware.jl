# Middleware pipeline for AgentFramework.jl
# Three-layer middleware: Agent, Chat, and Function invocation.
# Each follows the intercept pattern: request → middleware → handler → middleware → response.

# ── Context Types ────────────────────────────────────────────────────────────

"""
    AgentContext

Context passed through the agent middleware pipeline.

# Fields
- `agent`: The agent being invoked.
- `messages::Vector{Message}`: Input messages.
- `session::Union{Nothing, AgentSession}`: Current session.
- `options::Dict{String, Any}`: Agent run options.
- `stream::Bool`: Whether this is a streaming invocation.
- `metadata::Dict{String, Any}`: Shared data between middleware.
- `result::Any`: Execution result (set after `call_next`).
"""
Base.@kwdef mutable struct AgentContext
    agent::Any = nothing
    messages::Vector{Message} = Message[]
    session::Union{Nothing, AgentSession} = nothing
    options::Dict{String, Any} = Dict{String, Any}()
    stream::Bool = false
    metadata::Dict{String, Any} = Dict{String, Any}()
    result::Any = nothing
end

"""
    ChatContext

Context passed through the chat middleware pipeline.

# Fields
- `messages::Vector{Message}`: Messages being sent to the chat client.
- `options::ChatOptions`: Chat options for this request.
- `stream::Bool`: Whether this is a streaming invocation.
- `metadata::Dict{String, Any}`: Shared data between middleware.
- `result::Any`: Chat response (set after `call_next`).
"""
Base.@kwdef mutable struct ChatContext
    messages::Vector{Message} = Message[]
    options::ChatOptions = ChatOptions()
    stream::Bool = false
    metadata::Dict{String, Any} = Dict{String, Any}()
    result::Any = nothing
end

"""
    FunctionInvocationContext

Context passed through the function/tool middleware pipeline.

# Fields
- `tool::FunctionTool`: The tool being invoked.
- `arguments::Dict{String, Any}`: Parsed function arguments.
- `call_id::String`: The tool call ID from the LLM.
- `metadata::Dict{String, Any}`: Shared data between middleware.
- `result::Any`: Function execution result (set after `call_next`).
"""
Base.@kwdef mutable struct FunctionInvocationContext
    tool::Union{Nothing, FunctionTool} = nothing
    arguments::Dict{String, Any} = Dict{String, Any}()
    call_id::String = ""
    metadata::Dict{String, Any} = Dict{String, Any}()
    result::Any = nothing
end

# ── Middleware Function Types ────────────────────────────────────────────────

"""
    AgentMiddlewareFunc

A function `(ctx::AgentContext, next::Function) -> Any` that wraps agent execution.
Call `next(ctx)` to continue the pipeline.
"""
const AgentMiddlewareFunc = Function

"""
    ChatMiddlewareFunc

A function `(ctx::ChatContext, next::Function) -> Any` that wraps chat client calls.
Call `next(ctx)` to continue the pipeline.
"""
const ChatMiddlewareFunc = Function

"""
    FunctionMiddlewareFunc

A function `(ctx::FunctionInvocationContext, next::Function) -> Any` that wraps tool invocations.
Call `next(ctx)` to continue the pipeline.
"""
const FunctionMiddlewareFunc = Function

# ── Middleware Pipeline Execution ────────────────────────────────────────────

"""
    apply_agent_middleware(middlewares, ctx, handler) -> Any

Execute the agent middleware chain, ending with the handler.
Middlewares are called in order; each receives `(ctx, next)`.
"""
function apply_agent_middleware(middlewares::Vector, ctx::AgentContext, handler::Function)
    if isempty(middlewares)
        return handler(ctx)
    end

    function build_chain(idx::Int)
        if idx > length(middlewares)
            return handler
        end
        mw = middlewares[idx]
        return function(c::AgentContext)
            next_fn = build_chain(idx + 1)
            return mw(c, next_fn)
        end
    end

    chain = build_chain(1)
    return chain(ctx)
end

"""
    apply_chat_middleware(middlewares, ctx, handler) -> Any

Execute the chat middleware chain.
"""
function apply_chat_middleware(middlewares::Vector, ctx::ChatContext, handler::Function)
    if isempty(middlewares)
        return handler(ctx)
    end

    function build_chain(idx::Int)
        if idx > length(middlewares)
            return handler
        end
        mw = middlewares[idx]
        return function(c::ChatContext)
            next_fn = build_chain(idx + 1)
            return mw(c, next_fn)
        end
    end

    chain = build_chain(1)
    return chain(ctx)
end

"""
    apply_function_middleware(middlewares, ctx, handler) -> Any

Execute the function invocation middleware chain.
"""
function apply_function_middleware(middlewares::Vector, ctx::FunctionInvocationContext, handler::Function)
    if isempty(middlewares)
        return handler(ctx)
    end

    function build_chain(idx::Int)
        if idx > length(middlewares)
            return handler
        end
        mw = middlewares[idx]
        return function(c::FunctionInvocationContext)
            next_fn = build_chain(idx + 1)
            return mw(c, next_fn)
        end
    end

    chain = build_chain(1)
    return chain(ctx)
end
