# Telemetry and observability for AgentFramework.jl
# Provides structured logging, span tracking, and metrics collection
# with a pluggable backend abstraction (no OpenTelemetry dependency).

# ── Telemetry Span ───────────────────────────────────────────────────────────

"""
    TelemetrySpan

Represents a span/activity in a trace, following OpenTelemetry span semantics.

# Fields
- `id::String`: Unique span identifier.
- `parent_id::Union{Nothing, String}`: Parent span ID for nested traces.
- `name::String`: Span name (e.g., "agent.run", "chat.completion").
- `kind::Symbol`: `:internal`, `:client`, or `:server`.
- `start_time::DateTime`: UTC start time.
- `end_time::Union{Nothing, DateTime}`: UTC end time (set on finish).
- `status::Symbol`: `:unset`, `:ok`, or `:error`.
- `attributes::Dict{String, Any}`: Key-value span attributes.
- `events::Vector{Dict{String, Any}}`: Timestamped span events.
"""
Base.@kwdef mutable struct TelemetrySpan
    id::String = string(UUIDs.uuid4())
    parent_id::Union{Nothing, String} = nothing
    name::String
    kind::Symbol = :internal
    start_time::DateTime = Dates.now(Dates.UTC)
    end_time::Union{Nothing, DateTime} = nothing
    status::Symbol = :unset
    attributes::Dict{String, Any} = Dict{String, Any}()
    events::Vector{Dict{String, Any}} = Dict{String, Any}[]
end

"""
    finish_span!(span; status=:ok) -> TelemetrySpan

Mark a span as finished with the given status.
"""
function finish_span!(span::TelemetrySpan; status::Symbol = :ok)
    span.end_time = Dates.now(Dates.UTC)
    span.status = status
    return span
end

"""
    add_event!(span, name; attributes=Dict{String,Any}())

Append a timestamped event to the span.
"""
function add_event!(span::TelemetrySpan, name::String; attributes::Dict{String, Any} = Dict{String, Any}())
    push!(span.events, Dict{String, Any}("name" => name, "time" => Dates.now(Dates.UTC), "attributes" => attributes))
end

"""
    duration_ms(span) -> Union{Nothing, Int}

Return span duration in milliseconds, or `nothing` if not yet finished.
"""
duration_ms(span::TelemetrySpan) = span.end_time === nothing ? nothing : Dates.value(span.end_time - span.start_time)

# ── GenAI Semantic Conventions ───────────────────────────────────────────────

"""
    GenAIConventions

Constants following OpenTelemetry Semantic Conventions for Generative AI.
"""
module GenAIConventions
    const SYSTEM = "gen_ai.system"
    const OPERATION_NAME = "gen_ai.operation.name"
    const REQUEST_MODEL = "gen_ai.request.model"
    const RESPONSE_MODEL = "gen_ai.response.model"
    const REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    const REQUEST_TOP_P = "gen_ai.request.top_p"
    const REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    const RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    const USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    const USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    const USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    const AGENT_NAME = "gen_ai.agent.name"
    const AGENT_ID = "gen_ai.agent.id"
    const TOOL_NAME = "gen_ai.tool.name"
    const TOOL_CALL_ID = "gen_ai.tool.call_id"
end

# ── Telemetry Backends ───────────────────────────────────────────────────────

"""
    AbstractTelemetryBackend

Abstract telemetry backend. Implement `record_span!` to integrate with
OpenTelemetry, Datadog, or any other observability system.
"""
abstract type AbstractTelemetryBackend end

"""
    LoggingTelemetryBackend <: AbstractTelemetryBackend

Log-based backend using Julia's `Logging` module. Emits structured log
messages at the configured level for each completed span.
"""
struct LoggingTelemetryBackend <: AbstractTelemetryBackend
    level::Logging.LogLevel
end
LoggingTelemetryBackend() = LoggingTelemetryBackend(Logging.Info)

"""
    InMemoryTelemetryBackend <: AbstractTelemetryBackend

In-memory backend for testing — stores all recorded spans in a thread-safe vector.
"""
mutable struct InMemoryTelemetryBackend <: AbstractTelemetryBackend
    spans::Vector{TelemetrySpan}
    lock::ReentrantLock
end
InMemoryTelemetryBackend() = InMemoryTelemetryBackend(TelemetrySpan[], ReentrantLock())

"""
    record_span!(backend, span)

Record a finished span to the telemetry backend.
"""
function record_span!(backend::LoggingTelemetryBackend, span::TelemetrySpan)
    dur = duration_ms(span)
    attrs_str = join(["$k=$v" for (k, v) in span.attributes], ", ")
    @logmsg backend.level "$(span.name)" span_id=span.id duration_ms=dur status=span.status attributes=attrs_str
end

function record_span!(backend::InMemoryTelemetryBackend, span::TelemetrySpan)
    lock(backend.lock) do
        push!(backend.spans, span)
    end
end

"""
    get_spans(backend::InMemoryTelemetryBackend) -> Vector{TelemetrySpan}

Return a copy of all recorded spans.
"""
function get_spans(backend::InMemoryTelemetryBackend)
    lock(backend.lock) do
        return copy(backend.spans)
    end
end

"""
    clear_spans!(backend::InMemoryTelemetryBackend)

Remove all recorded spans.
"""
function clear_spans!(backend::InMemoryTelemetryBackend)
    lock(backend.lock) do
        empty!(backend.spans)
    end
end

# ── Telemetry Middleware ─────────────────────────────────────────────────────

"""
    telemetry_agent_middleware(backend) -> Function

Create an agent middleware that records agent invocation spans with GenAI
semantic convention attributes.
"""
function telemetry_agent_middleware(backend::AbstractTelemetryBackend)
    return (ctx::AgentContext, next::Function) -> begin
        span = TelemetrySpan(
            name = "agent.run",
            kind = :internal,
            attributes = Dict{String, Any}(
                GenAIConventions.AGENT_NAME => ctx.agent isa Agent ? ctx.agent.name : "unknown",
                GenAIConventions.OPERATION_NAME => "chat",
                "message_count" => length(ctx.messages),
            )
        )
        try
            result = next(ctx)
            if ctx.result isa AgentResponse
                resp = ctx.result
                span.attributes[GenAIConventions.RESPONSE_MODEL] = something(resp.model_id, "unknown")
                if resp.finish_reason !== nothing
                    span.attributes[GenAIConventions.RESPONSE_FINISH_REASONS] = string(resp.finish_reason)
                end
                if resp.usage_details !== nothing
                    ud = resp.usage_details
                    if ud.input_tokens !== nothing
                        span.attributes[GenAIConventions.USAGE_INPUT_TOKENS] = ud.input_tokens
                    end
                    if ud.output_tokens !== nothing
                        span.attributes[GenAIConventions.USAGE_OUTPUT_TOKENS] = ud.output_tokens
                    end
                end
            end
            finish_span!(span; status=:ok)
            record_span!(backend, span)
            return result
        catch e
            add_event!(span, "exception"; attributes=Dict{String, Any}("type" => string(typeof(e)), "message" => sprint(showerror, e)))
            finish_span!(span; status=:error)
            record_span!(backend, span)
            rethrow()
        end
    end
end

"""
    telemetry_chat_middleware(backend) -> Function

Create a chat middleware that records LLM call spans with request/response attributes.
"""
function telemetry_chat_middleware(backend::AbstractTelemetryBackend)
    return (ctx::ChatContext, next::Function) -> begin
        span = TelemetrySpan(
            name = "chat.completion",
            kind = :client,
            attributes = Dict{String, Any}(
                GenAIConventions.OPERATION_NAME => "chat",
                "message_count" => length(ctx.messages),
            )
        )
        if ctx.options.model !== nothing
            span.attributes[GenAIConventions.REQUEST_MODEL] = ctx.options.model
        end
        if ctx.options.temperature !== nothing
            span.attributes[GenAIConventions.REQUEST_TEMPERATURE] = ctx.options.temperature
        end
        if ctx.options.max_tokens !== nothing
            span.attributes[GenAIConventions.REQUEST_MAX_TOKENS] = ctx.options.max_tokens
        end
        tool_count = length(ctx.options.tools)
        if tool_count > 0
            span.attributes["gen_ai.request.tool_count"] = tool_count
        end
        try
            result = next(ctx)
            finish_span!(span; status=:ok)
            record_span!(backend, span)
            return result
        catch e
            add_event!(span, "exception"; attributes=Dict{String, Any}("type" => string(typeof(e)), "message" => sprint(showerror, e)))
            finish_span!(span; status=:error)
            record_span!(backend, span)
            rethrow()
        end
    end
end

"""
    telemetry_function_middleware(backend) -> Function

Create a function middleware that records tool invocation spans.
"""
function telemetry_function_middleware(backend::AbstractTelemetryBackend)
    return (ctx::FunctionInvocationContext, next::Function) -> begin
        span = TelemetrySpan(
            name = "tool.invoke",
            kind = :internal,
            attributes = Dict{String, Any}(
                GenAIConventions.TOOL_NAME => ctx.tool !== nothing ? ctx.tool.name : "unknown",
                GenAIConventions.TOOL_CALL_ID => ctx.call_id,
            )
        )
        try
            result = next(ctx)
            finish_span!(span; status=:ok)
            record_span!(backend, span)
            return result
        catch e
            add_event!(span, "exception"; attributes=Dict{String, Any}("type" => string(typeof(e)), "message" => sprint(showerror, e)))
            finish_span!(span; status=:error)
            record_span!(backend, span)
            rethrow()
        end
    end
end

# ── Convenience: Instrument an Agent ─────────────────────────────────────────

"""
    instrument!(agent::Agent, backend::AbstractTelemetryBackend) -> Agent

Add telemetry middleware to an agent for all three pipeline layers
(agent, chat, function). Returns the modified agent.
"""
function instrument!(agent::Agent, backend::AbstractTelemetryBackend)
    pushfirst!(agent.agent_middlewares, telemetry_agent_middleware(backend))
    pushfirst!(agent.chat_middlewares, telemetry_chat_middleware(backend))
    pushfirst!(agent.function_middlewares, telemetry_function_middleware(backend))
    return agent
end
