# Telemetry

The telemetry system records spans and events for agent, chat client, and tool
invocations. It follows OpenTelemetry Semantic Conventions for Generative AI
via the [`GenAIConventions`](@ref) submodule. Use [`instrument!`](@ref) to
attach telemetry middleware to an agent in a single call.

## Span Management

```@docs
AgentFramework.TelemetrySpan
AgentFramework.finish_span!
AgentFramework.add_event!
AgentFramework.duration_ms
```

## Backends

```@docs
AgentFramework.AbstractTelemetryBackend
AgentFramework.LoggingTelemetryBackend
AgentFramework.InMemoryTelemetryBackend
AgentFramework.record_span!
AgentFramework.get_spans
AgentFramework.clear_spans!
```

## Middleware

```@docs
AgentFramework.telemetry_agent_middleware
AgentFramework.telemetry_chat_middleware
AgentFramework.telemetry_function_middleware
AgentFramework.instrument!
```

## GenAI Conventions

The `GenAIConventions` submodule provides attribute name constants that follow
the [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

```@docs
AgentFramework.GenAIConventions
```
