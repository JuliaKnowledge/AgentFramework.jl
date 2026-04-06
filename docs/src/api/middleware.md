# Middleware

The middleware pipeline intercepts requests and responses at three layers: agent,
chat client, and function invocation. Each layer follows the same pattern —
a chain of functions that can inspect, modify, or short-circuit the pipeline.
Use the [`@middleware`](@ref) macro for concise definitions.

## Abstract Types

```@docs
AgentFramework.AbstractMiddleware
```

## Context Types

```@docs
AgentFramework.AgentContext
AgentFramework.ChatContext
AgentFramework.FunctionInvocationContext
```

## Middleware Function Types

```@docs
AgentFramework.AgentMiddlewareFunc
AgentFramework.ChatMiddlewareFunc
AgentFramework.FunctionMiddlewareFunc
```

## Pipeline Execution

```@docs
AgentFramework.apply_agent_middleware
AgentFramework.apply_chat_middleware
AgentFramework.apply_function_middleware
```

## Termination

```@docs
AgentFramework.MiddlewareTermination
AgentFramework.terminate_pipeline
```

## Macros

```@docs
AgentFramework.@middleware
```
