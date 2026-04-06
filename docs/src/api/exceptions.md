# Exceptions

A structured exception hierarchy for fine-grained error handling. All
exceptions derive from [`AgentFrameworkError`](@ref). Catch specific
subtypes to handle agent errors, chat client errors, tool errors,
middleware errors, or workflow errors independently.

## Base Error

```@docs
AgentFramework.AgentFrameworkError
```

## Agent Errors

```@docs
AgentFramework.AgentError
AgentFramework.AgentInvalidAuthError
AgentFramework.AgentInvalidRequestError
AgentFramework.AgentInvalidResponseError
```

## Chat Client Errors

```@docs
AgentFramework.ChatClientError
AgentFramework.ChatClientInvalidAuthError
AgentFramework.ChatClientInvalidRequestError
```

## Tool Errors

```@docs
AgentFramework.ToolError
AgentFramework.ToolExecutionError
```

## Content Errors

```@docs
AgentFramework.ContentError
```

## Middleware Errors

```@docs
AgentFramework.MiddlewareError
```

## Workflow Errors

```@docs
AgentFramework.WorkflowError
```

## Declarative Errors

```@docs
AgentFramework.DeclarativeError
```

## User Input Errors

```@docs
AgentFramework.UserInputRequiredError
```
