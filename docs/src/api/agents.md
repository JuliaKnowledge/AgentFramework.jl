# Agents

The agent layer is the primary entry point for interacting with LLM-powered assistants.
[`AbstractAgent`](@ref) defines the base type, while [`Agent`](@ref) provides a
full-featured implementation with middleware, context providers, and telemetry.
Convenience aliases [`ChatCompletionAgent`](@ref) and [`AssistantAgent`](@ref) are
available for compatibility with the Python SDK naming conventions.

## Abstract Types

```@docs
AgentFramework.AbstractAgent
```

## Agent

```@docs
AgentFramework.Agent
AgentFramework.ChatCompletionAgent
AgentFramework.AssistantAgent
```

## Running Agents

```@docs
AgentFramework.run_agent
AgentFramework.run_agent_streaming
AgentFramework.create_session
```

## Agent as Tool

```@docs
AgentFramework.as_tool
```

## Response Types

```@docs
AgentFramework.AgentResponse
AgentFramework.AgentResponseUpdate
AgentFramework.get_final_response
```
