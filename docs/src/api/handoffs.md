# Handoffs

Handoffs enable one agent to transfer control to another mid-conversation.
A [`HandoffTool`](@ref) wraps a target agent so the current agent can invoke
it like any other tool. The framework detects the [`HandoffResult`](@ref)
and routes subsequent messages to the new agent.

```@docs
AgentFramework.HandoffTool
AgentFramework.HandoffResult
AgentFramework.execute_handoff
AgentFramework.handoff_as_function_tool
AgentFramework.normalize_agent_tools
```
