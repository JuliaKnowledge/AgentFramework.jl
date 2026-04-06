# Tools

Tools let agents invoke Julia functions during a conversation. The [`@tool`](@ref)
macro inspects a function signature and generates JSON Schema metadata that LLMs
use to decide when and how to call it. At runtime, [`invoke_tool`](@ref) executes
the function and returns a result the LLM can consume.

## Abstract Types

```@docs
AgentFramework.AbstractTool
```

## FunctionTool

```@docs
AgentFramework.FunctionTool
```

## Tool Macro

```@docs
AgentFramework.@tool
```

## Schema and Invocation

```@docs
AgentFramework.tool_to_schema
AgentFramework.invoke_tool
```

## Tool Collections

```@docs
AgentFramework.normalize_tools
AgentFramework.find_tool
```

## Utilities

```@docs
AgentFramework.is_declaration_only
AgentFramework.parse_result
AgentFramework.reset_invocation_count!
```
