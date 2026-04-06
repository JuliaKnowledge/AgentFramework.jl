# Structured Output

Structured output constrains LLM responses to a given JSON Schema, enabling
type-safe extraction of data from natural language. Use [`schema_from_type`](@ref)
to derive a schema from a Julia type and [`parse_structured`](@ref) to
deserialize the LLM output.

```@docs
AgentFramework.StructuredOutput
AgentFramework.schema_from_type
AgentFramework.response_format_for
AgentFramework.parse_structured
```
