# Declarative Configuration

The declarative layer lets you define agents and workflows in JSON, YAML, or
plain dictionaries instead of Julia code. A registry system maps string
identifiers to handler functions, tools, clients, and context providers so
that serialized definitions can be hydrated at runtime.

## Workflow Serialization

```@docs
AgentFramework.workflow_from_dict
AgentFramework.workflow_to_dict
AgentFramework.workflow_from_json
AgentFramework.workflow_to_json
AgentFramework.workflow_from_yaml
AgentFramework.workflow_to_yaml
AgentFramework.workflow_from_file
AgentFramework.workflow_to_file
```

## Agent Serialization

```@docs
AgentFramework.agent_from_dict
AgentFramework.agent_to_dict
AgentFramework.agent_from_json
AgentFramework.agent_to_json
AgentFramework.agent_from_yaml
AgentFramework.agent_to_yaml
AgentFramework.agent_from_file
AgentFramework.agent_to_file
```

## Handler Registry

```@docs
AgentFramework.register_handler!
AgentFramework.get_handler
AgentFramework.@register_handler
```

## Tool Registry

```@docs
AgentFramework.register_tool!
AgentFramework.get_tool
```

## Client Registry

```@docs
AgentFramework.register_client!
AgentFramework.get_client
```

## Context Provider Registry

```@docs
AgentFramework.register_context_provider!
AgentFramework.get_context_provider
```
