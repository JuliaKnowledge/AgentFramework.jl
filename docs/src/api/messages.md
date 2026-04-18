# Messages

Messages are the fundamental communication unit in the agent framework. Each
[`Message`](@ref) carries a [`Role`](@ref) (system, user, assistant, or tool)
and a vector of [`Content`](@ref) items. Helper functions handle normalization,
grouping, and serialization of message sequences.

## Core Types

```@docs
AgentFramework.Message
AgentFramework.Role
```

## Role Constants

```@docs

```

## Agent Run Inputs

```@docs
AgentFramework.AgentRunInputs
```

## Message Manipulation

```@docs
AgentFramework.normalize_messages
AgentFramework.prepend_instructions
```

## Serialization

```@docs
AgentFramework.message_to_dict
AgentFramework.message_from_dict
```

## Message Grouping

```@docs
AgentFramework.MessageGroup
AgentFramework.group_messages
AgentFramework.annotate_message_groups
```
