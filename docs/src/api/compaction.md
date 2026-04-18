# Message Compaction

When conversation history exceeds a token budget, compaction strategies
reduce message count while preserving essential context. The
[`CompactionPipeline`](@ref) applies one or more [`CompactionStrategy`](@ref)
values in sequence.

## Strategies

```@docs
AgentFramework.CompactionStrategy
```

## Configuration

```@docs
AgentFramework.CompactionConfig
AgentFramework.CompactionPipeline
```

## Operations

```@docs
AgentFramework.compact_messages
AgentFramework.needs_compaction
```

## Token Estimation

```@docs
AgentFramework.estimate_tokens
AgentFramework.estimate_message_tokens
AgentFramework.estimate_messages_tokens
```
