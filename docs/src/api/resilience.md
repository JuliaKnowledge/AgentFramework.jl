# Resilience

Resilience utilities add automatic retry and rate-limiting behaviour to the
chat middleware pipeline. Wrap any [`AbstractChatClient`](@ref) with
[`with_retry!`](@ref) or [`with_rate_limit!`](@ref) for production-grade
fault tolerance.

## Retry

```@docs
AgentFramework.RetryConfig
AgentFramework.retry_chat_middleware
AgentFramework.with_retry!
```

## Rate Limiting

```@docs
AgentFramework.TokenBucketRateLimiter
AgentFramework.acquire!
AgentFramework.rate_limit_chat_middleware
AgentFramework.with_rate_limit!
```

## Agent Configuration Helpers

```@docs
AgentFramework.with_instructions
AgentFramework.with_tools
AgentFramework.with_name
AgentFramework.with_options
```
