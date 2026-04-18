# Chat Client

The chat client abstraction provides a uniform interface to LLM providers.
[`AbstractChatClient`](@ref) defines the protocol; concrete implementations
(e.g., [`OpenAIChatClient`](@ref), [`AnthropicChatClient`](@ref)) handle
provider-specific details. All clients support both synchronous and streaming
response modes.

## Abstract Types

```@docs
AgentFramework.AbstractChatClient
```

## Options and Responses

```@docs
AgentFramework.ChatOptions
AgentFramework.ChatResponse
AgentFramework.ChatResponseUpdate
```

## Finish Reasons

```@docs
AgentFramework.FinishReason
```

## Streaming

```@docs
AgentFramework.ResponseStream
AgentFramework.StreamingToolAccumulator
```

## Request Functions

```@docs
AgentFramework.get_response
AgentFramework.get_response_streaming
```

## Utilities

```@docs
AgentFramework.merge_chat_options
AgentFramework.parse_finish_reason
```

## Streaming Tool Accumulation

```@docs
AgentFramework.accumulate_tool_call!
AgentFramework.get_accumulated_tool_calls
AgentFramework.reset_accumulator!
AgentFramework.has_tool_calls
```
