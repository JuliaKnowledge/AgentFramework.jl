# Capabilities

Capabilities describe optional features that a [`AbstractChatClient`](@ref)
may support. The trait-like system uses singleton types (e.g., [`HasStreaming`](@ref))
and query functions (e.g., [`supports_streaming`](@ref)) so agent code can adapt
to provider features at runtime.

## Core Types

```@docs
AgentFramework.Capability
AgentFramework.NoCapability
```

## Capability Traits

```@docs
AgentFramework.HasEmbeddings
AgentFramework.HasImageGeneration
AgentFramework.HasCodeInterpreter
AgentFramework.HasFileSearch
AgentFramework.HasWebSearch
AgentFramework.HasStreaming
AgentFramework.HasStructuredOutput
AgentFramework.HasToolCalling
```

## Query Functions

```@docs
AgentFramework.has_capability
AgentFramework.supports_embeddings
AgentFramework.supports_image_generation
AgentFramework.supports_code_interpreter
AgentFramework.supports_file_search
AgentFramework.supports_web_search
AgentFramework.supports_streaming
AgentFramework.supports_structured_output
AgentFramework.supports_tool_calling
```

## Capability Introspection

```@docs
AgentFramework.list_capabilities
AgentFramework.require_capability
```

## Capability Actions

```@docs
AgentFramework.get_embeddings
AgentFramework.generate_image
```

## Capability Constructors

```@docs
AgentFramework.embedding_capability
AgentFramework.image_generation_capability
AgentFramework.code_interpreter_capability
AgentFramework.file_search_capability
AgentFramework.web_search_capability
AgentFramework.streaming_capability
AgentFramework.structured_output_capability
AgentFramework.tool_calling_capability
```
