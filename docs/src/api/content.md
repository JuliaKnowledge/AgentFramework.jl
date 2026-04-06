# Content

The content system provides a polymorphic representation of message payloads.
Every [`Message`](@ref) carries a vector of [`Content`](@ref) items, each tagged
with a [`ContentType`](@ref) discriminator. Factory functions such as
[`text_content`](@ref) and [`function_call_content`](@ref) make it easy to
construct content items without manipulating the struct directly.

## Core Types

```@docs
AgentFramework.AbstractContent
AgentFramework.Content
AgentFramework.ContentType
```

## Content Type Constants

```@docs
AgentFramework.TEXT
AgentFramework.TEXT_REASONING
AgentFramework.ERROR_CONTENT
AgentFramework.URI
AgentFramework.DATA
AgentFramework.HOSTED_FILE
AgentFramework.FUNCTION_CALL
AgentFramework.FUNCTION_RESULT
AgentFramework.USAGE
AgentFramework.HOSTED_VECTOR_STORE
```

## Factory Functions

```@docs
AgentFramework.text_content
AgentFramework.reasoning_content
AgentFramework.data_content
AgentFramework.uri_content
AgentFramework.error_content
AgentFramework.function_call_content
AgentFramework.function_result_content
AgentFramework.function_approval_request_content
AgentFramework.function_approval_response_content
AgentFramework.usage_content
AgentFramework.hosted_file_content
AgentFramework.hosted_vector_store_content
```

## Approval Helpers

```@docs
AgentFramework.to_approval_response
AgentFramework.is_approval_request
AgentFramework.is_approval_response
```

## Query Functions

```@docs
AgentFramework.is_text
AgentFramework.is_reasoning
AgentFramework.is_function_call
AgentFramework.is_function_result
AgentFramework.get_text
AgentFramework.parse_arguments
```

## Serialization

```@docs
AgentFramework.content_to_dict
AgentFramework.content_from_dict
```

## Usage and Annotations

```@docs
AgentFramework.UsageDetails
AgentFramework.add_usage_details
AgentFramework.Annotation
```

## Media Type Utilities

```@docs
AgentFramework.content_type_string
AgentFramework.parse_content_type
AgentFramework.detect_media_type_from_base64
```
