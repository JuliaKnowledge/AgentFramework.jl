# Multimodal Content

Utilities for constructing image, audio, and file [`Content`](@ref) items
that can be included in agent messages. MIME type detection is automatic
when not specified explicitly.

## MIME Type Detection

```@docs
AgentFramework.detect_mime_type
AgentFramework.is_image_mime
AgentFramework.is_audio_mime
```

## Content Constructors

```@docs
AgentFramework.image_content
AgentFramework.image_url_content
AgentFramework.audio_content
AgentFramework.file_content
```

## Encoding Utilities

```@docs
AgentFramework.base64_to_bytes
AgentFramework.content_to_openai_multimodal
```
