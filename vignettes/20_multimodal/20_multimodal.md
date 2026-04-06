# Multimodal Input
Simon Frost

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Image Content from a File](#image-content-from-a-file)
- [Image Content from a URL](#image-content-from-a-url)
- [Sending Images to a Vision Agent](#sending-images-to-a-vision-agent)
- [Audio Content](#audio-content)
- [File Attachments](#file-attachments)
- [MIME Type Detection](#mime-type-detection)
- [Multiple Content Items in a
  Message](#multiple-content-items-in-a-message)
- [Base64 Utilities](#base64-utilities)
- [Converting Content for Provider
  APIs](#converting-content-for-provider-apis)
- [Summary](#summary)

## Overview

Modern LLMs can process more than text — many accept images, audio, and
other binary data alongside natural-language prompts. This vignette
shows how to:

1.  Load an image from disk and create multimodal `Content`.
2.  Reference an image by URL instead of embedding it.
3.  Send images to a vision-capable agent for description.
4.  Attach audio and arbitrary file content to messages.
5.  Use MIME type detection and base64 utilities.
6.  Compose messages with multiple content items.

## Prerequisites

You need [Ollama](https://ollama.com) running locally. For text-only
examples the `qwen3:8b` model is sufficient, but image examples require
a vision model such as `llava:7b`:

``` bash
ollama pull qwen3:8b
ollama pull llava:7b
```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..",".."))
using AgentFramework
using Base64
```

## Image Content from a File

The `image_content` helper reads a file from disk, base64-encodes its
bytes, and wraps them in a `Content` struct with `type = DATA` and the
appropriate MIME type:

``` julia
img = image_content("photo.png")

# The resulting Content has:
#   type       = DATA
#   uri        = "data:image/png;base64,iVBORw0KGgo..."
#   media_type = "image/png"
println(img.type)        # DATA
println(img.media_type)  # image/png
```

You can also construct image content from raw bytes when the data is
already in memory:

``` julia
raw_bytes = read("photo.png")
img = image_content(raw_bytes; media_type = "image/png")
```

## Image Content from a URL

When you have a publicly accessible image URL, use `image_url_content`
to create a `URI`-type content reference without downloading or encoding
the image locally:

``` julia
img_url = image_url_content(
    "https://example.com/photo.jpg";
    media_type = "image/jpeg",
)

println(img_url.type)        # URI
println(img_url.uri)         # https://example.com/photo.jpg
println(img_url.media_type)  # image/jpeg
```

## Sending Images to a Vision Agent

To describe an image, pair a text prompt with image content in a single
message and send it to a vision-capable model. Here we use `llava:7b`
via Ollama:

``` julia
client = OllamaChatClient(model = "llava:7b")

agent = Agent(
    name = "VisionAgent",
    instructions = "Describe images concisely.",
    client = client,
)

msg = Message(ROLE_USER, [
    text_content("What do you see in this image?"),
    image_content("photo.png"),
])

response = run_agent(agent, msg)
println(response.text)
```

**Expected output (varies by image):**

    The image shows a sunset over a mountain range with warm orange and purple tones.

## Audio Content

The `audio_content` helper works the same way as `image_content` — it
reads a file, base64-encodes it, and detects the MIME type from the
extension:

``` julia
aud = audio_content("recording.wav")

println(aud.type)        # DATA
println(aud.media_type)  # audio/wav
```

From raw bytes:

``` julia
raw_audio = read("recording.wav")
aud = audio_content(raw_audio; media_type = "audio/wav")
```

## File Attachments

For arbitrary documents — PDFs, spreadsheets, or any other binary format
— use `file_content`:

``` julia
doc = file_content("report.pdf")

println(doc.type)        # DATA
println(doc.media_type)  # application/pdf
```

If the extension is unrecognised, the MIME type defaults to
`application/octet-stream`:

``` julia
unknown = file_content("data.xyz")
println(unknown.media_type)  # application/octet-stream
```

## MIME Type Detection

AgentFramework provides helpers to detect and classify MIME types from
file extensions:

``` julia
# Detection from file path
println(detect_mime_type("image.png"))    # image/png
println(detect_mime_type("photo.jpg"))    # image/jpeg
println(detect_mime_type("audio.mp3"))    # audio/mpeg
println(detect_mime_type("clip.wav"))     # audio/wav
println(detect_mime_type("doc.pdf"))      # application/pdf

# Classification helpers
println(is_image_mime("image/png"))   # true
println(is_image_mime("audio/wav"))   # false
println(is_audio_mime("audio/wav"))   # true
println(is_audio_mime("image/png"))   # false
```

## Multiple Content Items in a Message

A single `Message` can contain any number of content items. This is
useful for tasks that involve comparing images or mixing media types:

``` julia
msg = Message(ROLE_USER, [
    text_content("Compare these two images:"),
    image_content("photo1.jpg"),
    image_content("photo2.jpg"),
])

response = run_agent(agent, msg)
println(response.text)
```

You can also mix text, images, and other media in one message:

``` julia
msg = Message(ROLE_USER, [
    text_content("Here is a photo and a recording. Describe both."),
    image_content("photo.png"),
    audio_content("recording.wav"),
])
```

## Base64 Utilities

AgentFramework relies on Julia’s `Base64` stdlib for encoding. A
convenience decoder is also provided for round-tripping:

``` julia
# Encode raw bytes to a base64 string
raw_bytes = read("photo.png")
encoded = base64encode(raw_bytes)
println(encoded[1:40], "...")

# Decode back to bytes
decoded = base64_to_bytes(encoded)
println(length(decoded), " bytes")
println(decoded == raw_bytes)  # true
```

## Converting Content for Provider APIs

When building custom providers, use `content_to_openai_multimodal` to
convert a `Content` struct into the dictionary format expected by
OpenAI-compatible APIs:

``` julia
img = image_content("photo.png")
openai_dict = content_to_openai_multimodal(img)
println(keys(openai_dict))  # ["type", "image_url"]
```

This is handled automatically by the built-in `OllamaChatClient` and
`OpenAIChatClient`, but is useful if you are implementing a custom chat
client.

## Summary

AgentFramework provides a consistent set of helpers for multimodal
input:

| Helper                     | Content Type | Use Case                       |
|----------------------------|--------------|--------------------------------|
| `text_content(text)`       | `TEXT`       | Plain text                     |
| `image_content(path)`      | `DATA`       | Image from disk (base64)       |
| `image_url_content(url)`   | `URI`        | Image by URL reference         |
| `audio_content(path)`      | `DATA`       | Audio from disk (base64)       |
| `file_content(path)`       | `DATA`       | Arbitrary file (base64)        |
| `detect_mime_type(path)`   | —            | Extension → MIME string        |
| `is_image_mime(mime)`      | —            | Check if MIME is an image type |
| `is_audio_mime(mime)`      | —            | Check if MIME is an audio type |
| `base64_to_bytes(encoded)` | —            | Decode base64 → raw bytes      |

Key points:

- **Vision models** like `llava:7b` can process images alongside text
  prompts.
- **Multiple content items** can be combined in a single `Message`.
- **MIME types** are detected automatically from file extensions.
- **Base64 encoding** is handled transparently by the content
  constructors.
- **`content_to_openai_multimodal`** converts content for
  OpenAI-compatible APIs.

Next, see [21 — Mem0 Integration](../21_mem0/21_mem0.qmd) to add
long-term memory to your agents.
