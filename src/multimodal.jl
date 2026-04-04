# Multimodal content helpers for AgentFramework.jl
# Support for image, audio, file content with MIME detection and base64 encoding.

# ── MIME Type Detection ──────────────────────────────────────────────────────

const _EXT_TO_MIME = Dict{String, String}(
    # Images
    ".png" => "image/png",
    ".jpg" => "image/jpeg",
    ".jpeg" => "image/jpeg",
    ".gif" => "image/gif",
    ".webp" => "image/webp",
    ".svg" => "image/svg+xml",
    ".bmp" => "image/bmp",
    ".ico" => "image/x-icon",
    ".tiff" => "image/tiff",
    ".tif" => "image/tiff",
    # Audio
    ".mp3" => "audio/mpeg",
    ".wav" => "audio/wav",
    ".ogg" => "audio/ogg",
    ".flac" => "audio/flac",
    ".aac" => "audio/aac",
    ".m4a" => "audio/mp4",
    ".wma" => "audio/x-ms-wma",
    # Video
    ".mp4" => "video/mp4",
    ".webm" => "video/webm",
    ".avi" => "video/x-msvideo",
    ".mov" => "video/quicktime",
    ".mkv" => "video/x-matroska",
    # Documents
    ".pdf" => "application/pdf",
    ".json" => "application/json",
    ".xml" => "application/xml",
    ".csv" => "text/csv",
    ".txt" => "text/plain",
    ".md" => "text/markdown",
    ".html" => "text/html",
    ".css" => "text/css",
    ".js" => "application/javascript",
    # Archives
    ".zip" => "application/zip",
    ".gz" => "application/gzip",
    ".tar" => "application/x-tar",
)

"""
    detect_mime_type(path::AbstractString) -> String

Detect MIME type from file extension. Returns "application/octet-stream" for unknown types.
"""
function detect_mime_type(path::AbstractString)::String
    ext = lowercase(splitext(path)[2])
    return get(_EXT_TO_MIME, ext, "application/octet-stream")
end

"""
    is_image_mime(mime::AbstractString) -> Bool

Check if a MIME type is an image type.
"""
is_image_mime(mime::AbstractString) = startswith(mime, "image/")

"""
    is_audio_mime(mime::AbstractString) -> Bool

Check if a MIME type is an audio type.
"""
is_audio_mime(mime::AbstractString) = startswith(mime, "audio/")

# ── Multimodal Content Constructors ──────────────────────────────────────────

"""
    image_content(data::Vector{UInt8}; media_type="image/png") -> Content
    image_content(path::AbstractString) -> Content

Create an image content item from raw bytes or a file path.
When given a path, auto-detects MIME type and reads/encodes the file.
"""
function image_content(data::Vector{UInt8}; media_type::String="image/png")::Content
    encoded = base64encode(data)
    Content(type=DATA, text=encoded, media_type=media_type,
        additional_properties=Dict{String, Any}("content_category" => "image"))
end

function image_content(path::AbstractString)::Content
    isfile(path) || throw(ContentError("File not found: $path"))
    mime = detect_mime_type(path)
    is_image_mime(mime) || @warn "File extension does not appear to be an image: $path (detected: $mime)"
    data = read(path)
    return image_content(data; media_type=mime)
end

"""
    image_url_content(url::AbstractString; media_type=nothing, detail=nothing) -> Content

Create an image content item from a URL (for OpenAI vision API pattern).
"""
function image_url_content(url::AbstractString;
    media_type::Union{Nothing, String}=nothing,
    detail::Union{Nothing, String}=nothing,
)::Content
    props = Dict{String, Any}("content_category" => "image")
    detail !== nothing && (props["detail"] = detail)
    Content(type=URI, uri=String(url), media_type=media_type,
        additional_properties=props)
end

"""
    audio_content(data::Vector{UInt8}; media_type="audio/wav") -> Content
    audio_content(path::AbstractString) -> Content

Create an audio content item from raw bytes or a file path.
"""
function audio_content(data::Vector{UInt8}; media_type::String="audio/wav")::Content
    encoded = base64encode(data)
    Content(type=DATA, text=encoded, media_type=media_type,
        additional_properties=Dict{String, Any}("content_category" => "audio"))
end

function audio_content(path::AbstractString)::Content
    isfile(path) || throw(ContentError("File not found: $path"))
    mime = detect_mime_type(path)
    is_audio_mime(mime) || @warn "File extension does not appear to be audio: $path (detected: $mime)"
    data = read(path)
    return audio_content(data; media_type=mime)
end

"""
    file_content(data::Vector{UInt8}; media_type="application/octet-stream", filename=nothing) -> Content
    file_content(path::AbstractString) -> Content

Create a file content item from raw bytes or a file path.
"""
function file_content(data::Vector{UInt8};
    media_type::String="application/octet-stream",
    filename::Union{Nothing, String}=nothing,
)::Content
    encoded = base64encode(data)
    props = Dict{String, Any}("content_category" => "file")
    filename !== nothing && (props["filename"] = filename)
    Content(type=DATA, text=encoded, media_type=media_type, additional_properties=props)
end

function file_content(path::AbstractString)::Content
    isfile(path) || throw(ContentError("File not found: $path"))
    mime = detect_mime_type(path)
    data = read(path)
    return file_content(data; media_type=mime, filename=basename(path))
end

"""
    base64_to_bytes(encoded::AbstractString) -> Vector{UInt8}

Decode a base64-encoded string to bytes.
"""
function base64_to_bytes(encoded::AbstractString)::Vector{UInt8}
    return base64decode(encoded)
end

"""
    content_to_openai_multimodal(content::Content) -> Dict{String, Any}

Convert a multimodal Content to the OpenAI vision/audio message format.
Returns the `content` part dict for an OpenAI message.
"""
function content_to_openai_multimodal(content::Content)::Dict{String, Any}
    if content.type == TEXT
        return Dict{String, Any}("type" => "text", "text" => something(content.text, ""))
    elseif content.type == URI && get(content.additional_properties, "content_category", "") == "image"
        img_url = Dict{String, Any}("url" => content.uri)
        detail = get(content.additional_properties, "detail", nothing)
        detail !== nothing && (img_url["detail"] = detail)
        return Dict{String, Any}("type" => "image_url", "image_url" => img_url)
    elseif content.type == DATA && content.media_type !== nothing
        category = get(content.additional_properties, "content_category", "")
        if category == "image" || is_image_mime(content.media_type)
            data_url = "data:$(content.media_type);base64,$(content.text)"
            return Dict{String, Any}(
                "type" => "image_url",
                "image_url" => Dict{String, Any}("url" => data_url),
            )
        elseif category == "audio" || is_audio_mime(content.media_type)
            return Dict{String, Any}(
                "type" => "input_audio",
                "input_audio" => Dict{String, Any}(
                    "data" => content.text,
                    "format" => _audio_format_from_mime(content.media_type),
                ),
            )
        end
    end
    # Fallback: text representation
    return Dict{String, Any}("type" => "text", "text" => something(content.text, string(content)))
end

function _audio_format_from_mime(mime::String)::String
    mime == "audio/wav" && return "wav"
    mime == "audio/mpeg" && return "mp3"
    mime == "audio/mp4" && return "mp4"
    mime == "audio/flac" && return "flac"
    mime == "audio/ogg" && return "ogg"
    return "wav"  # default
end
