# Content type system for AgentFramework.jl
# Mirrors the Python Content class with its variant types.
#
# Python uses a single Content class with a `type` discriminator and many optional fields.
# Julia uses an enum for the type discriminator and a single struct, matching the Python
# design — this keeps serialization straightforward and avoids a deep type hierarchy.

"""
    ContentType

Enum of all content variant types, matching the Python `ContentType` literal.
"""
@enum ContentType begin
    TEXT
    TEXT_REASONING
    DATA
    URI
    ERROR_CONTENT
    FUNCTION_CALL
    FUNCTION_RESULT
    USAGE
    HOSTED_FILE
    HOSTED_VECTOR_STORE
    CODE_INTERPRETER_TOOL_CALL
    CODE_INTERPRETER_TOOL_RESULT
    IMAGE_GENERATION_TOOL_CALL
    IMAGE_GENERATION_TOOL_RESULT
    SHELL_TOOL_CALL
    SHELL_TOOL_RESULT
    SHELL_COMMAND_OUTPUT
    MCP_SERVER_TOOL_CALL
    MCP_SERVER_TOOL_RESULT
    FUNCTION_APPROVAL_REQUEST
    FUNCTION_APPROVAL_RESPONSE
    OAUTH_CONSENT_REQUEST
end

const CONTENT_TYPE_STRINGS = Dict{ContentType, String}(
    TEXT => "text",
    TEXT_REASONING => "text_reasoning",
    DATA => "data",
    URI => "uri",
    ERROR_CONTENT => "error",
    FUNCTION_CALL => "function_call",
    FUNCTION_RESULT => "function_result",
    USAGE => "usage",
    HOSTED_FILE => "hosted_file",
    HOSTED_VECTOR_STORE => "hosted_vector_store",
    CODE_INTERPRETER_TOOL_CALL => "code_interpreter_tool_call",
    CODE_INTERPRETER_TOOL_RESULT => "code_interpreter_tool_result",
    IMAGE_GENERATION_TOOL_CALL => "image_generation_tool_call",
    IMAGE_GENERATION_TOOL_RESULT => "image_generation_tool_result",
    SHELL_TOOL_CALL => "shell_tool_call",
    SHELL_TOOL_RESULT => "shell_tool_result",
    SHELL_COMMAND_OUTPUT => "shell_command_output",
    MCP_SERVER_TOOL_CALL => "mcp_server_tool_call",
    MCP_SERVER_TOOL_RESULT => "mcp_server_tool_result",
    FUNCTION_APPROVAL_REQUEST => "function_approval_request",
    FUNCTION_APPROVAL_RESPONSE => "function_approval_response",
    OAUTH_CONSENT_REQUEST => "oauth_consent_request",
)

const STRING_TO_CONTENT_TYPE = Dict{String, ContentType}(v => k for (k, v) in CONTENT_TYPE_STRINGS)

function content_type_string(ct::ContentType)::String
    CONTENT_TYPE_STRINGS[ct]
end

function parse_content_type(s::String)::ContentType
    get(STRING_TO_CONTENT_TYPE, s) do
        throw(ContentError("Unknown content type: $s"))
    end
end

"""
    UsageDetails

Token usage information from an LLM response.
"""
Base.@kwdef mutable struct UsageDetails
    input_tokens::Union{Nothing, Int} = nothing
    output_tokens::Union{Nothing, Int} = nothing
    total_tokens::Union{Nothing, Int} = nothing
    additional::Dict{String, Int} = Dict{String, Int}()
end

function add_usage_details(a::Union{Nothing, UsageDetails}, b::Union{Nothing, UsageDetails})::Union{Nothing, UsageDetails}
    a === nothing && return b
    b === nothing && return a
    merged = UsageDetails()
    for field in (:input_tokens, :output_tokens, :total_tokens)
        va = getfield(a, field)
        vb = getfield(b, field)
        if va !== nothing || vb !== nothing
            setfield!(merged, field, something(va, 0) + something(vb, 0))
        end
    end
    merged.additional = merge(+, a.additional, b.additional)
    return merged
end

"""
    Annotation

Metadata annotation on a content item (citations, file paths, etc.).
"""
const Annotation = Dict{String, Any}

"""
    Content

Unified content container covering all content variants.

Use the constructor functions (`text_content`, `data_content`, `function_call_content`, etc.)
rather than constructing directly.

# Core fields
- `type::ContentType`: The content variant.
- `text::Union{Nothing, String}`: Text payload (for TEXT, TEXT_REASONING).
- `uri::Union{Nothing, String}`: URI reference (for URI).
- `media_type::Union{Nothing, String}`: MIME type (for DATA, URI).
- `call_id::Union{Nothing, String}`: Tool call ID (for FUNCTION_CALL, FUNCTION_RESULT).
- `name::Union{Nothing, String}`: Function/tool name (for FUNCTION_CALL).
- `arguments::Union{Nothing, String}`: JSON-encoded arguments (for FUNCTION_CALL).
- `result::Any`: Tool result value (for FUNCTION_RESULT).
- `annotations::Union{Nothing, Vector{Annotation}}`: Content annotations.
- `additional_properties::Dict{String, Any}`: Extension properties.
"""
Base.@kwdef mutable struct Content <: AbstractContent
    type::ContentType

    # Text fields
    text::Union{Nothing, String} = nothing
    protected_data::Union{Nothing, String} = nothing

    # Data/URI fields
    uri::Union{Nothing, String} = nothing
    media_type::Union{Nothing, String} = nothing

    # Error fields
    message::Union{Nothing, String} = nothing
    error_code::Union{Nothing, String} = nothing
    error_details::Union{Nothing, String} = nothing

    # Usage fields
    usage_details::Union{Nothing, UsageDetails} = nothing

    # Function call/result fields
    call_id::Union{Nothing, String} = nothing
    name::Union{Nothing, String} = nothing
    arguments::Union{Nothing, String} = nothing
    exception::Union{Nothing, String} = nothing
    result::Any = nothing
    items::Union{Nothing, Vector{Content}} = nothing

    # Hosted resource fields
    file_id::Union{Nothing, String} = nothing
    vector_store_id::Union{Nothing, String} = nothing

    # Code interpreter fields
    inputs::Union{Nothing, Vector{Content}} = nothing
    outputs::Any = nothing

    # Image generation fields
    image_id::Union{Nothing, String} = nothing

    # Shell tool fields
    commands::Union{Nothing, Vector{String}} = nothing
    timeout_ms::Union{Nothing, Int} = nothing
    max_output_length::Union{Nothing, Int} = nothing
    status::Union{Nothing, String} = nothing
    stdout::Union{Nothing, String} = nothing
    stderr::Union{Nothing, String} = nothing
    exit_code::Union{Nothing, Int} = nothing
    timed_out::Union{Nothing, Bool} = nothing

    # MCP tool fields
    tool_name::Union{Nothing, String} = nothing
    server_name::Union{Nothing, String} = nothing
    output::Any = nothing

    # Function approval fields
    id::Union{Nothing, String} = nothing
    function_call::Union{Nothing, Content} = nothing
    user_input_request::Union{Nothing, Bool} = nothing
    approved::Union{Nothing, Bool} = nothing

    # OAuth consent fields
    consent_link::Union{Nothing, String} = nothing

    # Common metadata
    annotations::Union{Nothing, Vector{Annotation}} = nothing
    additional_properties::Dict{String, Any} = Dict{String, Any}()
    raw_representation::Any = nothing
end

# ── Constructor Functions ────────────────────────────────────────────────────

"""
    text_content(text; annotations=nothing, additional_properties=nothing) -> Content

Create a text content item.
"""
function text_content(text::AbstractString;
    annotations::Union{Nothing, Vector{Annotation}} = nothing,
    additional_properties::Dict{String, Any} = Dict{String, Any}(),
)::Content
    Content(type=TEXT, text=String(text), annotations=annotations, additional_properties=additional_properties)
end

"""
    reasoning_content(text; annotations=nothing, additional_properties=nothing) -> Content

Create a reasoning/thinking content item (chain-of-thought from reasoning models).
"""
function reasoning_content(text::AbstractString;
    annotations::Union{Nothing, Vector{Annotation}} = nothing,
    additional_properties::Dict{String, Any} = Dict{String, Any}(),
)::Content
    Content(type=TEXT_REASONING, text=String(text), annotations=annotations, additional_properties=additional_properties)
end

"""
    data_content(data, media_type; annotations=nothing) -> Content

Create a data content item with base64-encoded binary data.
"""
function data_content(data::AbstractString, media_type::Union{Nothing, AbstractString} = nothing;
    annotations::Union{Nothing, Vector{Annotation}} = nothing,
)::Content
    Content(type=DATA, text=String(data), media_type=media_type === nothing ? nothing : String(media_type), annotations=annotations)
end

"""
    uri_content(uri; media_type=nothing, annotations=nothing) -> Content

Create a URI-referencing content item.
"""
function uri_content(uri::AbstractString;
    media_type::Union{Nothing, AbstractString} = nothing,
    annotations::Union{Nothing, Vector{Annotation}} = nothing,
)::Content
    Content(type=URI, uri=String(uri), media_type=media_type === nothing ? nothing : String(media_type), annotations=annotations)
end

"""
    error_content(message; error_code=nothing, error_details=nothing) -> Content

Create an error content item.
"""
function error_content(message::AbstractString;
    error_code::Union{Nothing, AbstractString} = nothing,
    error_details::Union{Nothing, AbstractString} = nothing,
)::Content
    Content(type=ERROR_CONTENT, message=String(message),
        error_code=error_code === nothing ? nothing : String(error_code),
        error_details=error_details === nothing ? nothing : String(error_details))
end

"""
    function_call_content(call_id, name, arguments; annotations=nothing) -> Content

Create a function/tool call content item (emitted by the LLM).
"""
function function_call_content(call_id::AbstractString, name::AbstractString, arguments::AbstractString;
    annotations::Union{Nothing, Vector{Annotation}} = nothing,
)::Content
    Content(type=FUNCTION_CALL, call_id=String(call_id), name=String(name),
        arguments=String(arguments), annotations=annotations)
end

"""
    function_result_content(call_id, result; name=nothing, exception=nothing) -> Content

Create a function/tool result content item.
"""
function function_result_content(call_id::AbstractString, result;
    name::Union{Nothing, AbstractString} = nothing,
    exception::Union{Nothing, AbstractString} = nothing,
)::Content
    Content(type=FUNCTION_RESULT, call_id=String(call_id), result=result,
        name=name === nothing ? nothing : String(name),
        exception=exception === nothing ? nothing : String(exception))
end

"""
    usage_content(details) -> Content

Create a usage details content item.
"""
function usage_content(details::UsageDetails)::Content
    Content(type=USAGE, usage_details=details)
end

"""
    hosted_file_content(file_id) -> Content

Create a hosted file reference content item.
"""
function hosted_file_content(file_id::AbstractString)::Content
    Content(type=HOSTED_FILE, file_id=String(file_id))
end

"""
    hosted_vector_store_content(vector_store_id) -> Content

Create a hosted vector store reference content item.
"""
function hosted_vector_store_content(vector_store_id::AbstractString)::Content
    Content(type=HOSTED_VECTOR_STORE, vector_store_id=String(vector_store_id))
end

"""
    function_approval_request_content(id, function_call; annotations=nothing) -> Content

Create a function approval request, asking the user to approve a tool call.
"""
function function_approval_request_content(id::AbstractString, function_call::Content;
                                            annotations=nothing)::Content
    Content(
        type=FUNCTION_APPROVAL_REQUEST,
        id=String(id),
        function_call=function_call,
        user_input_request=true,
        annotations=annotations,
    )
end

"""
    function_approval_response_content(approved, id, function_call; annotations=nothing) -> Content

Create a function approval response indicating whether a tool call was approved.
"""
function function_approval_response_content(approved::Bool, id::AbstractString, function_call::Content;
                                             annotations=nothing)::Content
    Content(
        type=FUNCTION_APPROVAL_RESPONSE,
        approved=approved,
        id=String(id),
        function_call=function_call,
        annotations=annotations,
    )
end

"""
    to_approval_response(content, approved) -> Content

Convert a function_approval_request content to a function_approval_response.
"""
function to_approval_response(content::Content, approved::Bool)::Content
    content.type == FUNCTION_APPROVAL_REQUEST || error("Can only convert FUNCTION_APPROVAL_REQUEST to response")
    function_approval_response_content(approved, content.id, content.function_call;
                                        annotations=content.annotations)
end

"""Check if content is a function approval request."""
is_approval_request(c::Content) = c.type == FUNCTION_APPROVAL_REQUEST

"""Check if content is a function approval response."""
is_approval_response(c::Content) = c.type == FUNCTION_APPROVAL_RESPONSE

# ── Accessors & Utilities ────────────────────────────────────────────────────

"""
    get_text(content::Content) -> String

Extract text from a content item. Returns empty string if not a text type.
"""
function get_text(content::Content)::String
    (content.type == TEXT || content.type == TEXT_REASONING) && content.text !== nothing ? content.text : ""
end

"""
    is_text(content::Content) -> Bool

Check if content is a text variant.
"""
is_text(c::Content) = c.type == TEXT

"""
    is_reasoning(content::Content) -> Bool

Check if content is a reasoning/thinking variant.
"""
is_reasoning(c::Content) = c.type == TEXT_REASONING

"""
    is_function_call(content::Content) -> Bool

Check if content is a function call.
"""
is_function_call(c::Content) = c.type == FUNCTION_CALL

"""
    is_function_result(content::Content) -> Bool

Check if content is a function result.
"""
is_function_result(c::Content) = c.type == FUNCTION_RESULT

"""
    parse_arguments(content::Content) -> Union{Nothing, Dict{String, Any}}

Parse the JSON arguments of a function call content item.
"""
function parse_arguments(content::Content)::Union{Nothing, Dict{String, Any}}
    content.type != FUNCTION_CALL && return nothing
    content.arguments === nothing && return nothing
    isempty(content.arguments) && return Dict{String, Any}()
    try
        return JSON3.read(content.arguments, Dict{String, Any})
    catch
        return nothing
    end
end

# ── Serialization ────────────────────────────────────────────────────────────

"""
    content_to_dict(content::Content; exclude_none=true) -> Dict{String, Any}

Serialize a Content to a Dict, omitting nothing fields by default.
"""
function content_to_dict(content::Content; exclude_none::Bool = true)::Dict{String, Any}
    d = Dict{String, Any}("type" => content_type_string(content.type))
    for field in fieldnames(Content)
        field == :type && continue
        field == :raw_representation && continue
        field == :additional_properties && continue
        val = getfield(content, field)
        if exclude_none && val === nothing
            continue
        end
        if val isa Content
            d[String(field)] = content_to_dict(val; exclude_none)
        elseif val isa Vector{Content}
            d[String(field)] = [content_to_dict(c; exclude_none) for c in val]
        elseif val isa UsageDetails
            ud = Dict{String, Any}()
            val.input_tokens !== nothing && (ud["input_tokens"] = val.input_tokens)
            val.output_tokens !== nothing && (ud["output_tokens"] = val.output_tokens)
            val.total_tokens !== nothing && (ud["total_tokens"] = val.total_tokens)
            !isempty(val.additional) && merge!(ud, Dict{String, Any}(k => v for (k, v) in val.additional))
            d[String(field)] = ud
        else
            d[String(field)] = val
        end
    end
    # Merge additional_properties at top level
    if !isempty(content.additional_properties)
        merge!(d, content.additional_properties)
    end
    return d
end

"""
    content_from_dict(d::Dict{String, Any}) -> Content

Deserialize a Content from a Dict.
"""
function content_from_dict(d::Dict{String, Any})::Content
    ct = parse_content_type(d["type"]::String)
    kwargs = Dict{Symbol, Any}(:type => ct)

    for field in fieldnames(Content)
        field == :type && continue
        field == :raw_representation && continue
        field == :additional_properties && continue
        key = String(field)
        haskey(d, key) || continue
        val = d[key]
        val === nothing && continue

        if field == :usage_details && val isa Dict
            ud = UsageDetails()
            haskey(val, "input_tokens") && (ud.input_tokens = val["input_tokens"]::Int)
            haskey(val, "output_tokens") && (ud.output_tokens = val["output_tokens"]::Int)
            haskey(val, "total_tokens") && (ud.total_tokens = val["total_tokens"]::Int)
            kwargs[field] = ud
        elseif field == :function_call && val isa Dict
            kwargs[field] = content_from_dict(val)
        elseif field in (:items, :inputs) && val isa Vector
            kwargs[field] = Content[content_from_dict(v) for v in val if v isa Dict]
        elseif field == :annotations && val isa Vector
            kwargs[field] = Annotation[v for v in val if v isa Dict]
        else
            kwargs[field] = val
        end
    end

    Content(; kwargs...)
end

function Base.show(io::IO, c::Content)
    s = content_type_string(c.type)
    if c.type == TEXT
        print(io, "Content(text, \"", something(c.text, ""), "\")")
    elseif c.type == FUNCTION_CALL
        print(io, "Content(function_call, ", something(c.name, "?"), ")")
    elseif c.type == FUNCTION_RESULT
        print(io, "Content(function_result, call_id=", something(c.call_id, "?"), ")")
    else
        print(io, "Content(", s, ")")
    end
end

function Base.:(==)(a::Content, b::Content)
    a.type == b.type || return false
    for field in fieldnames(Content)
        field == :raw_representation && continue
        getfield(a, field) != getfield(b, field) && return false
    end
    return true
end

# ── Media Type Detection ────────────────────────────────────────────────────

# Magic byte prefixes in base64 for common file formats
const _MAGIC_BASE64_PREFIXES = [
    # Images
    ("iVBOR",        "image/png"),        # PNG: 0x89 0x50 0x4E 0x47
    ("/9j/",         "image/jpeg"),       # JPEG: 0xFF 0xD8 0xFF
    ("R0lGOD",       "image/gif"),        # GIF: GIF87a / GIF89a
    ("UklGR",        "image/webp"),       # WebP: RIFF....WEBP
    ("AAABAA",       "image/x-icon"),     # ICO
    ("Qk",           "image/bmp"),        # BMP: BM
    ("SUkq",         "image/tiff"),       # TIFF (little-endian): II
    ("TU0A",         "image/tiff"),       # TIFF (big-endian): MM
    # Audio
    ("T2dnU",        "audio/ogg"),        # OGG: OggS
    ("RIFF",         "audio/wav"),        # WAV: RIFF
    ("fLaC",         "audio/flac"),       # FLAC
    ("//uQx",        "audio/mpeg"),       # MP3
    ("SUQz",         "audio/mpeg"),       # MP3 (ID3)
    # Documents
    ("JVBERi0",      "application/pdf"),  # PDF: %PDF-
    ("UEsDBBQ",      "application/zip"),  # ZIP / DOCX / XLSX
    # Video
    ("AAAA",         "video/mp4"),        # MP4 (ftyp box)
    ("GkXfow",       "video/webm"),       # WebM
]

"""
    detect_media_type_from_base64(data::String) -> Union{String, Nothing}

Detect the media type of base64-encoded binary data by examining magic byte
prefixes. Returns a MIME type string or `nothing` if unrecognised.

# Examples
```julia
detect_media_type_from_base64("iVBORw0KGgoAAAANSU...")  # "image/png"
detect_media_type_from_base64("JVBERi0xLjQ...")          # "application/pdf"
detect_media_type_from_base64("dW5rbm93bg==")            # nothing
```
"""
function detect_media_type_from_base64(data::AbstractString)::Union{String, Nothing}
    isempty(data) && return nothing
    for (prefix, mime) in _MAGIC_BASE64_PREFIXES
        if startswith(data, prefix)
            return mime
        end
    end
    return nothing
end
