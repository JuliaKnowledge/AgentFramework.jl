# Content filtering / safety for AgentFramework.jl
# Exception types and filter result structures for content safety.

"""
    ContentFilterSeverity

Severity level for content filter results.
"""
@enum ContentFilterSeverity begin
    FILTER_SAFE       # Content is safe
    FILTER_LOW        # Low severity — may be acceptable
    FILTER_MEDIUM     # Medium severity — potentially problematic
    FILTER_HIGH       # High severity — should be blocked
end

"""
    ContentFilterCategory

Categories of content that may be filtered.
"""
@enum ContentFilterCategory begin
    FILTER_HATE           # Hate speech / discrimination
    FILTER_SELF_HARM      # Self-harm content
    FILTER_SEXUAL          # Sexual content
    FILTER_VIOLENCE        # Violence
    FILTER_PROFANITY       # Profanity
    FILTER_JAILBREAK       # Jailbreak / prompt injection attempt
    FILTER_PROTECTED_MATERIAL  # Copyrighted / protected content
    FILTER_CUSTOM          # Custom category
end

const FILTER_CATEGORY_STRINGS = Dict{ContentFilterCategory, String}(
    FILTER_HATE => "hate",
    FILTER_SELF_HARM => "self_harm",
    FILTER_SEXUAL => "sexual",
    FILTER_VIOLENCE => "violence",
    FILTER_PROFANITY => "profanity",
    FILTER_JAILBREAK => "jailbreak",
    FILTER_PROTECTED_MATERIAL => "protected_material",
    FILTER_CUSTOM => "custom",
)

"""
    ContentFilterResult

Result of a single content filter check.

# Fields
- `category::ContentFilterCategory`: What was flagged.
- `severity::ContentFilterSeverity`: How severe.
- `filtered::Bool`: Whether the content was actually blocked.
- `details::Union{Nothing, String}`: Optional detail string.
"""
Base.@kwdef struct ContentFilterResult
    category::ContentFilterCategory
    severity::ContentFilterSeverity
    filtered::Bool = false
    details::Union{Nothing, String} = nothing
end

"""
    ContentFilterResults

Aggregated content filter results from a single response.
"""
Base.@kwdef mutable struct ContentFilterResults
    results::Vector{ContentFilterResult} = ContentFilterResult[]
    blocked::Bool = false
    reason::Union{Nothing, String} = nothing
end

"""
    is_blocked(results::ContentFilterResults) -> Bool

Check if any filter blocked the content.
"""
is_blocked(r::ContentFilterResults) = r.blocked

"""
    get_filtered_categories(results::ContentFilterResults) -> Vector{ContentFilterCategory}

Get all categories that triggered filtering.
"""
function get_filtered_categories(r::ContentFilterResults)::Vector{ContentFilterCategory}
    return [fr.category for fr in r.results if fr.filtered]
end

"""
    max_severity(results::ContentFilterResults) -> ContentFilterSeverity

Get the highest severity across all filter results.
"""
function max_severity(r::ContentFilterResults)::ContentFilterSeverity
    isempty(r.results) && return FILTER_SAFE
    return maximum(fr.severity for fr in r.results)
end

"""
    ContentFilteredException <: Exception

Thrown when content is blocked by a safety filter.
"""
struct ContentFilteredException <: Exception
    message::String
    results::ContentFilterResults
end

Base.showerror(io::IO, e::ContentFilteredException) = print(io, "ContentFilteredException: ", e.message)

"""
    parse_openai_content_filter(data::Dict) -> ContentFilterResults

Parse OpenAI-format content filter annotations from a response.
"""
function parse_openai_content_filter(data::Dict)::ContentFilterResults
    results = ContentFilterResult[]
    blocked = false
    
    category_map = Dict(
        "hate" => FILTER_HATE,
        "self_harm" => FILTER_SELF_HARM,
        "sexual" => FILTER_SEXUAL,
        "violence" => FILTER_VIOLENCE,
        "jailbreak" => FILTER_JAILBREAK,
        "protected_material_text" => FILTER_PROTECTED_MATERIAL,
        "protected_material_code" => FILTER_PROTECTED_MATERIAL,
    )
    
    severity_map = Dict(
        "safe" => FILTER_SAFE,
        "low" => FILTER_LOW,
        "medium" => FILTER_MEDIUM,
        "high" => FILTER_HIGH,
    )
    
    for (key, cat) in category_map
        if haskey(data, key)
            info = data[key]
            if info isa Dict
                sev_str = get(info, "severity", "safe")
                sev = get(severity_map, string(sev_str), FILTER_SAFE)
                is_filtered = get(info, "filtered", false)
                is_filtered && (blocked = true)
                push!(results, ContentFilterResult(
                    category=cat, severity=sev, filtered=is_filtered))
            elseif info isa Bool && info
                blocked = true
                push!(results, ContentFilterResult(
                    category=cat, severity=FILTER_HIGH, filtered=true))
            end
        end
    end
    
    return ContentFilterResults(results=results, blocked=blocked)
end
