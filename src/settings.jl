# Settings management for AgentFramework.jl

# ── SecretString ─────────────────────────────────────────────────────────────

"""
    SecretString

String type that redacts its value in show/print to prevent accidental logging.
Access the raw value via `s.value`.
"""
struct SecretString
    value::String
end

Base.show(io::IO, ::SecretString) = print(io, "SecretString(\"***\")")
Base.string(s::SecretString) = s.value
Base.convert(::Type{String}, s::SecretString) = s.value
Base.length(s::SecretString) = length(s.value)
Base.isempty(s::SecretString) = isempty(s.value)

# ── Settings ─────────────────────────────────────────────────────────────────

"""
    Settings

Configuration container loaded from environment variables and .env files.
"""
Base.@kwdef mutable struct Settings
    values::Dict{String, String} = Dict{String, String}()
end

"""
    load_from_env!(settings::Settings; prefix="AGENTFRAMEWORK_")

Load settings from environment variables matching a prefix.
The prefix is stripped from the key name.
"""
function load_from_env!(settings::Settings; prefix::String = "AGENTFRAMEWORK_")
    for (k, v) in ENV
        if startswith(uppercase(k), uppercase(prefix))
            key = k[length(prefix)+1:end]
            settings.values[key] = v
        end
    end
    return settings
end

"""
    load_from_dotenv!(settings::Settings, path::String)

Load settings from a .env file (KEY=VALUE format, supports # comments and quoted values).
Returns settings unchanged if the file does not exist.
"""
function load_from_dotenv!(settings::Settings, path::String)
    !isfile(path) && return settings
    for line in readlines(path)
        line = strip(line)
        isempty(line) && continue
        startswith(line, '#') && continue
        m = match(r"^([^=]+)=(.*)$", line)
        if m !== nothing
            key = strip(m.captures[1])
            value = strip(m.captures[2])
            # Strip surrounding quotes
            if length(value) >= 2 && ((value[1] == '"' && value[end] == '"') || (value[1] == '\'' && value[end] == '\''))
                value = value[2:end-1]
            end
            settings.values[key] = value
        end
    end
    return settings
end

"""
    load_from_toml!(settings::Settings, path::String)

Load settings from a TOML file (flat key = value format).
Section headers are skipped; only top-level and in-section key-value pairs are loaded.
Returns settings unchanged if the file does not exist.
"""
function load_from_toml!(settings::Settings, path::String)
    !isfile(path) && return settings
    content = read(path, String)
    parsed = _parse_simple_toml(content)
    merge!(settings.values, parsed)
    return settings
end

function _parse_simple_toml(content::String)::Dict{String, String}
    result = Dict{String, String}()
    for line in split(content, "\n")
        line = strip(line)
        isempty(line) && continue
        startswith(line, '#') && continue
        startswith(line, '[') && continue  # skip section headers
        m = match(r"^([^=]+)=\s*(.*)$", line)
        if m !== nothing
            key = strip(m.captures[1])
            value = strip(m.captures[2])
            if length(value) >= 2 && value[1] == '"' && value[end] == '"'
                value = value[2:end-1]
            end
            result[key] = value
        end
    end
    return result
end

"""Get a setting value, with optional default."""
function get_setting(settings::Settings, key::String, default::Union{Nothing, String} = nothing)::Union{Nothing, String}
    get(settings.values, key, default)
end

"""Get a setting as a SecretString."""
function get_secret(settings::Settings, key::String)::Union{Nothing, SecretString}
    v = get(settings.values, key, nothing)
    v === nothing ? nothing : SecretString(v)
end

"""Check if a setting exists."""
has_setting(settings::Settings, key::String) = haskey(settings.values, key)
