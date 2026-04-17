# Agent Skills system for AgentFramework.jl
# Discovers, loads, and provides skills from SKILL.md files and code-defined resources.
# Follows Microsoft's Agent Skills specification.

# ── Core Types ───────────────────────────────────────────────────────────────

"""
    SkillResource

A named supplementary resource within a skill. Can be static content or a callable function.

# Fields
- `name::String`: Resource identifier (must be unique within a skill).
- `description::String`: What this resource provides.
- `content::Union{Nothing, String}`: Static text content.
- `mime_type::String`: Content MIME type.
- `fn::Union{Nothing, Function}`: Callable that returns content lazily (takes no args or a Dict).
"""
Base.@kwdef struct SkillResource
    name::String
    description::String = ""
    content::Union{Nothing, String} = nothing
    mime_type::String = "text/plain"
    fn::Union{Nothing, Function} = nothing
end

"""
    get_resource_content(resource::SkillResource) -> String

Get the content of a resource, calling the function if needed.
"""
function get_resource_content(resource::SkillResource)::String
    if resource.content !== nothing
        return resource.content
    elseif resource.fn !== nothing
        return string(resource.fn())
    else
        return ""
    end
end

"""
    Skill

A discoverable, composable skill with metadata and resources.

# Fields
- `name::String`: Skill name (used for identification).
- `description::String`: What this skill does.
- `version::String`: Skill version.
- `instructions::String`: Detailed instructions for the LLM on how to use this skill.
- `resources::Dict{String, SkillResource}`: Named resources.
- `tags::Vector{String}`: Categorization tags.
- `source_path::Union{Nothing, String}`: Path to SKILL.md if file-based.
"""
Base.@kwdef struct Skill
    name::String
    description::String = ""
    version::String = "1.0.0"
    instructions::String = ""
    resources::Dict{String, SkillResource} = Dict{String, SkillResource}()
    tags::Vector{String} = String[]
    source_path::Union{Nothing, String} = nothing
end

# ── SKILL.md Parser ──────────────────────────────────────────────────────────

"""
    parse_skill_md(path::String) -> Skill

Parse a SKILL.md file into a Skill object.
"""
function parse_skill_md(path::String)::Skill
    content = read(path, String)
    return parse_skill_md_content(content; source_path=path)
end

"""
    parse_skill_md_content(content::String; source_path=nothing) -> Skill

Parse SKILL.md content string into a Skill object.

Supports YAML frontmatter between `---` markers:
```markdown
---
name: MySkill
description: Does something useful
version: 1.0.0
tags: [utility, text]
---

# Instructions

Detailed instructions...
```
"""
function parse_skill_md_content(content::String; source_path::Union{Nothing, String} = nothing)::Skill
    frontmatter_match = match(r"^\s*---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)"s, content)

    if frontmatter_match === nothing
        return Skill(
            name = source_path !== nothing ? basename(dirname(source_path)) : "unnamed",
            instructions = strip(content),
            source_path = source_path,
        )
    end

    frontmatter_str = frontmatter_match.captures[1]
    body = content[frontmatter_match.offset + length(frontmatter_match.match):end]

    metadata = _load_yaml_definition(frontmatter_str)

    Skill(
        name = string(get(metadata, "name", source_path !== nothing ? basename(dirname(source_path)) : "unnamed")),
        description = string(get(metadata, "description", "")),
        version = string(get(metadata, "version", "1.0.0")),
        instructions = strip(body),
        tags = _parse_tags(get(metadata, "tags", "")),
        source_path = source_path,
    )
end

"""Parse tags from YAML scalar or sequence values."""
function _parse_tags(tags_value)::Vector{String}
    tags_value === nothing && return String[]
    if tags_value isa AbstractVector
        return [string(tag) for tag in tags_value if !isempty(strip(string(tag)))]
    elseif !(tags_value isa AbstractString)
        return [string(tags_value)]
    end

    s = strip(tags_value, ['[', ']', ' '])
    isempty(s) && return String[]
    return [strip(t) for t in split(s, ",") if !isempty(strip(t))]
end

# ── Directory Scanner ────────────────────────────────────────────────────────

const SKILL_FILENAME = "SKILL.md"
const DEFAULT_SCAN_EXTENSIONS = [".md", ".txt", ".jl", ".py", ".json", ".yaml", ".yml"]

"""
    discover_skills(directory::String; recursive=true, max_depth=2) -> Vector{Skill}

Scan a directory for SKILL.md files and return discovered skills.

# Security
- Validates paths are within the given directory (no path traversal)
- Skips symlinks by default
- Limits recursion depth
"""
function discover_skills(
    directory::String;
    recursive::Bool = true,
    max_depth::Int = 2,
    follow_symlinks::Bool = false,
)::Vector{Skill}
    abs_dir = abspath(directory)
    !isdir(abs_dir) && throw(ArgumentError("Not a directory: $abs_dir"))

    skills = Skill[]
    _scan_directory!(skills, abs_dir, abs_dir, 0, max_depth, recursive, follow_symlinks)
    return skills
end

function _scan_directory!(
    skills::Vector{Skill},
    root_dir::String,
    current_dir::String,
    depth::Int,
    max_depth::Int,
    recursive::Bool,
    follow_symlinks::Bool,
)
    depth > max_depth && return

    for entry in readdir(current_dir; join=true)
        if islink(entry) && !follow_symlinks
            continue
        end

        if !_is_safe_path(entry, root_dir)
            @warn "Skipping path outside root: $entry"
            continue
        end

        if isfile(entry) && basename(entry) == SKILL_FILENAME
            try
                skill = parse_skill_md(entry)
                skill_dir = dirname(entry)
                resources = _discover_file_resources(skill_dir, root_dir, follow_symlinks)
                if !isempty(resources)
                    merged = merge(skill.resources, resources)
                    skill = Skill(
                        name = skill.name,
                        description = skill.description,
                        version = skill.version,
                        instructions = skill.instructions,
                        resources = merged,
                        tags = skill.tags,
                        source_path = skill.source_path,
                    )
                end
                push!(skills, skill)
            catch e
                @warn "Failed to parse skill at $entry" exception=e
            end
        elseif isdir(entry) && recursive
            _scan_directory!(skills, root_dir, entry, depth + 1, max_depth, recursive, follow_symlinks)
        end
    end
end

"""Discover file-based resources in a skill directory."""
function _discover_file_resources(skill_dir::String, root_dir::String, follow_symlinks::Bool)::Dict{String, SkillResource}
    resources = Dict{String, SkillResource}()
    for entry in readdir(skill_dir; join=true)
        if islink(entry) && !follow_symlinks
            continue
        end
        if isfile(entry) && basename(entry) != SKILL_FILENAME
            ext = lowercase(splitext(entry)[2])
            if ext in DEFAULT_SCAN_EXTENSIONS
                name = basename(entry)
                mime = ext == ".json" ? "application/json" : ext == ".jl" ? "text/x-julia" : "text/plain"
                path = entry  # capture for closure
                resources[name] = SkillResource(
                    name = name,
                    description = "File: $name",
                    fn = () -> read(path, String),
                    mime_type = mime,
                )
            end
        end
    end
    return resources
end

# ── Security Guards ──────────────────────────────────────────────────────────

"""Check that a path doesn't escape the root directory (prevents path traversal)."""
function _is_safe_path(path::String, root::String)::Bool
    try
        real = realpath(path)
        real_root = realpath(root)
        return startswith(real, real_root)
    catch
        return false
    end
end

# ── SkillsProvider ───────────────────────────────────────────────────────────

const DEFAULT_SKILLS_PROMPT = """You have access to the following skills. Use the load_skill tool to get detailed instructions before using a skill.

Available skills:
{skills}"""

"""
    SkillsProvider <: BaseContextProvider

Context provider that injects skill advertisements into the system prompt
and provides `load_skill` / `read_skill_resource` function tools.

# Fields
- `skills::Vector{Skill}`: Available skills.
- `system_prompt_template::String`: Template with `{skills}` placeholder.
"""
Base.@kwdef mutable struct SkillsProvider <: BaseContextProvider
    skills::Vector{Skill} = Skill[]
    system_prompt_template::String = DEFAULT_SKILLS_PROMPT
end

"""Add a skill to the provider."""
function add_skill!(provider::SkillsProvider, skill::Skill)
    push!(provider.skills, skill)
    return provider
end

"""Add skills discovered from a directory."""
function add_skills_from_directory!(provider::SkillsProvider, directory::String; kwargs...)
    skills = discover_skills(directory; kwargs...)
    append!(provider.skills, skills)
    return provider
end

"""Generate the skill advertisement string for the system prompt."""
function _format_skill_list(skills::Vector{Skill})::String
    lines = String[]
    for skill in skills
        tags_str = isempty(skill.tags) ? "" : " [$(join(skill.tags, ", "))]"
        push!(lines, "- **$(skill.name)** (v$(skill.version)): $(skill.description)$tags_str")
        for (rname, _) in skill.resources
            push!(lines, "  - Resource: $rname")
        end
    end
    return join(lines, "\n")
end

"""Implement context provider protocol — inject skills into system prompt and tools."""
function before_run!(provider::SkillsProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    isempty(provider.skills) && return

    skill_list = _format_skill_list(provider.skills)
    prompt = replace(provider.system_prompt_template, "{skills}" => skill_list)
    extend_instructions!(ctx, [prompt])

    extend_tools!(ctx, [
        _make_load_skill_tool(provider),
        _make_read_resource_tool(provider),
    ])
end

# ── Skill Tools ──────────────────────────────────────────────────────────────

"""Create the load_skill FunctionTool."""
function _make_load_skill_tool(provider::SkillsProvider)::FunctionTool
    FunctionTool(
        name = "load_skill",
        description = "Load detailed instructions for a skill by name. Call this before using any skill.",
        func = (args) -> begin
            name = args["skill_name"]
            skill = _find_skill(provider, name)
            if skill === nothing
                return "Error: Skill '$name' not found. Available: $(join([s.name for s in provider.skills], ", "))"
            end
            return _format_skill_instructions(skill)
        end,
        parameters = Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "skill_name" => Dict{String, Any}(
                    "type" => "string",
                    "description" => "Name of the skill to load",
                )
            ),
            "required" => ["skill_name"],
        ),
    )
end

"""Create the read_skill_resource FunctionTool."""
function _make_read_resource_tool(provider::SkillsProvider)::FunctionTool
    FunctionTool(
        name = "read_skill_resource",
        description = "Read a named resource from a skill.",
        func = (args) -> begin
            skill_name = args["skill_name"]
            resource_name = args["resource_name"]
            skill = _find_skill(provider, skill_name)
            if skill === nothing
                return "Error: Skill '$skill_name' not found."
            end
            resource = get(skill.resources, resource_name, nothing)
            if resource === nothing
                available = join(collect(keys(skill.resources)), ", ")
                return "Error: Resource '$resource_name' not found in skill '$skill_name'. Available: $available"
            end
            return get_resource_content(resource)
        end,
        parameters = Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "skill_name" => Dict{String, Any}(
                    "type" => "string",
                    "description" => "Name of the skill",
                ),
                "resource_name" => Dict{String, Any}(
                    "type" => "string",
                    "description" => "Name of the resource to read",
                ),
            ),
            "required" => ["skill_name", "resource_name"],
        ),
    )
end

function _find_skill(provider::SkillsProvider, name::String)::Union{Nothing, Skill}
    idx = findfirst(s -> lowercase(s.name) == lowercase(name), provider.skills)
    return idx === nothing ? nothing : provider.skills[idx]
end

function _format_skill_instructions(skill::Skill)::String
    parts = String["# $(skill.name) (v$(skill.version))", ""]
    !isempty(skill.description) && push!(parts, skill.description, "")
    !isempty(skill.instructions) && push!(parts, "## Instructions", "", skill.instructions, "")
    if !isempty(skill.resources)
        push!(parts, "## Available Resources")
        for (name, resource) in skill.resources
            push!(parts, "- **$name**: $(resource.description) ($(resource.mime_type))")
        end
    end
    return join(parts, "\n")
end

# ── Skill Source Decorators ──────────────────────────────────────────────────

"""
    AbstractSkillSource

Base type for composable skill sources. Implement `get_skills(source)::Vector{Skill}`.
"""
abstract type AbstractSkillSource end

"""Get skills from a source."""
function get_skills end

"""Simple skill source wrapping a vector of skills."""
struct StaticSkillSource <: AbstractSkillSource
    skills::Vector{Skill}
end
get_skills(s::StaticSkillSource)::Vector{Skill} = s.skills

"""Skill source from a directory of SKILL.md files."""
Base.@kwdef struct DirectorySkillSource <: AbstractSkillSource
    directory::String
    recursive::Bool = true
    follow_symlinks::Bool = false
end
get_skills(s::DirectorySkillSource)::Vector{Skill} =
    discover_skills(s.directory; recursive=s.recursive, follow_symlinks=s.follow_symlinks)

"""
    DeduplicatingSkillSource

Wraps a skill source and removes duplicate skills by name.
First occurrence wins.
"""
struct DeduplicatingSkillSource <: AbstractSkillSource
    inner::AbstractSkillSource
end

function get_skills(s::DeduplicatingSkillSource)::Vector{Skill}
    seen = Set{String}()
    result = Skill[]
    for skill in get_skills(s.inner)
        if skill.name ∉ seen
            push!(seen, skill.name)
            push!(result, skill)
        end
    end
    return result
end

"""
    FilteringSkillSource

Wraps a skill source and filters skills by a predicate function.

# Example
```julia
source = FilteringSkillSource(
    inner = DirectorySkillSource(directory="skills/"),
    filter = skill -> "production" in skill.tags
)
```
"""
struct FilteringSkillSource <: AbstractSkillSource
    inner::AbstractSkillSource
    filter::Function  # (Skill) -> Bool
end

get_skills(s::FilteringSkillSource)::Vector{Skill} =
    filter(s.filter, get_skills(s.inner))

"""
    AggregatingSkillSource

Combines multiple skill sources into a single source.
Skills from all sources are concatenated in order.
"""
struct AggregatingSkillSource <: AbstractSkillSource
    sources::Vector{AbstractSkillSource}
end

function get_skills(s::AggregatingSkillSource)::Vector{Skill}
    result = Skill[]
    for source in s.sources
        append!(result, get_skills(source))
    end
    return result
end

# ── Skill Source Builder ─────────────────────────────────────────────────────

"""
    SkillSourceBuilder

Fluent API for composing skill sources with decorators.

# Example
```julia
source = SkillSourceBuilder() |>
    b -> add_directory!(b, "skills/") |>
    b -> add_directory!(b, "extra_skills/") |>
    b -> filter_by!(b, skill -> "approved" in skill.tags) |>
    b -> deduplicate!(b) |>
    build
```
"""
mutable struct SkillSourceBuilder
    sources::Vector{AbstractSkillSource}
    decorators::Vector{Any}  # Callables that wrap AbstractSkillSource → AbstractSkillSource
end

SkillSourceBuilder() = SkillSourceBuilder(AbstractSkillSource[], Any[])

"""Add a directory of SKILL.md files to the builder."""
function add_directory!(b::SkillSourceBuilder, dir::String; recursive=true)
    push!(b.sources, DirectorySkillSource(directory=dir, recursive=recursive))
    return b
end

"""Add a static set of skills to the builder."""
function add_skills!(b::SkillSourceBuilder, skills::Vector{Skill})
    push!(b.sources, StaticSkillSource(skills))
    return b
end

"""Add a custom skill source to the builder."""
function add_source!(b::SkillSourceBuilder, source::AbstractSkillSource)
    push!(b.sources, source)
    return b
end

"""Apply deduplication (by name) to the builder's output."""
function deduplicate!(b::SkillSourceBuilder)
    push!(b.decorators, DeduplicatingSkillSource)
    return b
end

"""Apply a filter predicate to the builder's output."""
function filter_by!(b::SkillSourceBuilder, pred::Function)
    push!(b.decorators, inner -> FilteringSkillSource(inner, pred))
    return b
end

"""Build the composed skill source from the builder."""
function build(b::SkillSourceBuilder)::AbstractSkillSource
    base = if length(b.sources) == 1
        b.sources[1]
    else
        AggregatingSkillSource(copy(b.sources))
    end

    result = base
    for dec in b.decorators
        result = dec(result)
    end
    return result
end

"""Load skills from a composed skill source into a SkillsProvider."""
function load_skills!(provider::SkillsProvider, source::AbstractSkillSource)
    skills = get_skills(source)
    append!(provider.skills, skills)
    return provider
end

# ── Feature-stage registration ────────────────────────────────────────────────
# Upstream marks the skills subsystem as experimental (ExperimentalFeature.SKILLS).
# Register metadata so `feature_stage(Skill)` etc. return the correct tuple.
for T in (SkillResource, Skill, SkillsProvider)
    _register_feature_stage(T, :experimental, :SKILLS)
end
