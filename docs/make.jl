using Documenter
using AgentFramework

include("AgentFrameworkDocExamples.jl")
using .AgentFrameworkDocExamples

const DOCS_ROOT = @__DIR__
const RAW_SRC = joinpath(DOCS_ROOT, "src")
const GENERATED_SRC = joinpath(DOCS_ROOT, "_generated")

function _block_name(relpath::AbstractString)
    stem = first(splitext(relpath))
    return replace(stem, r"[^A-Za-z0-9]+" => "_")
end

function _manual_page(relpath::AbstractString)
    relpath == "index.md" && return true
    startswith(relpath, "guide/") && return true
    startswith(relpath, "submodules/") && return true
    return false
end

function _known_docs_modules()
    modules = Module[AgentFramework]
    isdefined(AgentFramework, :A2A) && push!(modules, AgentFramework.A2A)
    isdefined(AgentFramework, :Bedrock) && push!(modules, AgentFramework.Bedrock)
    isdefined(AgentFramework, :Hosting) && push!(modules, AgentFramework.Hosting)
    isdefined(AgentFramework, :Mem0Integration) && push!(modules, AgentFramework.Mem0Integration)
    isdefined(AgentFramework, :CodingAgents) && push!(modules, AgentFramework.CodingAgents)
    isdefined(AgentFramework, :Neo4jIntegration) && push!(modules, AgentFramework.Neo4jIntegration)
    return modules
end

function _defined_in_docs_modules(type_name::AbstractString)
    symbol = Symbol(type_name)
    return any(isdefined(mod, symbol) for mod in _known_docs_modules())
end

function _should_eval_block(content::AbstractString, relpath::AbstractString = "")
    occursin("Pkg.add(", content) && return false
    occursin("using AzureIdentity", content) && return false
    occursin("DefaultAzureCredential(", content) && return false
    # Neo4j examples need a live Neo4j server; skip evaluation.
    relpath == "submodules/neo4j.md" && return false
    occursin(r"^\s*\w+\(.*\)\s*->"m, content) && return false

    struct_match = match(
        r"^\s*(?:Base\.@kwdef\s+)?(?:mutable\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)",
        content,
    )
    if struct_match !== nothing && _defined_in_docs_modules(struct_match.captures[1])
        return false
    end

    abstract_match = match(r"^\s*abstract\s+type\s+([A-Za-z_][A-Za-z0-9_]*)", content)
    if abstract_match !== nothing && _defined_in_docs_modules(abstract_match.captures[1])
        return false
    end

    return true
end

function _page_prelude(block_name::AbstractString)
    return """
```@meta
CurrentModule = AgentFramework
ShareDefaultModule = true
```

```@setup $block_name
Main.AgentFrameworkDocExamples.install!()
Main.AgentFrameworkDocExamples.setup_page!(@__MODULE__)
```

"""
end

function _rewrite_manual_page(text::String, relpath::AbstractString)
    lines = split(text, '\n'; keepempty = true)
    output = String[]
    block_name = _block_name(relpath)
    i = 1
    while i <= length(lines)
        line = lines[i]
        if startswith(line, "```julia")
            block = String[]
            j = i + 1
            while j <= length(lines) && strip(lines[j]) != "```"
                push!(block, lines[j])
                j += 1
            end
            content = join(block, "\n")
            push!(output, _should_eval_block(content, relpath) ? "```@example $block_name" : line)
            append!(output, block)
            if j <= length(lines)
                push!(output, lines[j])
            end
            i = j + 1
        else
            push!(output, line)
            i += 1
        end
    end
    return _page_prelude(block_name) * join(output, "\n")
end

function preprocess_docs!(raw_src::AbstractString, generated_src::AbstractString)
    rm(generated_src; recursive = true, force = true)
    for (root, _, files) in walkdir(raw_src)
        rel_root = relpath(root, raw_src)
        target_root = rel_root == "." ? generated_src : joinpath(generated_src, rel_root)
        mkpath(target_root)
        for file in files
            source = joinpath(root, file)
            target = joinpath(target_root, file)
            if endswith(file, ".md")
                relpath_to_source = relpath(source, raw_src)
                text = read(source, String)
                if _manual_page(relpath_to_source)
                    write(target, _rewrite_manual_page(text, relpath_to_source))
                else
                    write(target, text)
                end
            else
                cp(source, target; force = true)
            end
        end
    end
    return generated_src
end

function build_docs(; deploy::Bool = get(ENV, "CI", nothing) == "true")
    AgentFrameworkDocExamples.install!()
    preprocess_docs!(RAW_SRC, GENERATED_SRC)

    makedocs(
        sitename = "AgentFramework.jl",
        modules = [AgentFramework],
        source = GENERATED_SRC,
        build = joinpath(DOCS_ROOT, "build"),
        remotes = nothing,
        format = Documenter.HTML(
            prettyurls = deploy,
            canonical = "https://sdwfrost.github.io/AgentFramework.jl",
            assets = String[],
            size_threshold = 500_000,  # some API pages are large
        ),
        pages = [
            "Home" => "index.md",
            "Guide" => [
                "Getting Started" => "guide/getting_started.md",
                "Agents" => "guide/agents.md",
                "Tools" => "guide/tools.md",
                "Middleware" => "guide/middleware.md",
                "Sessions & Memory" => "guide/sessions.md",
                "Workflows" => "guide/workflows.md",
                "Providers" => "guide/providers.md",
                "Streaming" => "guide/streaming.md",
                "Advanced Topics" => "guide/advanced.md",
            ],
            "API Reference" => [
                "Agents" => "api/agents.md",
                "Content" => "api/content.md",
                "Messages" => "api/messages.md",
                "Tools" => "api/tools.md",
                "Sessions & Context" => "api/sessions.md",
                "Chat Client" => "api/chat_client.md",
                "Middleware" => "api/middleware.md",
                "Capabilities" => "api/capabilities.md",
                "Resilience" => "api/resilience.md",
                "Structured Output" => "api/structured_output.md",
                "Workflows" => "api/workflows.md",
                "Declarative" => "api/declarative.md",
                "Multimodal" => "api/multimodal.md",
                "Compaction" => "api/compaction.md",
                "MCP" => "api/mcp.md",
                "Evaluation" => "api/evaluation.md",
                "Skills" => "api/skills.md",
                "Handoffs" => "api/handoffs.md",
                "Telemetry" => "api/telemetry.md",
                "Serialization" => "api/serialization.md",
                "Exceptions" => "api/exceptions.md",
            ],
            "Submodules" => [
                "A2A Protocol" => "submodules/a2a.md",
                "Hosting" => "submodules/hosting.md",
                "Mem0 Integration" => "submodules/mem0.md",
                "Neo4j Integration" => "submodules/neo4j.md",
                "Bedrock" => "submodules/bedrock.md",
                "Coding Agents" => "submodules/coding_agents.md",
            ],
            "Vignettes" => [
                "Copilot × AgentFramework" => "vignettes/copilot_with_agentframework.md",
                "Memory Backends Compared" => "vignettes/memory_backends_compared.md",
                "Structured Output RAG" => "vignettes/structured_output_rag.md",
                "LLMWiki as a Tool" => "vignettes/llmwiki_as_tool.md",
            ],
        ],
        warnonly = [:missing_docs, :cross_references, :docs_block],
    ),

    if deploy
        deploydocs(
            repo = "github.com/sdwfrost/AgentFramework.jl.git",
            push_preview = true,
        )
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    build_docs()
end
