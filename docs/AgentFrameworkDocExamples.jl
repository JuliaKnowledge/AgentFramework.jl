module AgentFrameworkDocExamples

using Base64
using Dates
using AgentFramework

const HTTP = AgentFramework.HTTP
const JSON3 = AgentFramework.JSON3

const _INSTALLED = Ref(false)
const _DOCS_TMPDIR = Ref{String}("")

docs_tmpdir() = _DOCS_TMPDIR[]

function _ensure_docs_tmpdir!()
    if isempty(_DOCS_TMPDIR[])
        root = mktempdir()
        write(joinpath(root, "photo.jpg"), UInt8[0xFF, 0xD8, 0xFF, 0xD9])
        write(joinpath(root, "recording.wav"), UInt8[0x52, 0x49, 0x46, 0x46])
        write(joinpath(root, "knowledge.ttl"), "@prefix ex: <http://example.com/> .\nex:alice ex:knows ex:bob .\n")
        write(joinpath(root, "resource.txt"), "Docs resource")
        mkpath(joinpath(root, "skills", "summarizer"))
        write(
            joinpath(root, "skills", "summarizer", "SKILL.md"),
            "# Summarizer\n\n## Instructions\n\nUse this skill to summarize text.\n",
        )
        mkpath(joinpath(root, "agents"))
        mkpath(joinpath(root, "conversation_history"))
        mkpath(joinpath(root, "checkpoints"))
        mkpath(joinpath(root, "storage"))
        mkpath(joinpath(root, "runs"))
        mkpath(joinpath(root, "extra-dir"))
        write(joinpath(root, "mcp-config.json"), "{\"mcpServers\":{}}")
        write(joinpath(root, ".env"), "OPENAI_API_KEY=docs-openai-key\n")
        write(joinpath(root, "config.toml"), "MODEL = \"qwen3:8b\"\n")
        _DOCS_TMPDIR[] = root
    end
    return _DOCS_TMPDIR[]
end

function _last_user_text(messages::Vector{AgentFramework.Message})
    for message in Iterators.reverse(messages)
        if message.role == :user
            return AgentFramework.get_text(message)
        end
    end
    return "Hello from the docs"
end

function _schema_to_value(schema)
    schema isa AbstractDict || return "example"
    typ = get(schema, "type", "string")
    if typ == "object"
        props = get(schema, "properties", Dict{String, Any}())
        return Dict{String, Any}(String(key) => _schema_to_value(value) for (key, value) in pairs(props))
    elseif typ == "array"
        items = get(schema, "items", Dict{String, Any}("type" => "string"))
        return Any[_schema_to_value(items)]
    elseif typ == "integer"
        return 7
    elseif typ == "number"
        return 3.14
    elseif typ == "boolean"
        return true
    else
        return "example"
    end
end

function _response_text(messages::Vector{AgentFramework.Message}, options::AgentFramework.ChatOptions)
    response_format = options.response_format
    if response_format isa AbstractDict
        if get(response_format, "type", nothing) == "json_schema"
            schema = get(get(response_format, "json_schema", Dict{String, Any}()), "schema", Dict{String, Any}())
            return JSON3.write(_schema_to_value(schema))
        elseif get(response_format, "type", nothing) == "object"
            return JSON3.write(_schema_to_value(response_format))
        end
    end

    text = _last_user_text(messages)
    return "Docs example response for: " * text
end

function _chat_response(messages::Vector{AgentFramework.Message}, options::AgentFramework.ChatOptions)
    text = _response_text(messages, options)
    return AgentFramework.ChatResponse(
        messages = [AgentFramework.Message(:assistant, text)],
        finish_reason = AgentFramework.STOP,
        model_id = "docs-example-model",
    )
end

function _chat_stream(messages::Vector{AgentFramework.Message}, options::AgentFramework.ChatOptions)
    text = _response_text(messages, options)
    midpoint = max(1, fld(lastindex(text), 2))
    first_chunk = text[1:midpoint]
    second_chunk = midpoint < lastindex(text) ? text[(midpoint + 1):end] : ""

    return Channel{AgentFramework.ChatResponseUpdate}(4) do channel
        put!(
            channel,
            AgentFramework.ChatResponseUpdate(
                role = :assistant,
                contents = [AgentFramework.text_content(first_chunk)],
                model_id = "docs-example-model",
            ),
        )
        if !isempty(second_chunk)
            put!(
                channel,
                AgentFramework.ChatResponseUpdate(
                    contents = [AgentFramework.text_content(second_chunk)],
                    finish_reason = AgentFramework.STOP,
                    model_id = "docs-example-model",
                ),
            )
        else
            put!(
                channel,
                AgentFramework.ChatResponseUpdate(
                    finish_reason = AgentFramework.STOP,
                    model_id = "docs-example-model",
                ),
            )
        end
    end
end

function _agent_response(text::String; continuation_token=nothing, raw_representation=nothing)
    return AgentFramework.AgentResponse(
        messages = [AgentFramework.Message(:assistant, text)],
        finish_reason = AgentFramework.STOP,
        model_id = "docs-example-model",
        continuation_token = continuation_token,
        raw_representation = raw_representation,
    )
end

example_request() = HTTP.Request("GET", "/health")

function install!()
    _ensure_docs_tmpdir!()
    _INSTALLED[] && return nothing
    ENV["MEM0_API_KEY"] = "docs-mem0-key"

    provider_types = [
        AgentFramework.OllamaChatClient,
        AgentFramework.OpenAIChatClient,
        AgentFramework.AzureOpenAIChatClient,
        AgentFramework.AnthropicChatClient,
        AgentFramework.FoundryChatClient,
    ]

    if isdefined(AgentFramework, :Bedrock) && isdefined(AgentFramework.Bedrock, :BedrockChatClient)
        push!(provider_types, AgentFramework.Bedrock.BedrockChatClient)
    end

    if isdefined(AgentFramework, :CodingAgents)
        if isdefined(AgentFramework.CodingAgents, :GitHubCopilotChatClient)
            push!(provider_types, AgentFramework.CodingAgents.GitHubCopilotChatClient)
        end
        if isdefined(AgentFramework.CodingAgents, :ClaudeCodeChatClient)
            push!(provider_types, AgentFramework.CodingAgents.ClaudeCodeChatClient)
        end
    end

    for provider_type in provider_types
        @eval begin
            function AgentFramework.get_response(
                client::$provider_type,
                messages::Vector{AgentFramework.Message},
                options::AgentFramework.ChatOptions,
            )::AgentFramework.ChatResponse
                return AgentFrameworkDocExamples._chat_response(messages, options)
            end

            function AgentFramework.get_response_streaming(
                client::$provider_type,
                messages::Vector{AgentFramework.Message},
                options::AgentFramework.ChatOptions,
            )
                return AgentFrameworkDocExamples._chat_stream(messages, options)
            end
        end
    end

    if AgentFramework.get_tool("search_web") === nothing
        AgentFramework.register_tool!(
            "search_web",
            AgentFramework.FunctionTool(
                name = "search_web",
                description = "Search the web for documents examples.",
                func = (query::String) -> "Docs search results for: " * query,
                parameters = Dict{String, Any}(
                    "type" => "object",
                    "properties" => Dict{String, Any}(
                        "query" => Dict{String, Any}("type" => "string"),
                    ),
                    "required" => ["query"],
                ),
            ),
        )
    end

    if AgentFramework.get_tool("summarize") === nothing
        AgentFramework.register_tool!(
            "summarize",
            AgentFramework.FunctionTool(
                name = "summarize",
                description = "Summarize docs example text.",
                func = (text::String) -> "Summary: " * text,
                parameters = Dict{String, Any}(
                    "type" => "object",
                    "properties" => Dict{String, Any}(
                        "text" => Dict{String, Any}("type" => "string"),
                    ),
                    "required" => ["text"],
                ),
            ),
        )
    end

    if isdefined(AgentFramework, :StdioMCPClient) && isdefined(AgentFramework, :HTTPMCPClient)
        for mcp_type in (AgentFramework.StdioMCPClient, AgentFramework.HTTPMCPClient)
            @eval begin
                function AgentFramework.connect!(client::$mcp_type)
                    client._initialized = true
                    client._capabilities = AgentFramework.MCPServerCapabilities(
                        tools = true,
                        resources = true,
                        prompts = true,
                    )
                    return client
                end

                function AgentFramework.close_mcp!(client::$mcp_type)
                    client._initialized = false
                    return nothing
                end

                function AgentFramework.list_tools(client::$mcp_type)::Vector{AgentFramework.MCPToolInfo}
                    return [
                        AgentFramework.MCPToolInfo(
                            name = "read_file",
                            description = "Read a file from the docs workspace.",
                            input_schema = Dict{String, Any}(
                                "type" => "object",
                                "properties" => Dict{String, Any}(
                                    "path" => Dict{String, Any}("type" => "string"),
                                ),
                            ),
                        ),
                    ]
                end

                function AgentFramework.call_tool(
                    client::$mcp_type,
                    name::String,
                    arguments::Dict{String, Any} = Dict{String, Any}(),
                )::AgentFramework.MCPToolResult
                    path = get(arguments, "path", "resource.txt")
                    return AgentFramework.MCPToolResult(
                        content = [Dict{String, Any}(
                            "type" => "text",
                            "text" => "Docs MCP result for $(name): $(path)",
                        )],
                    )
                end

                function AgentFramework.list_resources(client::$mcp_type)::Vector{AgentFramework.MCPResource}
                    return [
                        AgentFramework.MCPResource(
                            uri = "file://" * joinpath(AgentFrameworkDocExamples.docs_tmpdir(), "resource.txt"),
                            name = "resource.txt",
                            description = "Docs resource",
                            mime_type = "text/plain",
                        ),
                    ]
                end

                function AgentFramework.read_resource(
                    client::$mcp_type,
                    uri::String,
                )::Vector{Dict{String, Any}}
                    return [Dict{String, Any}("uri" => uri, "text" => "Docs resource")]
                end

                function AgentFramework.list_prompts(client::$mcp_type)::Vector{AgentFramework.MCPPrompt}
                    return [
                        AgentFramework.MCPPrompt(
                            name = "my_prompt",
                            description = "Docs MCP prompt",
                        ),
                    ]
                end

                function AgentFramework.get_prompt(
                    client::$mcp_type,
                    name::String,
                    arguments::Dict{String, Any} = Dict{String, Any}(),
                )::Dict{String, Any}
                    return Dict{String, Any}(
                        "name" => name,
                        "arguments" => arguments,
                        "messages" => [Dict{String, Any}(
                            "role" => "user",
                            "content" => "Docs prompt content",
                        )],
                    )
                end
            end
        end
    end

    if isdefined(AgentFramework, :A2A)
        @eval begin
            function AgentFramework.A2A.get_agent_card(client::AgentFramework.A2A.A2AClient)
                return AgentFramework.A2A.A2AAgentCard(
                    name = "Docs Remote Agent",
                    description = "Stub A2A agent for evaluated docs examples",
                    url = client.base_url,
                )
            end

            function AgentFramework.A2A.send_message(
                client::AgentFramework.A2A.A2AClient,
                message::AgentFramework.Message;
                kwargs...,
            )::AgentFramework.AgentResponse
                continuation_token = get(kwargs, :background, false) ?
                    AgentFramework.A2A.A2AContinuationToken(task_id = "docs-task", context_id = "docs-context") :
                    nothing
                return AgentFrameworkDocExamples._agent_response(
                    "Docs A2A response for: " * AgentFramework.get_text(message);
                    continuation_token,
                )
            end

            function AgentFramework.A2A.get_task(
                client::AgentFramework.A2A.A2AClient,
                token::AgentFramework.A2A.A2AContinuationToken;
                kwargs...,
            )::AgentFramework.AgentResponse
                return AgentFrameworkDocExamples._agent_response("Docs task response")
            end

            function AgentFramework.A2A.get_task(
                client::AgentFramework.A2A.A2AClient,
                task_id::AbstractString;
                kwargs...,
            )::AgentFramework.AgentResponse
                return AgentFrameworkDocExamples._agent_response("Docs task response for " * String(task_id))
            end

            function AgentFramework.A2A.wait_for_completion(
                client::AgentFramework.A2A.A2AClient,
                token::AgentFramework.A2A.A2AContinuationToken;
                kwargs...,
            )::AgentFramework.AgentResponse
                return AgentFrameworkDocExamples._agent_response("Docs completed task response")
            end

            function AgentFramework.A2A.poll_task(
                agent::AgentFramework.A2A.A2ARemoteAgent,
                token::AgentFramework.A2A.A2AContinuationToken;
                session::Union{Nothing, AgentFramework.AgentSession} = nothing,
                background::Bool = true,
            )::AgentFramework.AgentResponse
                return AgentFrameworkDocExamples._agent_response("Docs poll task response")
            end
        end
    end

    if isdefined(AgentFramework, :Hosting)
        @eval begin
            function AgentFramework.Hosting.serve(runtime::AgentFramework.Hosting.HostedRuntime; kwargs...)
                return nothing
            end

            function AgentFramework.Hosting.start_workflow_run!(
                runtime::AgentFramework.Hosting.HostedRuntime,
                workflow_name::AbstractString,
                input;
                run_id::Union{Nothing, AbstractString}=nothing,
                metadata::AbstractDict=Dict{String, Any}(),
            )
                return AgentFramework.Hosting.HostedWorkflowRun(
                    id = run_id === nothing ? "docs-run-id" : String(run_id),
                    workflow_name = String(workflow_name),
                    internal_workflow_name = String(workflow_name),
                    metadata = Dict{String, Any}(string(k) => v for (k, v) in pairs(metadata)),
                )
            end

            function AgentFramework.Hosting.get_workflow_run(
                runtime::AgentFramework.Hosting.HostedRuntime,
                workflow_name::AbstractString,
                run_id::AbstractString,
            )
                return AgentFramework.Hosting.HostedWorkflowRun(
                    id = String(run_id),
                    workflow_name = String(workflow_name),
                    internal_workflow_name = String(workflow_name),
                )
            end

            function AgentFramework.Hosting.list_workflow_runs(
                runtime::AgentFramework.Hosting.HostedRuntime,
                workflow_name::Union{Nothing, AbstractString}=nothing,
            )
                workflow = workflow_name === nothing ? "docs-workflow" : String(workflow_name)
                return [
                    AgentFramework.Hosting.HostedWorkflowRun(
                        id = "docs-run-id",
                        workflow_name = workflow,
                        internal_workflow_name = workflow,
                    ),
                ]
            end

            function AgentFramework.Hosting.resume_workflow_run!(
                runtime::AgentFramework.Hosting.HostedRuntime,
                workflow_name::AbstractString,
                run_id::AbstractString,
                responses::AbstractDict,
            )
                return AgentFramework.Hosting.HostedWorkflowRun(
                    id = String(run_id),
                    workflow_name = String(workflow_name),
                    internal_workflow_name = String(workflow_name),
                    metadata = Dict{String, Any}(string(k) => v for (k, v) in pairs(responses)),
                )
            end

            function AgentFramework.Hosting.handle_request(
                runtime::AgentFramework.Hosting.HostedRuntime,
                request::HTTP.Request,
            )
                return HTTP.Response(200, JSON3.write(Dict("status" => "ok")))
            end
        end
    end

    if isdefined(AgentFramework, :Mem0Integration)
        @eval begin
            function AgentFramework.before_run!(
                provider::AgentFramework.Mem0Integration.Mem0ContextProvider,
                agent,
                session::AgentFramework.AgentSession,
                ctx::AgentFramework.SessionContext,
                state::Dict{String, Any},
            )
                return nothing
            end

            function AgentFramework.after_run!(
                provider::AgentFramework.Mem0Integration.Mem0ContextProvider,
                agent,
                session::AgentFramework.AgentSession,
                ctx::AgentFramework.SessionContext,
                state::Dict{String, Any},
            )
                return nothing
            end
        end
    end

    if isdefined(AgentFramework, :Bedrock) && isdefined(AgentFramework.Bedrock, :BedrockEmbeddingClient)
        @eval begin
            function AgentFramework.Bedrock.get_embeddings(
                client::AgentFramework.Bedrock.BedrockEmbeddingClient,
                texts::Vector{String};
                model::Union{Nothing, String} = nothing,
            )
                return [fill(Float64(index), 3) for index in eachindex(texts)]
            end
        end
    end

    _INSTALLED[] = true
    return nothing
end

function setup_page!(mod::Module)
    _ensure_docs_tmpdir!()
    page_name = String(nameof(mod))
    Core.eval(mod, :(using Main.AgentFramework))
    Core.eval(mod, :(using Dates))
    Core.eval(mod, :(using Base64))
    Core.eval(mod, :(cd(Main.AgentFrameworkDocExamples.docs_tmpdir())))
    Core.eval(mod, :(storage_dir = joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), "storage")))
    Core.eval(mod, :(runs_dir = joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), "runs")))
    Core.eval(mod, :(skills_dir = joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), "skills")))
    Core.eval(mod, :(photo_path = joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), "photo.jpg")))
    Core.eval(mod, :(recording_path = joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), "recording.wav")))
    Core.eval(mod, :(dotenv_path = joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), ".env")))
    Core.eval(mod, :(config_toml_path = joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), "config.toml")))
    Core.eval(mod, :(resource_uri = "file://" * joinpath(Main.AgentFrameworkDocExamples.docs_tmpdir(), "resource.txt")))
    Core.eval(mod, :(request = Main.AgentFrameworkDocExamples.example_request()))
    Core.eval(mod, :(input_data = Dict("input" => "example")))
    Core.eval(mod, :(input = "docs workflow input"))
    Core.eval(mod, :(session_id = "docs-session"))
    Core.eval(mod, :(run_id = "docs-run-id"))
    Core.eval(mod, :(client = Main.AgentFramework.OllamaChatClient(model = "docs-example-model", base_url = "http://localhost:11434")))
    Core.eval(mod, :(azure_client = client))
    Core.eval(mod, :(agent = Main.AgentFramework.Agent(name = "DocsAgent", instructions = "You are a docs example agent.", client = client)))
    Core.eval(mod, :(specialist_agent = Main.AgentFramework.Agent(name = "SpecialistAgent", instructions = "You are a specialist docs example agent.", client = client)))
    Core.eval(mod, :(session = Main.AgentFramework.AgentSession(id = "docs-session")))
    Core.eval(mod, :(messages = [Main.AgentFramework.Message(:user, "Hello"), Main.AgentFramework.Message(:assistant, "Hi there")]))
    Core.eval(mod, :(base64_data = Main.AgentFrameworkDocExamples.Base64.base64encode(UInt8[0x89, 0x50, 0x4E, 0x47])))
    Core.eval(mod, :(message = Main.AgentFramework.Message(:user, "Hello")))
    Core.eval(mod, :(msg = Main.AgentFramework.Message(:user, "Hello")))
    Core.eval(mod, :(token = isdefined(Main.AgentFramework, :A2A) ? Main.AgentFramework.A2A.A2AContinuationToken(task_id="task-123", context_id="ctx-123") : nothing))
    Core.eval(mod, :(params = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}())))
    Core.eval(mod, :(call_api(args...; kwargs...) = "docs-api-response"))
    Core.eval(mod, :(my_func(args...; kwargs...) = "docs-function-result"))
    Core.eval(mod, :(auth_middleware = (ctx, next) -> next(ctx)))
    Core.eval(mod, :(retry_middleware = (ctx, next) -> next(ctx)))
    Core.eval(mod, :(logging_middleware = (ctx, next) -> next(ctx)))
    Core.eval(mod, :(RESPONSE_CACHE = Dict{UInt64, Any}()))
    Core.eval(mod, :(msgs = [Main.AgentFramework.Message(:user, "Hello middleware")]))
    Core.eval(mod, :(mw1 = (ctx, next) -> next(ctx)))
    Core.eval(mod, :(mw2 = (ctx, next) -> next(ctx)))
    Core.eval(mod, :(convert_messages(messages) = [Dict("role" => String(msg.role), "content" => Main.AgentFramework.get_text(msg)) for msg in messages]))
    Core.eval(mod, :(stream_api_call(args...) = ((text = "chunk one",), (text = "chunk two",))))
    Core.eval(mod, :(get_my_token() = "docs-token"))
    Core.eval(mod, quote
        my_extra_tool = Main.AgentFramework.FunctionTool(
            name = "my_extra_tool",
            description = "Extra docs tool",
            func = () -> "extra",
            parameters = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}()),
        )
    end)
    Core.eval(mod, quote
        _docs_start = Main.AgentFramework.ExecutorSpec(
            id = "docs-start",
            description = "Docs workflow start",
            input_types = DataType[Any],
            output_types = DataType[Any],
            yield_types = DataType[],
            handler = (data, ctx) -> begin
                Main.AgentFramework.send_message(ctx, data)
                nothing
            end,
        )
        _docs_finish = Main.AgentFramework.ExecutorSpec(
            id = "docs-finish",
            description = "Docs workflow finish",
            input_types = DataType[Any],
            output_types = DataType[],
            yield_types = DataType[Any],
            handler = (data, ctx) -> begin
                Main.AgentFramework.yield_output(ctx, data)
                nothing
            end,
        )
        _docs_builder = Main.AgentFramework.WorkflowBuilder(
            name = "docs-workflow",
            start = _docs_start,
        )
        Main.AgentFramework.add_executor(_docs_builder, _docs_finish)
        Main.AgentFramework.add_edge(_docs_builder, _docs_start.id, _docs_finish.id)
        Main.AgentFramework.add_output(_docs_builder, _docs_finish.id)
        workflow = Main.AgentFramework.build(_docs_builder)
    end)
    if occursin("guide_workflows", page_name)
        Core.eval(mod, quote
            my_handler_function = (data, ctx) -> begin
                result = uppercase(string(data))
                Main.AgentFramework.send_message(ctx, result)
                Main.AgentFramework.yield_output(ctx, result)
                nothing
            end

            exec_a = Main.AgentFramework.ExecutorSpec(
                id = "a",
                description = "Docs executor A",
                input_types = DataType[String],
                output_types = DataType[String],
                yield_types = DataType[],
                handler = (data, ctx) -> begin
                    Main.AgentFramework.send_message(ctx, data)
                    nothing
                end,
            )
            exec_b = Main.AgentFramework.ExecutorSpec(
                id = "b",
                description = "Docs executor B",
                input_types = DataType[String],
                output_types = DataType[],
                yield_types = DataType[String],
                handler = (data, ctx) -> begin
                    Main.AgentFramework.yield_output(ctx, data)
                    nothing
                end,
            )

            router = Main.AgentFramework.ExecutorSpec(id = "router", handler = (data, ctx) -> nothing)
            planner_node = Main.AgentFramework.ExecutorSpec(id = "planner", handler = (data, ctx) -> nothing)
            handler_a = Main.AgentFramework.ExecutorSpec(id = "handler_a", handler = (data, ctx) -> nothing)
            writer_node = Main.AgentFramework.ExecutorSpec(id = "writer", handler = (data, ctx) -> nothing)
            coder_node = Main.AgentFramework.ExecutorSpec(id = "coder", handler = (data, ctx) -> nothing)
            reviewer_node = Main.AgentFramework.ExecutorSpec(id = "reviewer", handler = (data, ctx) -> nothing)
            assembler_node = Main.AgentFramework.ExecutorSpec(id = "assembler", handler = (data, ctx) -> nothing)
            classifier_node = Main.AgentFramework.ExecutorSpec(id = "classifier", handler = (data, ctx) -> nothing)
            celebrate_node = Main.AgentFramework.ExecutorSpec(id = "celebrate", handler = (data, ctx) -> nothing)
            console_node = Main.AgentFramework.ExecutorSpec(id = "console", handler = (data, ctx) -> nothing)
            fast_node = Main.AgentFramework.ExecutorSpec(id = "fast", handler = (data, ctx) -> nothing)
            medium_node = Main.AgentFramework.ExecutorSpec(id = "medium", handler = (data, ctx) -> nothing)
            slow_node = Main.AgentFramework.ExecutorSpec(id = "slow", handler = (data, ctx) -> nothing)

            b = Main.AgentFramework.WorkflowBuilder(name = "DocsEdges", start = router)
            for executor in (
                planner_node,
                handler_a,
                writer_node,
                coder_node,
                reviewer_node,
                assembler_node,
                classifier_node,
                celebrate_node,
                console_node,
                fast_node,
                medium_node,
                slow_node,
            )
                Main.AgentFramework.add_executor(b, executor)
            end

            step1 = Main.AgentFramework.Agent(name = "step1", instructions = "Step 1", client = client)
            step2 = Main.AgentFramework.Agent(name = "step2", instructions = "Step 2", client = client)
            step3 = Main.AgentFramework.Agent(name = "step3", instructions = "Step 3", client = client)
            researcher1 = Main.AgentFramework.Agent(name = "researcher1", instructions = "Research 1", client = client)
            researcher2 = Main.AgentFramework.Agent(name = "researcher2", instructions = "Research 2", client = client)
            researcher3 = Main.AgentFramework.Agent(name = "researcher3", instructions = "Research 3", client = client)
            agent_a = Main.AgentFramework.Agent(name = "agent-a", instructions = "Agent A", client = client)
            agent_b = Main.AgentFramework.Agent(name = "agent-b", instructions = "Agent B", client = client)
            agent_c = Main.AgentFramework.Agent(name = "agent-c", instructions = "Agent C", client = client)
            coder = Main.AgentFramework.Agent(name = "coder", instructions = "Code", client = client)
            reviewer = Main.AgentFramework.Agent(name = "reviewer", instructions = "Review", client = client)
            tester = Main.AgentFramework.Agent(name = "tester", instructions = "Test", client = client)

            merge_results(results) = join(string.(results), "\n")
        end)
    end
    return nothing
end

end # module
