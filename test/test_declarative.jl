using AgentFramework
using Test

mutable struct DeclarativeMockChatClient <: AbstractChatClient
    responses::Vector{String}
    call_count::Int
end

DeclarativeMockChatClient(responses::Vector{String}) = DeclarativeMockChatClient(responses, 0)
DeclarativeMockChatClient(response::String) = DeclarativeMockChatClient([response])

function AgentFramework.get_response(client::DeclarativeMockChatClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    client.call_count += 1
    index = min(client.call_count, length(client.responses))
    return ChatResponse(
        messages = [Message(:assistant, client.responses[index])],
        finish_reason = STOP,
        model_id = "declarative-mock",
    )
end

function AgentFramework.get_response_streaming(client::DeclarativeMockChatClient, messages::Vector{Message}, options::ChatOptions)::Channel{ChatResponseUpdate}
    response = AgentFramework.get_response(client, messages, options)
    channel = Channel{ChatResponseUpdate}(1)
    Threads.@spawn begin
        for message in response.messages
            for content in message.contents
                put!(channel, ChatResponseUpdate(role = message.role, contents = [content]))
            end
        end
        put!(channel, ChatResponseUpdate(finish_reason = response.finish_reason))
        close(channel)
    end
    return channel
end

Base.@kwdef mutable struct DeclarativeTrackingProvider <: BaseContextProvider
    source_id::String
    events::Vector{String} = String[]
end

function AgentFramework.before_run!(provider::DeclarativeTrackingProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    push!(provider.events, "before:" * provider.source_id)
    state["count"] = get(state, "count", 0) + 1
end

function AgentFramework.after_run!(provider::DeclarativeTrackingProvider, agent, session::AgentSession, ctx::SessionContext, state::Dict{String, Any})
    push!(provider.events, "after:" * provider.source_id * ":" * string(get(state, "count", 0)))
end

declarative_echo_name(name::String) = name

declarative_echo_tool = FunctionTool(
    name = "decl_echo",
    description = "Echo a provided name.",
    func = declarative_echo_name,
    parameters = Dict{String, Any}(
        "type" => "object",
        "properties" => Dict{String, Any}(
            "name" => Dict{String, Any}("type" => "string"),
        ),
        "required" => ["name"],
    ),
)

@testset "Declarative Workflows" begin

    # ── 1. workflow_from_dict basic definition ────────────────────────────────
    @testset "workflow_from_dict basic definition" begin
        definition = Dict{String, Any}(
            "name" => "TestPipeline",
            "max_iterations" => 50,
            "start" => "step1",
            "outputs" => ["step2"],
            "executors" => [
                Dict{String, Any}("id" => "step1", "description" => "First step",
                     "input_types" => ["String"], "output_types" => ["String"]),
                Dict{String, Any}("id" => "step2", "description" => "Second step",
                     "input_types" => ["String"], "output_types" => ["String"]),
            ],
            "edges" => [
                Dict{String, Any}("kind" => "direct", "source" => "step1", "target" => "step2"),
            ]
        )

        wf = workflow_from_dict(definition; allow_missing_handlers=true)
        @test wf.name == "TestPipeline"
        @test wf.max_iterations == 50
        @test wf.start_executor_id == "step1"
        @test wf.output_executor_ids == ["step2"]
        @test length(wf.executors) == 2
        @test haskey(wf.executors, "step1")
        @test haskey(wf.executors, "step2")
        @test wf.executors["step1"].description == "First step"
        @test length(wf.edge_groups) == 1
        @test wf.edge_groups[1].kind == DIRECT_EDGE
    end

    # ── 2. workflow_from_dict with handler registry ───────────────────────────
    @testset "workflow_from_dict with handler registry" begin
        # Clean registry
        empty!(AgentFramework._HANDLER_REGISTRY)

        my_handler = (msg, ctx) -> send_message(ctx, uppercase(string(msg)))
        register_handler!("my_upper", my_handler)

        definition = Dict{String, Any}(
            "start" => "proc",
            "executors" => [
                Dict{String, Any}("id" => "proc", "handler" => "my_upper"),
            ],
        )
        wf = workflow_from_dict(definition)
        @test wf.executors["proc"].handler === my_handler

        empty!(AgentFramework._HANDLER_REGISTRY)
    end

    # ── 3. workflow_from_dict with local handlers ─────────────────────────────
    @testset "workflow_from_dict with local handlers" begin
        local_handler = (msg, ctx) -> send_message(ctx, reverse(string(msg)))

        definition = Dict{String, Any}(
            "start" => "rev",
            "executors" => [
                Dict{String, Any}("id" => "rev", "handler" => "reverser"),
            ],
        )
        wf = workflow_from_dict(definition; handlers=Dict{String, Function}("reverser" => local_handler))
        @test wf.executors["rev"].handler === local_handler
    end

    # ── 4. workflow_from_dict with explicit passthrough opt-in ─────────────────
    @testset "workflow_from_dict with explicit passthrough opt-in" begin
        definition = Dict{String, Any}(
            "start" => "pass",
            "executors" => [
                Dict{String, Any}("id" => "pass"),
            ],
        )
        wf = workflow_from_dict(definition; allow_missing_handlers=true)
        # Passthrough handler should exist
        @test wf.executors["pass"].handler isa Function
    end

    @testset "workflow_from_dict missing handler throws" begin
        definition = Dict{String, Any}(
            "start" => "pass",
            "executors" => [
                Dict{String, Any}("id" => "pass"),
            ],
        )
        @test_throws WorkflowError workflow_from_dict(definition)
    end

    # ── 5. workflow_from_dict with fan_out edge ───────────────────────────────
    @testset "workflow_from_dict with fan_out edge" begin
        definition = Dict{String, Any}(
            "start" => "src",
            "executors" => [
                Dict{String, Any}("id" => "src"),
                Dict{String, Any}("id" => "t1"),
                Dict{String, Any}("id" => "t2"),
            ],
            "edges" => [
                Dict{String, Any}("kind" => "fan_out", "source" => "src", "targets" => ["t1", "t2"]),
            ]
        )
        wf = workflow_from_dict(definition; allow_missing_handlers=true)
        @test length(wf.edge_groups) == 1
        @test wf.edge_groups[1].kind == FAN_OUT_EDGE
        @test Set(target_executor_ids(wf.edge_groups[1])) == Set(["t1", "t2"])
    end

    # ── 6. workflow_from_dict with fan_in edge ────────────────────────────────
    @testset "workflow_from_dict with fan_in edge" begin
        definition = Dict{String, Any}(
            "start" => "a",
            "executors" => [
                Dict{String, Any}("id" => "a"),
                Dict{String, Any}("id" => "b"),
                Dict{String, Any}("id" => "c"),
            ],
            "edges" => [
                Dict{String, Any}("kind" => "fan_in", "sources" => ["a", "b"], "target" => "c"),
            ]
        )
        wf = workflow_from_dict(definition; allow_missing_handlers=true)
        @test length(wf.edge_groups) == 1
        @test wf.edge_groups[1].kind == FAN_IN_EDGE
        @test Set(source_executor_ids(wf.edge_groups[1])) == Set(["a", "b"])
        @test target_executor_ids(wf.edge_groups[1]) == ["c"]
    end

    # ── 7. workflow_to_dict roundtrip preserves structure ─────────────────────
    @testset "workflow_to_dict roundtrip" begin
        definition = Dict{String, Any}(
            "name" => "RoundTrip",
            "max_iterations" => 42,
            "start" => "a",
            "outputs" => ["b"],
            "executors" => [
                Dict{String, Any}("id" => "a", "description" => "Exec A",
                     "input_types" => ["String"], "output_types" => ["String"]),
                Dict{String, Any}("id" => "b", "description" => "Exec B",
                     "input_types" => ["String"], "output_types" => ["Int"]),
            ],
            "edges" => [
                Dict{String, Any}("kind" => "direct", "source" => "a", "target" => "b"),
            ]
        )

        wf = workflow_from_dict(definition; allow_missing_handlers=true)
        d = workflow_to_dict(wf)

        @test d["name"] == "RoundTrip"
        @test d["max_iterations"] == 42
        @test d["start"] == "a"
        @test d["outputs"] == ["b"]
        @test length(d["executors"]) == 2
        @test length(d["edges"]) == 1

        # Check executor IDs are preserved
        exec_ids = Set(e["id"] for e in d["executors"])
        @test exec_ids == Set(["a", "b"])

        # Check edge is preserved
        edge = d["edges"][1]
        @test edge["kind"] == "direct"
        @test edge["source"] == "a"
        @test edge["target"] == "b"
    end

    # ── 8. workflow_to_json / workflow_from_json roundtrip ─────────────────────
    @testset "JSON roundtrip" begin
        handler = (msg, ctx) -> send_message(ctx, msg)

        definition = Dict{String, Any}(
            "name" => "JSONTest",
            "start" => "x",
            "outputs" => ["y"],
            "executors" => [
                Dict{String, Any}("id" => "x", "description" => "Node X",
                     "input_types" => ["String"], "output_types" => ["String"]),
                Dict{String, Any}("id" => "y", "description" => "Node Y",
                     "input_types" => ["String"], "output_types" => ["String"]),
            ],
            "edges" => [
                Dict{String, Any}("kind" => "direct", "source" => "x", "target" => "y"),
            ]
        )

        wf1 = workflow_from_dict(definition; handlers=Dict{String, Function}("x" => handler, "y" => handler))
        json_str = workflow_to_json(wf1)
        @test json_str isa String
        @test occursin("JSONTest", json_str)

        wf2 = workflow_from_json(json_str; handlers=Dict{String, Function}("x" => handler, "y" => handler))
        @test wf2.name == "JSONTest"
        @test wf2.start_executor_id == "x"
        @test wf2.output_executor_ids == ["y"]
        @test length(wf2.executors) == 2
        @test length(wf2.edge_groups) == 1
    end

    # ── 9. workflow_to_file / workflow_from_file roundtrip ────────────────────
    @testset "File roundtrip" begin
        handler = (msg, ctx) -> send_message(ctx, msg)

        definition = Dict{String, Any}(
            "name" => "FileTest",
            "start" => "a",
            "executors" => [
                Dict{String, Any}("id" => "a"),
            ],
        )

        wf1 = workflow_from_dict(definition; handlers=Dict{String, Function}("a" => handler))
        tmpfile = tempname() * ".json"
        try
            workflow_to_file(wf1, tmpfile)
            @test isfile(tmpfile)

            wf2 = workflow_from_file(tmpfile; handlers=Dict{String, Function}("a" => handler))
            @test wf2.name == "FileTest"
            @test wf2.start_executor_id == "a"
        finally
            isfile(tmpfile) && rm(tmpfile)
        end
    end

    # ── 10. register_handler! / get_handler work ─────────────────────────────
    @testset "register_handler! and get_handler" begin
        empty!(AgentFramework._HANDLER_REGISTRY)

        @test get_handler("nonexistent") === nothing

        h = (msg, ctx) -> nothing
        register_handler!("test_h", h)
        @test get_handler("test_h") === h

        # Overwrite
        h2 = (msg, ctx) -> send_message(ctx, msg)
        register_handler!("test_h", h2)
        @test get_handler("test_h") === h2

        empty!(AgentFramework._HANDLER_REGISTRY)
    end

    # ── 11. @register_handler macro ──────────────────────────────────────────
    @testset "@register_handler macro" begin
        empty!(AgentFramework._HANDLER_REGISTRY)

        my_func = (msg, ctx) -> yield_output(ctx, msg)
        @register_handler "macro_handler" my_func
        @test get_handler("macro_handler") === my_func

        empty!(AgentFramework._HANDLER_REGISTRY)
    end

    # ── 12. _parse_type_name maps correctly ──────────────────────────────────
    @testset "_parse_type_name" begin
        @test AgentFramework._parse_type_name("Any") == Any
        @test AgentFramework._parse_type_name("String") == String
        @test AgentFramework._parse_type_name("Int") == Int
        @test AgentFramework._parse_type_name("Float64") == Float64
        @test AgentFramework._parse_type_name("Bool") == Bool
        @test AgentFramework._parse_type_name("Dict") == Dict{String, Any}
        @test AgentFramework._parse_type_name("Dict{String, String}") == Dict{String, String}
        @test AgentFramework._parse_type_name("Vector") == Vector{Any}
        @test AgentFramework._parse_type_name("Vector{String}") == Vector{String}
        @test AgentFramework._parse_type_name("Vector{Message}") == Vector{Message}
        @test AgentFramework._parse_type_name("Message") == Message
        @test AgentFramework._parse_type_name("UnknownType") == Any
    end

    # ── 13. Execute loaded workflow with actual handlers ─────────────────────
    @testset "Execute loaded workflow" begin
        upper_handler = (msg, ctx) -> send_message(ctx, uppercase(string(msg)))
        reverse_handler = (msg, ctx) -> yield_output(ctx, reverse(string(msg)))

        definition = Dict{String, Any}(
            "name" => "ExecuteTest",
            "start" => "upper",
            "outputs" => ["reverse"],
            "executors" => [
                Dict{String, Any}("id" => "upper", "description" => "Uppercase",
                     "input_types" => ["String"], "output_types" => ["String"]),
                Dict{String, Any}("id" => "reverse", "description" => "Reverse",
                     "input_types" => ["String"], "output_types" => ["String"]),
            ],
            "edges" => [
                Dict{String, Any}("kind" => "direct", "source" => "upper", "target" => "reverse"),
            ]
        )

        wf = workflow_from_dict(definition; handlers=Dict{String, Function}(
            "upper" => upper_handler,
            "reverse" => reverse_handler,
        ))

        result = run_workflow(wf, "hello")
        outputs = get_outputs(result)
        @test length(outputs) >= 1
        @test "OLLEH" in outputs
    end

    # ── 14. Missing start executor throws ────────────────────────────────────
    @testset "Missing start executor throws" begin
        @test_throws KeyError workflow_from_dict(Dict{String, Any}(
            "executors" => [Dict{String, Any}("id" => "a")],
        ))
    end

    # ── 15. Default name and max_iterations ──────────────────────────────────
    @testset "Default name and max_iterations" begin
        definition = Dict{String, Any}(
            "start" => "a",
            "executors" => [Dict{String, Any}("id" => "a")],
        )
        wf = workflow_from_dict(definition; allow_missing_handlers=true)
        @test wf.name == "Workflow"
        @test wf.max_iterations == 100
    end

    @testset "workflow_from_yaml supports YAML definitions" begin
        yaml = """
        kind: Workflow
        name: YamlWorkflow
        start: step1
        outputs: result
        executors:
          step1:
            description: First step
            input_types: String
            output_types: String
          result:
            description: Final step
            input_types:
              - String
            output_types:
              - String
        edges:
          - kind: direct
            source: step1
            target: result
        """

        workflow = workflow_from_yaml(yaml; allow_missing_handlers = true)
        @test workflow.name == "YamlWorkflow"
        @test workflow.start_executor_id == "step1"
        @test workflow.output_executor_ids == ["result"]
        @test length(workflow.executors) == 2
        @test workflow.executors["step1"].input_types == DataType[String]
    end

    @testset "workflow_from_dict supports switch edges" begin
        empty!(AgentFramework._HANDLER_REGISTRY)

        is_integer = value -> value isa Integer
        register_handler!("is_integer", is_integer)

        definition = Dict{String, Any}(
            "start" => "router",
            "executors" => [
                Dict{String, Any}("id" => "router"),
                Dict{String, Any}("id" => "numbers"),
                Dict{String, Any}("id" => "fallback"),
            ],
            "edges" => [
                Dict{String, Any}(
                    "kind" => "switch",
                    "source" => "router",
                    "cases" => [
                        Dict{String, Any}("condition" => "is_integer", "target" => "numbers"),
                    ],
                    "default" => "fallback",
                ),
            ],
        )

        workflow = workflow_from_dict(definition; allow_missing_handlers = true)
        @test length(workflow.edge_groups) == 2
        @test Set(only(target_executor_ids(group)) for group in workflow.edge_groups) == Set(["numbers", "fallback"])

        empty!(AgentFramework._HANDLER_REGISTRY)
    end

    @testset "YAML workflow file roundtrip" begin
        empty!(AgentFramework._HANDLER_REGISTRY)
        handler = (msg, ctx) -> send_message(ctx, msg)
        register_handler!("echo", handler)
        definition = Dict{String, Any}(
            "name" => "YamlFileTest",
            "start" => "a",
            "outputs" => ["a"],
            "executors" => [
                Dict{String, Any}("id" => "a", "handler" => "echo"),
            ],
        )

        workflow = workflow_from_dict(definition)
        tmpfile = tempname() * ".yaml"
        try
            workflow_to_file(workflow, tmpfile)
            @test isfile(tmpfile)
            raw = read(tmpfile, String)
            @test occursin("kind: \"Workflow\"", raw)

            roundtripped = workflow_from_file(tmpfile)
            @test roundtripped.name == "YamlFileTest"
            @test roundtripped.start_executor_id == "a"
        finally
            isfile(tmpfile) && rm(tmpfile)
            empty!(AgentFramework._HANDLER_REGISTRY)
        end
    end

end

@testset "Declarative Agents" begin
    @testset "tool/client/context registries" begin
        empty!(AgentFramework._TOOL_REGISTRY)
        empty!(AgentFramework._CLIENT_REGISTRY)
        empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)

        client = DeclarativeMockChatClient("registry hello")
        provider = DeclarativeTrackingProvider(source_id = "registry")

        register_tool!("decl_echo", declarative_echo_tool)
        register_client!("decl_client", client)
        register_context_provider!("decl_provider", provider)

        @test get_tool("decl_echo") === declarative_echo_tool
        @test get_client("decl_client") === client
        @test get_context_provider("decl_provider") === provider

        empty!(AgentFramework._TOOL_REGISTRY)
        empty!(AgentFramework._CLIENT_REGISTRY)
        empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)
    end

    @testset "agent_from_yaml resolves registered refs" begin
        empty!(AgentFramework._TOOL_REGISTRY)
        empty!(AgentFramework._CLIENT_REGISTRY)
        empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)

        client = DeclarativeMockChatClient("registered hello")
        provider = DeclarativeTrackingProvider(source_id = "tracking")
        register_tool!("decl_echo", declarative_echo_tool)
        register_client!("local-client", client)
        register_context_provider!("tracking", provider)

        yaml = """
        kind: Prompt
        name: RegisteredAgent
        instructions: Be concise.
        client: local-client
        tools:
          - decl_echo
        contextProviders:
          - tracking
        options:
          temperature: 0.1
          custom_seed: 7
        """

        agent = agent_from_yaml(yaml)
        response = run_agent(agent, "hello")

        @test agent.client === client
        @test agent.tools == [declarative_echo_tool]
        @test agent.context_providers == Any[provider]
        @test agent.options.temperature == 0.1
        @test agent.options.additional["custom_seed"] == 7
        @test response.text == "registered hello"
        @test provider.events == ["before:tracking", "after:tracking:1"]

        empty!(AgentFramework._TOOL_REGISTRY)
        empty!(AgentFramework._CLIENT_REGISTRY)
        empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)
    end

    @testset "agent_from_yaml supports inline model definitions" begin
        yaml = """
        kind: Prompt
        name: InlineAgent
        instructions: Use local inference.
        model:
          provider: ollama
          id: qwen3:8b
          baseUrl: http://localhost:11434
        options:
          temperature: 0.2
          stop:
            - DONE
          seed: 11
        """

        agent = agent_from_yaml(yaml)
        @test agent.client isa OllamaChatClient
        @test agent.client.model == "qwen3:8b"
        @test agent.client.base_url == "http://localhost:11434"
        @test agent.options.temperature == 0.2
        @test agent.options.stop == ["DONE"]
        @test agent.options.additional["seed"] == 11
    end

    @testset "agent YAML roundtrip" begin
        empty!(AgentFramework._TOOL_REGISTRY)
        empty!(AgentFramework._CLIENT_REGISTRY)
        empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)

        client = DeclarativeMockChatClient("roundtrip")
        provider = DeclarativeTrackingProvider(source_id = "roundtrip")
        register_tool!("decl_echo", declarative_echo_tool)
        register_client!("roundtrip-client", client)
        register_context_provider!("roundtrip-provider", provider)

        agent = Agent(
            name = "RoundTripAgent",
            instructions = "Round-trip me.",
            client = client,
            tools = [declarative_echo_tool],
            context_providers = Any[provider],
            options = ChatOptions(max_tokens = 32, additional = Dict{String, Any}("seed" => 3)),
        )

        yaml = agent_to_yaml(agent)
        restored = agent_from_yaml(yaml)

        @test occursin("roundtrip-client", yaml)
        @test restored.client === client
        @test restored.tools == [declarative_echo_tool]
        @test restored.context_providers == Any[provider]
        @test restored.options.max_tokens == 32
        @test restored.options.additional["seed"] == 3

        empty!(AgentFramework._TOOL_REGISTRY)
        empty!(AgentFramework._CLIENT_REGISTRY)
        empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)
    end

    @testset "agent_to_file / agent_from_file use YAML extension" begin
        empty!(AgentFramework._TOOL_REGISTRY)
        empty!(AgentFramework._CLIENT_REGISTRY)
        empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)

        client = DeclarativeMockChatClient("file roundtrip")
        provider = DeclarativeTrackingProvider(source_id = "file")
        register_tool!("decl_echo", declarative_echo_tool)
        register_client!("file-client", client)
        register_context_provider!("file-provider", provider)

        agent = Agent(
            name = "FileAgent",
            instructions = "Persist me.",
            client = client,
            tools = [declarative_echo_tool],
            context_providers = Any[provider],
        )

        tmpfile = tempname() * ".yaml"
        try
            agent_to_file(agent, tmpfile)
            @test isfile(tmpfile)
            raw = read(tmpfile, String)
            @test occursin("kind: \"Prompt\"", raw)

            restored = agent_from_file(tmpfile)
            @test restored.name == "FileAgent"
            @test restored.client === client
        finally
            isfile(tmpfile) && rm(tmpfile)
            empty!(AgentFramework._TOOL_REGISTRY)
            empty!(AgentFramework._CLIENT_REGISTRY)
            empty!(AgentFramework._CONTEXT_PROVIDER_REGISTRY)
        end
    end
end
