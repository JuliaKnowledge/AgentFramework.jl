using AgentFramework
using Test

# Mock chat client that scripts assistant turns. Each turn can be either:
# - a plain string (single text message)
# - a tuple (:handoff, "target_id") — emits a handoff_to_<target> function call
# - a tuple (:tool, "tool_name", "args") — emits a non-handoff tool call
# - a Vector of any of the above to emit multiple contents in one message
mutable struct HandoffOrchMockClient <: AbstractChatClient
    scripts::Dict{String, Vector{Any}}
    counters::Dict{String, Int}
    received::Vector{Tuple{String, Vector{Message}}}
    name::String
end

HandoffOrchMockClient(name::String, turns::Vector) =
    HandoffOrchMockClient(Dict(name => turns), Dict(name => 0), Tuple{String, Vector{Message}}[], name)

function _build_contents(spec)
    if spec isa AbstractString
        return [text_content(spec)]
    elseif spec isa Tuple
        kind = spec[1]
        if kind === :handoff
            target = spec[2]
            return [function_call_content("call_$(target)_$(rand(UInt16))",
                AgentFramework.get_handoff_tool_name(target), "{}")]
        elseif kind === :tool
            return [function_call_content("call_$(spec[2])_$(rand(UInt16))",
                spec[2], length(spec) >= 3 ? spec[3] : "{}")]
        elseif kind === :multi
            out = AgentFramework.Content[]
            for item in spec[2]
                append!(out, _build_contents(item))
            end
            return out
        end
    end
    error("Unknown script item: $spec")
end

function AgentFramework.get_response(client::HandoffOrchMockClient, messages::Vector{Message},
                                     options::ChatOptions)::ChatResponse
    push!(client.received, (client.name, deepcopy(messages)))
    client.counters[client.name] = get(client.counters, client.name, 0) + 1
    idx = min(client.counters[client.name], length(client.scripts[client.name]))
    spec = client.scripts[client.name][idx]
    contents = _build_contents(spec)
    finish = any(c -> AgentFramework.is_function_call(c), contents) ?
        AgentFramework.TOOL_CALLS : AgentFramework.STOP
    ChatResponse(
        messages = [Message(role = :assistant, contents = contents)],
        finish_reason = finish,
        model_id = "handoff-mock",
    )
end

function AgentFramework.get_response_streaming(client::HandoffOrchMockClient, messages::Vector{Message},
                                               options::ChatOptions)::Channel{ChatResponseUpdate}
    response = AgentFramework.get_response(client, messages, options)
    ch = Channel{ChatResponseUpdate}(1)
    Threads.@spawn begin
        for msg in response.messages
            for c in msg.contents
                put!(ch, ChatResponseUpdate(role = msg.role, contents = [c]))
            end
        end
        put!(ch, ChatResponseUpdate(finish_reason = response.finish_reason))
        close(ch)
    end
    return ch
end

make_handoff_agent(name::String, turns::Vector; tools = FunctionTool[]) = begin
    client = HandoffOrchMockClient(name, turns)
    Agent(name = name, instructions = "Be concise.", client = client, tools = tools), client
end

@testset "Handoff Orchestration" begin

    @testset "Construction validation" begin
        a, _ = make_handoff_agent("a", ["hi"])
        b, _ = make_handoff_agent("b", ["hi"])

        # Empty participants
        @test_throws ArgumentError build(HandoffBuilder())

        # Self-handoff
        builder = HandoffBuilder(participants = [a, b])
        @test_throws ArgumentError add_handoff(builder, "a", "a")

        # Unknown agent
        @test_throws ArgumentError add_handoff(builder, "a", "missing")

        # Tool collision: agent already has a tool named handoff_to_*
        bad_tool = FunctionTool(name = "handoff_to_b", description = "x",
                                func = (_) -> "x",
                                parameters = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}()))
        bad, _ = make_handoff_agent("c", ["hi"], tools = [bad_tool])
        bad_builder = HandoffBuilder(participants = [bad, a])
        @test_throws ArgumentError build(bad_builder)
    end

    @testset "Single handoff routes conversation" begin
        # Triage hands off to specialist; specialist replies and stops.
        triage, triage_client = make_handoff_agent("triage", [(:handoff, "specialist")])
        specialist, _ = make_handoff_agent("specialist", ["Resolved."])

        workflow = build(HandoffBuilder(participants = [triage, specialist]))
        result = run_workflow(workflow, "I have a billing question")

        outputs = get_outputs(result)
        convs = filter(o -> o isa Vector{Message}, outputs)
        @test length(convs) == 1
        conv = only(convs)
        # First message = user, then triage's tool-call assistant msg, then specialist's reply
        @test conv[1].role == :user
        @test AgentFramework.get_text(conv[end]) == "Resolved."
    end

    @testset "First handoff in a response wins" begin
        # If an agent emits two handoff calls (to b then to c), b is taken.
        a, _ = make_handoff_agent("a", [(:multi, [(:handoff, "b"), (:handoff, "c")])])
        b, _ = make_handoff_agent("b", ["from b"])
        c, _ = make_handoff_agent("c", ["from c"])

        workflow = build(HandoffBuilder(participants = [a, b, c]))
        outputs = get_outputs(run_workflow(workflow, "go"))
        conv = only(filter(o -> o isa Vector{Message}, outputs))
        @test AgentFramework.get_text(conv[end]) == "from b"
    end

    @testset "Handoff beats co-emitted regular tool call" begin
        # If an agent emits both a normal tool call and a handoff in the same
        # response, the handoff wins and the regular tool is NOT executed.
        normal_called = Ref(false)
        normal_tool = FunctionTool(
            name = "real_tool", description = "x",
            func = (_) -> begin normal_called[] = true; "ok" end,
            parameters = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}()),
        )
        a, _ = make_handoff_agent("a",
            [(:multi, [(:tool, "real_tool", "{}"), (:handoff, "b")])],
            tools = [normal_tool])
        b, _ = make_handoff_agent("b", ["done"])

        workflow = build(HandoffBuilder(participants = [a, b]))
        outputs = get_outputs(run_workflow(workflow, "go"))
        conv = only(filter(o -> o isa Vector{Message}, outputs))
        @test AgentFramework.get_text(conv[end]) == "done"
        @test normal_called[] == false
    end

    @testset "Default mesh — every agent can hand off to every other" begin
        a, _ = make_handoff_agent("a", [(:handoff, "c")])
        b, _ = make_handoff_agent("b", ["b reply"])
        c, _ = make_handoff_agent("c", ["c reply"])

        # No add_handoff calls → default mesh; a→c should be valid.
        workflow = build(HandoffBuilder(participants = [a, b, c]))
        outputs = get_outputs(run_workflow(workflow, "go"))
        conv = only(filter(o -> o isa Vector{Message}, outputs))
        @test AgentFramework.get_text(conv[end]) == "c reply"
    end

    @testset "with_start_agent overrides default first speaker" begin
        a, _ = make_handoff_agent("a", ["a never speaks"])
        b, _ = make_handoff_agent("b", ["from b"])

        builder = HandoffBuilder(participants = [a, b])
        with_start_agent(builder, "b")
        workflow = build(builder)
        outputs = get_outputs(run_workflow(workflow, "hi"))
        conv = only(filter(o -> o isa Vector{Message}, outputs))
        @test AgentFramework.get_text(conv[end]) == "from b"
    end

    @testset "Termination predicate halts after a turn" begin
        # Termination triggers immediately after the first agent's turn.
        a, _ = make_handoff_agent("a", ["a says hi", "a says bye"])

        builder = HandoffBuilder(participants = [a])
        with_termination(builder, conv -> any(m -> AgentFramework.get_text(m) == "a says hi", conv))
        workflow = build(builder)
        outputs = get_outputs(run_workflow(workflow, "hi"))
        conv = only(filter(o -> o isa Vector{Message}, outputs))
        @test AgentFramework.get_text(conv[end]) == "a says hi"
    end

    @testset "Autonomous mode loops then stops at limit" begin
        # When no handoff occurs and autonomous mode is on, the same agent
        # is asked to continue. Limit caps the loop.
        a, client = make_handoff_agent("a", ["t1", "t2", "t3", "t4", "t5"])

        builder = HandoffBuilder(participants = [a])
        with_autonomous_mode(builder, turn_limits = Dict("a" => 2))
        workflow = build(builder)
        result = run_workflow(workflow, "begin")
        conv = only(filter(o -> o isa Vector{Message}, get_outputs(result)))
        # Initial user, then 3 assistant turns: 1st turn + 2 autonomous continues
        # before the limit yields.
        assistant_msgs = [m for m in conv if m.role == :assistant]
        @test length(assistant_msgs) == 3
    end

    @testset "HandoffSentEvent emitted on handoff" begin
        a, _ = make_handoff_agent("a", [(:handoff, "b")])
        b, _ = make_handoff_agent("b", ["done"])

        workflow = build(HandoffBuilder(participants = [a, b]))
        result = run_workflow(workflow, "go")

        outputs = get_outputs(result)
        events = filter(o -> o isa HandoffSentEvent, outputs)
        @test length(events) == 1
        ev = first(events)
        @test ev.source == "a"
        @test ev.target == "b"
    end

    @testset "get_handoff_tool_name helper" begin
        @test get_handoff_tool_name("specialist") == "handoff_to_specialist"
    end
end
