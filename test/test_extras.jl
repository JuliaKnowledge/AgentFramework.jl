using AgentFramework
using Test
using Dates

struct MockClient <: AbstractChatClient end

@testset "Extras" begin

    # ── Agent Cloning ────────────────────────────────────────────────────────
    @testset "Agent Cloning" begin
        client = MockClient()
        agent = Agent(
            name = "TestAgent",
            description = "A test agent",
            instructions = "Be helpful",
            client = client,
            options = ChatOptions(temperature=0.7),
            max_tool_iterations = 5,
        )

        @testset "deepcopy creates independent copy" begin
            clone = deepcopy(agent)
            @test clone.name == agent.name
            @test clone.instructions == agent.instructions
            @test clone.max_tool_iterations == agent.max_tool_iterations
            clone.name = "Changed"
            @test agent.name == "TestAgent"
        end

        @testset "deepcopy shares client" begin
            clone = deepcopy(agent)
            @test clone.client === agent.client
        end

        @testset "with_instructions changes only instructions" begin
            variant = with_instructions(agent, "New instructions")
            @test variant.instructions == "New instructions"
            @test variant.name == agent.name
            @test variant.client === agent.client
            @test agent.instructions == "Be helpful"
        end

        @testset "with_tools changes only tools" begin
            tool = FunctionTool(
                name = "test_tool",
                description = "A test tool",
                func = x -> "result",
                parameters = Dict{String, Any}("type" => "object", "properties" => Dict{String, Any}()),
            )
            variant = with_tools(agent, [tool])
            @test length(variant.tools) == 1
            @test variant.tools[1].name == "test_tool"
            @test isempty(agent.tools)
            @test variant.name == agent.name
        end

        @testset "with_name changes only name" begin
            variant = with_name(agent, "NewName")
            @test variant.name == "NewName"
            @test variant.instructions == agent.instructions
            @test agent.name == "TestAgent"
        end

        @testset "with_options changes only options" begin
            new_opts = ChatOptions(temperature=0.1, max_tokens=100)
            variant = with_options(agent, new_opts)
            @test variant.options.temperature == 0.1
            @test variant.options.max_tokens == 100
            @test agent.options.temperature == 0.7
        end

        @testset "Original unchanged after variant creation" begin
            _ = with_instructions(agent, "X")
            _ = with_name(agent, "Y")
            _ = with_options(agent, ChatOptions(temperature=0.0))
            @test agent.name == "TestAgent"
            @test agent.instructions == "Be helpful"
            @test agent.options.temperature == 0.7
        end
    end

    # ── Streaming Tool Call Accumulation ──────────────────────────────────────
    @testset "Streaming Tool Accumulation" begin
        @testset "Empty accumulator has no tool calls" begin
            acc = StreamingToolAccumulator()
            @test !has_tool_calls(acc)
            @test isempty(get_accumulated_tool_calls(acc))
        end

        @testset "Accumulate name then arguments" begin
            acc = StreamingToolAccumulator()
            accumulate_tool_call!(acc, 0; call_id="call_1", name="get_weather")
            accumulate_tool_call!(acc, 0; arguments_fragment="{\"loc")
            accumulate_tool_call!(acc, 0; arguments_fragment="ation\": \"London\"}")
            @test has_tool_calls(acc)
            calls = get_accumulated_tool_calls(acc)
            @test length(calls) == 1
            @test calls[1].name == "get_weather"
            @test calls[1].arguments == "{\"location\": \"London\"}"
            @test calls[1].call_id == "call_1"
        end

        @testset "Accumulate multiple tool calls by index" begin
            acc = StreamingToolAccumulator()
            accumulate_tool_call!(acc, 0; call_id="c1", name="tool_a")
            accumulate_tool_call!(acc, 1; call_id="c2", name="tool_b")
            accumulate_tool_call!(acc, 0; arguments_fragment="{}")
            accumulate_tool_call!(acc, 1; arguments_fragment="{}")
            calls = get_accumulated_tool_calls(acc)
            @test length(calls) == 2
            @test calls[1].name == "tool_a"
            @test calls[2].name == "tool_b"
        end

        @testset "get_accumulated_tool_calls returns Content items" begin
            acc = StreamingToolAccumulator()
            accumulate_tool_call!(acc, 0; call_id="c1", name="fn", arguments_fragment="{}")
            calls = get_accumulated_tool_calls(acc)
            @test calls[1] isa Content
            @test is_function_call(calls[1])
        end

        @testset "Argument fragments concatenate correctly" begin
            acc = StreamingToolAccumulator()
            accumulate_tool_call!(acc, 0; call_id="c1", name="fn")
            accumulate_tool_call!(acc, 0; arguments_fragment="{")
            accumulate_tool_call!(acc, 0; arguments_fragment="\"a\"")
            accumulate_tool_call!(acc, 0; arguments_fragment=": 1}")
            calls = get_accumulated_tool_calls(acc)
            @test calls[1].arguments == "{\"a\": 1}"
        end

        @testset "reset_accumulator! clears state" begin
            acc = StreamingToolAccumulator()
            accumulate_tool_call!(acc, 0; call_id="c1", name="fn", arguments_fragment="{}")
            @test has_tool_calls(acc)
            reset_accumulator!(acc)
            @test !has_tool_calls(acc)
            @test isempty(get_accumulated_tool_calls(acc))
        end

        @testset "Thread safety with concurrent accumulation" begin
            acc = StreamingToolAccumulator()
            tasks = Task[]
            for i in 0:9
                t = @async begin
                    accumulate_tool_call!(acc, i; call_id="c$i", name="tool_$i", arguments_fragment="{}")
                end
                push!(tasks, t)
            end
            foreach(wait, tasks)
            calls = get_accumulated_tool_calls(acc)
            @test length(calls) == 10
        end

        @testset "ChatResponse rebuilds fragmented tool calls from streaming metadata" begin
            updates = [
                ChatResponseUpdate(
                    role = :assistant,
                    contents = [function_call_content("call_1", "lookup", "")],
                    raw_representation = Dict{String, Any}(
                        "choices" => Any[
                            Dict{String, Any}(
                                "delta" => Dict{String, Any}(
                                    "tool_calls" => Any[
                                        Dict{String, Any}(
                                            "index" => 0,
                                            "id" => "call_1",
                                            "function" => Dict{String, Any}("name" => "lookup"),
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ),
                ChatResponseUpdate(
                    contents = [function_call_content("call_1", "", "{\"q\": \"weather\"}")],
                    finish_reason = TOOL_CALLS,
                    raw_representation = Dict{String, Any}(
                        "choices" => Any[
                            Dict{String, Any}(
                                "delta" => Dict{String, Any}(
                                    "tool_calls" => Any[
                                        Dict{String, Any}(
                                            "index" => 0,
                                            "function" => Dict{String, Any}("arguments" => "{\"q\": \"weather\"}"),
                                        ),
                                    ],
                                ),
                                "finish_reason" => "tool_calls",
                            ),
                        ],
                    ),
                ),
            ]

            response = ChatResponse(updates)
            calls = [content for content in response.messages[1].contents if is_function_call(content)]
            @test length(calls) == 1
            @test calls[1].name == "lookup"
            @test calls[1].arguments == "{\"q\": \"weather\"}"
            @test response.finish_reason == TOOL_CALLS
        end
    end

    # ── Turn Management ──────────────────────────────────────────────────────
    @testset "Turn Management" begin
        @testset "TurnTracker starts empty" begin
            tracker = TurnTracker()
            @test turn_count(tracker) == 0
            @test last_turn(tracker) === nothing
            @test isempty(all_turn_messages(tracker))
        end

        @testset "start_turn! creates a turn" begin
            tracker = TurnTracker()
            msgs = [Message(:user, "Hello")]
            turn = start_turn!(tracker, msgs)
            @test turn.index == 1
            @test length(turn.user_messages) == 1
            @test turn.user_messages[1].role == :user
        end

        @testset "complete_turn! finishes a turn" begin
            tracker = TurnTracker()
            start_turn!(tracker, [Message(:user, "Hi")])
            turn = complete_turn!(tracker, [Message(:assistant, "Hello!")])
            @test turn.index == 1
            @test length(turn.assistant_messages) == 1
            @test turn_count(tracker) == 1
        end

        @testset "turn_count increments" begin
            tracker = TurnTracker()
            for i in 1:3
                start_turn!(tracker, [Message(:user, "msg $i")])
                complete_turn!(tracker, [Message(:assistant, "resp $i")])
            end
            @test turn_count(tracker) == 3
        end

        @testset "get_turn by index" begin
            tracker = TurnTracker()
            start_turn!(tracker, [Message(:user, "first")])
            complete_turn!(tracker, [Message(:assistant, "r1")])
            start_turn!(tracker, [Message(:user, "second")])
            complete_turn!(tracker, [Message(:assistant, "r2")])

            t1 = get_turn(tracker, 1)
            @test t1 !== nothing
            @test t1.index == 1
            t2 = get_turn(tracker, 2)
            @test t2 !== nothing
            @test t2.index == 2
            @test get_turn(tracker, 3) === nothing
            @test get_turn(tracker, 0) === nothing
        end

        @testset "last_turn returns most recent" begin
            tracker = TurnTracker()
            start_turn!(tracker, [Message(:user, "a")])
            complete_turn!(tracker, [Message(:assistant, "b")])
            start_turn!(tracker, [Message(:user, "c")])
            complete_turn!(tracker, [Message(:assistant, "d")])
            lt = last_turn(tracker)
            @test lt !== nothing
            @test lt.index == 2
        end

        @testset "all_turn_messages returns ordered history" begin
            tracker = TurnTracker()
            start_turn!(tracker, [Message(:user, "u1")])
            complete_turn!(tracker, [Message(:assistant, "a1")])
            start_turn!(tracker, [Message(:user, "u2")])
            complete_turn!(tracker, [Message(:assistant, "a2")])
            msgs = all_turn_messages(tracker)
            @test length(msgs) == 4
            @test msgs[1].role == :user
            @test msgs[2].role == :assistant
            @test msgs[3].role == :user
            @test msgs[4].role == :assistant
        end

        @testset "complete_turn! without start throws" begin
            tracker = TurnTracker()
            @test_throws AgentError complete_turn!(tracker, [Message(:assistant, "oops")])
        end

        @testset "Multiple turns accumulate correctly" begin
            tracker = TurnTracker()
            for i in 1:5
                start_turn!(tracker, [Message(:user, "q$i")])
                complete_turn!(tracker, [Message(:assistant, "a$i")])
            end
            @test turn_count(tracker) == 5
            msgs = all_turn_messages(tracker)
            @test length(msgs) == 10
            for i in 1:5
                t = get_turn(tracker, i)
                @test t.index == i
            end
        end
    end

    # ── Embeddings ───────────────────────────────────────────────────────────
    @testset "Embeddings" begin
        @testset "get_embeddings method exists for OpenAIChatClient" begin
            @test hasmethod(get_embeddings, Tuple{OpenAIChatClient, Vector{String}})
        end

        @testset "get_embeddings method exists for OllamaChatClient" begin
            @test hasmethod(get_embeddings, Tuple{OllamaChatClient, Vector{String}})
        end

        @testset "get_embeddings method exists for AzureOpenAIChatClient" begin
            @test hasmethod(get_embeddings, Tuple{AzureOpenAIChatClient, Vector{String}})
        end

        @testset "Ollama embedding_capability trait" begin
            client = OllamaChatClient(model="test")
            @test supports_embeddings(client)
        end
    end

end
