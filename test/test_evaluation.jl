using Test
using AgentFramework

# ─── Helper: build conversations ─────────────────────────────────────────────

function make_conversation(pairs::Vector{Tuple{Symbol, String}})
    [Message(role, [text_content(text)]) for (role, text) in pairs]
end

function make_tool_conversation()
    [
        Message(:user, [text_content("What's the weather?")]),
        Message(:assistant, [
            function_call_content("c1", "get_weather", """{"location":"NYC"}"""),
        ]),
        Message(:tool, [
            function_result_content("c1", "72°F sunny"),
        ]),
        Message(:assistant, [text_content("The weather in NYC is 72°F and sunny.")]),
    ]
end

# ─── Mock agent for evaluate_agent testing ───────────────────────────────────

mutable struct EvalMockClient <: AbstractChatClient
    response_text::String
    tool_calls::Vector{Content}
end
EvalMockClient(text::String="Mock response") = EvalMockClient(text, Content[])

function AgentFramework.get_response(client::EvalMockClient, messages::Vector{Message},
                                      options=nothing)
    contents = Content[text_content(client.response_text)]
    append!(contents, client.tool_calls)
    msg = Message(:assistant, contents)
    return ChatResponse(messages=[msg], finish_reason=STOP)
end


@testset "Evaluation Framework" begin

    # ═══════════════════════════════════════════════════════════════════════
    #  1. Conversation Splitters
    # ═══════════════════════════════════════════════════════════════════════

    @testset "Conversation Splitters" begin
        @testset "SPLIT_LAST_TURN — basic" begin
            conv = make_conversation([
                (:user, "Hello"),
                (:assistant, "Hi there!"),
                (:user, "What's the weather?"),
                (:assistant, "It's sunny."),
            ])
            q, r = split_last_turn(conv)
            @test length(q) == 3  # up to and including last user msg
            @test length(r) == 1  # the final assistant response
            @test q[end].role == :user
            @test get_text(r[1]) == "It's sunny."
        end

        @testset "SPLIT_LAST_TURN — single turn" begin
            conv = make_conversation([(:user, "Hello"), (:assistant, "Hi")])
            q, r = split_last_turn(conv)
            @test length(q) == 1
            @test length(r) == 1
        end

        @testset "SPLIT_LAST_TURN — no user messages" begin
            conv = make_conversation([(:assistant, "Unprompted response")])
            q, r = split_last_turn(conv)
            @test isempty(q)
            @test length(r) == 1
        end

        @testset "SPLIT_FULL — basic" begin
            conv = make_conversation([
                (:system, "You are helpful"),
                (:user, "First question"),
                (:assistant, "Answer 1"),
                (:user, "Second question"),
                (:assistant, "Answer 2"),
            ])
            q, r = split_full(conv)
            @test length(q) == 2  # system + first user
            @test length(r) == 3  # rest of conversation
            @test q[end].role == :user
        end

        @testset "SPLIT_FULL — no user messages" begin
            conv = make_conversation([(:system, "System prompt")])
            q, r = split_full(conv)
            @test isempty(q)
            @test length(r) == 1
        end

        @testset "Custom splitter" begin
            function custom_split(conv)
                # Split at index 1
                return conv[1:1], conv[2:end]
            end
            conv = make_conversation([(:user, "A"), (:assistant, "B"), (:user, "C")])
            q, r = custom_split(conv)
            @test length(q) == 1
            @test length(r) == 2
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  2. EvalItem
    # ═══════════════════════════════════════════════════════════════════════

    @testset "EvalItem" begin
        @testset "eval_query and eval_response" begin
            conv = make_conversation([
                (:user, "What is 2+2?"),
                (:assistant, "The answer is 4."),
            ])
            item = EvalItem(conversation=conv)
            @test eval_query(item) == "What is 2+2?"
            @test eval_response(item) == "The answer is 4."
        end

        @testset "eval_query — multi-turn last_turn" begin
            conv = make_conversation([
                (:user, "Hello"),
                (:assistant, "Hi"),
                (:user, "What's the weather?"),
                (:assistant, "It's sunny."),
            ])
            item = EvalItem(conversation=conv)
            @test eval_query(item) == "Hello What's the weather?"  # all user msgs in query portion
        end

        @testset "eval_response — custom split strategy" begin
            conv = make_conversation([
                (:user, "Q1"),
                (:assistant, "A1"),
                (:user, "Q2"),
                (:assistant, "A2"),
            ])
            item = EvalItem(conversation=conv, split_strategy=split_full)
            @test eval_response(item) == "A1 A2"  # all assistant msgs after first user
        end

        @testset "split_messages" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            item = EvalItem(conversation=conv)
            q, r = split_messages(item)
            @test length(q) == 1
            @test length(r) == 1

            # Explicit override
            q2, r2 = split_messages(item; split=split_full)
            @test length(q2) == 1
        end

        @testset "per_turn_items" begin
            conv = make_conversation([
                (:user, "Q1"),
                (:assistant, "A1"),
                (:user, "Q2"),
                (:assistant, "A2"),
                (:user, "Q3"),
                (:assistant, "A3"),
            ])
            items = per_turn_items(conv)
            @test length(items) == 3
            # First item: conversation up to A1
            @test length(items[1].conversation) == 2
            # Second item: conversation up to A2
            @test length(items[2].conversation) == 4
            # Third item: full conversation
            @test length(items[3].conversation) == 6
        end

        @testset "per_turn_items — no user messages" begin
            conv = make_conversation([(:assistant, "No user")])
            @test isempty(per_turn_items(conv))
        end

        @testset "per_turn_items — with tools and context" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            tools = [FunctionTool(name="test_tool", description="test", func=identity)]
            items = per_turn_items(conv; tools=tools, context="Background info")
            @test length(items) == 1
            @test items[1].tools == tools
            @test items[1].context == "Background info"
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  3. ExpectedToolCall
    # ═══════════════════════════════════════════════════════════════════════

    @testset "ExpectedToolCall" begin
        @testset "Name only" begin
            etc = ExpectedToolCall(name="get_weather")
            @test etc.name == "get_weather"
            @test etc.arguments === nothing
        end

        @testset "Name with arguments" begin
            etc = ExpectedToolCall(name="search", arguments=Dict{String,Any}("query" => "hello"))
            @test etc.arguments["query"] == "hello"
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  4. Score and Result Types
    # ═══════════════════════════════════════════════════════════════════════

    @testset "Result Types" begin
        @testset "EvalScoreResult" begin
            s = EvalScoreResult(name="relevance", score=0.8, passed=true)
            @test s.name == "relevance"
            @test s.score == 0.8
            @test s.passed == true
        end

        @testset "EvalItemResult — predicates" begin
            pass_item = EvalItemResult(item_id="0", status="pass")
            fail_item = EvalItemResult(item_id="1", status="fail")
            err_item = EvalItemResult(item_id="2", status="error")

            @test is_passed(pass_item) == true
            @test is_failed(pass_item) == false
            @test is_error(pass_item) == false

            @test is_passed(fail_item) == false
            @test is_failed(fail_item) == true

            @test is_error(err_item) == true
        end

        @testset "EvalResults — basic" begin
            r = EvalResults(provider="Test",
                result_counts=Dict("passed" => 3, "failed" => 1, "errored" => 0))
            @test eval_passed(r) == 3
            @test eval_failed(r) == 1
            @test eval_total(r) == 4
            @test all_passed(r) == false
        end

        @testset "EvalResults — all passed" begin
            r = EvalResults(provider="Test",
                result_counts=Dict("passed" => 5, "failed" => 0, "errored" => 0))
            @test all_passed(r) == true
        end

        @testset "EvalResults — non-completed status" begin
            r = EvalResults(provider="Test", status="failed",
                result_counts=Dict("passed" => 5, "failed" => 0, "errored" => 0))
            @test all_passed(r) == false
        end

        @testset "EvalResults — with sub_results" begin
            sub1 = EvalResults(provider="Test",
                result_counts=Dict("passed" => 2, "failed" => 0, "errored" => 0))
            sub2 = EvalResults(provider="Test",
                result_counts=Dict("passed" => 1, "failed" => 1, "errored" => 0))
            parent = EvalResults(provider="Test",
                result_counts=Dict("passed" => 3, "failed" => 0, "errored" => 0),
                sub_results=Dict("agent1" => sub1, "agent2" => sub2))
            @test all_passed(parent) == false  # sub2 has failures
        end

        @testset "raise_for_status — passing" begin
            r = EvalResults(provider="Test",
                result_counts=Dict("passed" => 1, "failed" => 0, "errored" => 0))
            @test raise_for_status(r) === nothing
        end

        @testset "raise_for_status — failing" begin
            r = EvalResults(provider="Test",
                result_counts=Dict("passed" => 1, "failed" => 1, "errored" => 0))
            @test_throws EvalNotPassedError raise_for_status(r)
        end

        @testset "raise_for_status — custom message" begin
            r = EvalResults(provider="Test",
                result_counts=Dict("passed" => 0, "failed" => 1, "errored" => 0))
            try
                raise_for_status(r; msg="Custom failure")
                @test false  # should not reach
            catch e
                @test e isa EvalNotPassedError
                @test occursin("Custom failure", e.message)
            end
        end

        @testset "raise_for_status — with errored items" begin
            r = EvalResults(provider="Test",
                result_counts=Dict("passed" => 0, "failed" => 0, "errored" => 2))
            @test_throws EvalNotPassedError raise_for_status(r)
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  5. Built-in Checks
    # ═══════════════════════════════════════════════════════════════════════

    @testset "Built-in Checks" begin
        @testset "keyword_check — all present" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "The weather is sunny and warm")])
            item = EvalItem(conversation=conv)
            check = keyword_check("weather", "sunny")
            result = check(item)
            @test result.passed == true
            @test result.check_name == "keyword_check"
        end

        @testset "keyword_check — missing keyword" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "It's a nice day")])
            item = EvalItem(conversation=conv)
            check = keyword_check("weather", "temperature")
            result = check(item)
            @test result.passed == false
            @test occursin("Missing keywords", result.reason)
        end

        @testset "keyword_check — case insensitive" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "WEATHER is great")])
            item = EvalItem(conversation=conv)
            check = keyword_check("weather")
            @test check(item).passed == true
        end

        @testset "keyword_check — case sensitive" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "WEATHER is great")])
            item = EvalItem(conversation=conv)
            check = keyword_check("weather"; case_sensitive=true)
            @test check(item).passed == false
        end

        @testset "tool_called_check — all mode" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv)
            check = tool_called_check("get_weather")
            result = check(item)
            @test result.passed == true
        end

        @testset "tool_called_check — missing tool" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv)
            check = tool_called_check("get_weather", "get_flight")
            result = check(item)
            @test result.passed == false
            @test occursin("get_flight", result.reason)
        end

        @testset "tool_called_check — any mode" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv)
            check = tool_called_check("get_weather", "get_flight"; mode=:any)
            result = check(item)
            @test result.passed == true
        end

        @testset "tool_called_check — any mode none found" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "No tools")])
            item = EvalItem(conversation=conv)
            check = tool_called_check("missing_tool"; mode=:any)
            result = check(item)
            @test result.passed == false
        end

        @testset "tool_calls_present — all expected present" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv,
                expected_tool_calls=[ExpectedToolCall(name="get_weather")])
            result = tool_calls_present(item)
            @test result.passed == true
        end

        @testset "tool_calls_present — missing" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv,
                expected_tool_calls=[ExpectedToolCall(name="get_weather"),
                                    ExpectedToolCall(name="get_news")])
            result = tool_calls_present(item)
            @test result.passed == false
            @test occursin("get_news", result.reason)
        end

        @testset "tool_calls_present — no expected" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            item = EvalItem(conversation=conv)
            result = tool_calls_present(item)
            @test result.passed == true
        end

        @testset "tool_call_args_match — matching args" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv,
                expected_tool_calls=[ExpectedToolCall(name="get_weather",
                    arguments=Dict{String,Any}("location" => "NYC"))])
            result = tool_call_args_match(item)
            @test result.passed == true
        end

        @testset "tool_call_args_match — wrong args" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv,
                expected_tool_calls=[ExpectedToolCall(name="get_weather",
                    arguments=Dict{String,Any}("location" => "London"))])
            result = tool_call_args_match(item)
            @test result.passed == false
        end

        @testset "tool_call_args_match — name only (no arg check)" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv,
                expected_tool_calls=[ExpectedToolCall(name="get_weather")])
            result = tool_call_args_match(item)
            @test result.passed == true
        end

        @testset "tool_call_args_match — tool not called" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            item = EvalItem(conversation=conv,
                expected_tool_calls=[ExpectedToolCall(name="missing_tool")])
            result = tool_call_args_match(item)
            @test result.passed == false
            @test occursin("not called", result.reason)
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  6. Function evaluator wrapper (make_evaluator)
    # ═══════════════════════════════════════════════════════════════════════

    @testset "make_evaluator" begin
        @testset "Bool return" begin
            fn = make_evaluator(; name="bool_check") do response
                length(response) > 5
            end
            conv = make_conversation([(:user, "Q"), (:assistant, "Short")])
            item = EvalItem(conversation=conv)
            result = fn(item)
            @test result isa CheckResult
            @test result.passed == false  # "Short" is 5 chars, not > 5

            conv2 = make_conversation([(:user, "Q"), (:assistant, "Long enough response")])
            item2 = EvalItem(conversation=conv2)
            @test fn(item2).passed == true
        end

        @testset "Numeric return" begin
            fn = make_evaluator(; name="score_fn") do response
                return 0.75  # above 0.5 threshold
            end
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            result = fn(EvalItem(conversation=conv))
            @test result.passed == true
            @test occursin("0.75", result.reason)
        end

        @testset "Dict with score" begin
            fn = make_evaluator(; name="dict_score") do response
                return Dict{String,Any}("score" => 0.3)
            end
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            result = fn(EvalItem(conversation=conv))
            @test result.passed == false  # 0.3 < 0.5
        end

        @testset "Dict with passed" begin
            fn = make_evaluator(; name="dict_passed") do response
                return Dict{String,Any}("passed" => true, "reason" => "good enough")
            end
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            result = fn(EvalItem(conversation=conv))
            @test result.passed == true
            @test result.reason == "good enough"
        end

        @testset "CheckResult passthrough" begin
            fn = make_evaluator(; name="cr_fn") do response
                return CheckResult(passed=true, reason="custom", check_name="inner")
            end
            conv = make_conversation([(:user, "Q"), (:assistant, "A")])
            result = fn(EvalItem(conversation=conv))
            @test result.passed == true
            @test result.check_name == "inner"
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  7. LocalEvaluator
    # ═══════════════════════════════════════════════════════════════════════

    @testset "LocalEvaluator" begin
        @testset "All checks pass" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "The weather is sunny")])
            item = EvalItem(conversation=conv)
            local_eval = LocalEvaluator(keyword_check("weather", "sunny"))
            results = evaluate(local_eval, [item])
            @test results.provider == "Local"
            @test eval_passed(results) == 1
            @test eval_failed(results) == 0
            @test all_passed(results) == true
        end

        @testset "Some checks fail" begin
            conv = make_conversation([(:user, "Q"), (:assistant, "It's nice out")])
            item = EvalItem(conversation=conv)
            local_eval = LocalEvaluator(
                keyword_check("nice"),
                keyword_check("temperature"),
            )
            results = evaluate(local_eval, [item])
            @test eval_passed(results) == 0
            @test eval_failed(results) == 1  # item fails because second check fails
        end

        @testset "Multiple items" begin
            items = [
                EvalItem(conversation=make_conversation([(:user, "Q"), (:assistant, "sunny weather")])),
                EvalItem(conversation=make_conversation([(:user, "Q"), (:assistant, "rainy weather")])),
                EvalItem(conversation=make_conversation([(:user, "Q"), (:assistant, "no info")])),
            ]
            local_eval = LocalEvaluator(keyword_check("weather"))
            results = evaluate(local_eval, items)
            @test eval_passed(results) == 2
            @test eval_failed(results) == 1
            @test length(results.items) == 3
            @test results.items[1].status == "pass"
            @test results.items[3].status == "fail"
        end

        @testset "Per-evaluator breakdown" begin
            conv = make_tool_conversation()
            item = EvalItem(conversation=conv)
            local_eval = LocalEvaluator(
                keyword_check("sunny"),
                tool_called_check("get_weather"),
            )
            results = evaluate(local_eval, [item])
            @test haskey(results.per_evaluator, "keyword_check")
            @test haskey(results.per_evaluator, "tool_called")
            @test results.per_evaluator["keyword_check"]["passed"] == 1
            @test results.per_evaluator["tool_called"]["passed"] == 1
        end

        @testset "Item detail" begin
            conv = make_conversation([(:user, "What's the weather?"), (:assistant, "It's sunny!")])
            item = EvalItem(conversation=conv)
            local_eval = LocalEvaluator(keyword_check("sunny"))
            results = evaluate(local_eval, [item])

            @test length(results.items) == 1
            detail = results.items[1]
            @test detail.item_id == "0"
            @test detail.status == "pass"
            @test detail.input_text == "What's the weather?"
            @test detail.output_text == "It's sunny!"
            @test length(detail.scores) == 1
            @test detail.scores[1].name == "keyword_check"
            @test detail.scores[1].passed == true
        end

        @testset "Mixed evaluators auto-resolved" begin
            check1 = keyword_check("hello")
            conv = make_conversation([(:user, "Q"), (:assistant, "hello world")])
            items = [EvalItem(conversation=conv)]
            results_list = AgentFramework._run_evaluators([check1], items; eval_name="test")
            @test length(results_list) == 1
            @test results_list[1].provider == "Local"
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  8. evaluate_agent
    # ═══════════════════════════════════════════════════════════════════════

    @testset "evaluate_agent" begin
        @testset "With pre-existing responses" begin
            resp_msg = Message(:assistant, [text_content("The weather is sunny and 72°F.")])
            response = AgentResponse(messages=[
                Message(:user, [text_content("What's the weather?")]),
                resp_msg,
            ])

            results = evaluate_agent(
                responses=[response],
                queries=["What's the weather?"],
                evaluators=LocalEvaluator(keyword_check("sunny")),
            )
            @test length(results) == 1
            @test all_passed(results[1])
        end

        @testset "With agent execution" begin
            client = EvalMockClient("The weather is great and sunny.")
            agent = Agent(name="test_agent", client=client,
                          instructions="You are a weather bot.")
            results = evaluate_agent(
                agent=agent,
                queries=["What's the weather?"],
                evaluators=LocalEvaluator(keyword_check("sunny")),
            )
            @test length(results) == 1
            @test all_passed(results[1])
        end

        @testset "Multiple queries" begin
            client = EvalMockClient("sunny weather report")
            agent = Agent(name="test_agent", client=client, instructions="test")
            results = evaluate_agent(
                agent=agent,
                queries=["Q1", "Q2", "Q3"],
                evaluators=LocalEvaluator(keyword_check("sunny")),
            )
            @test eval_total(results[1]) == 3
            @test eval_passed(results[1]) == 3
        end

        @testset "Expected output stamping" begin
            resp_msg = Message(:assistant, [text_content("4")])
            response = AgentResponse(messages=[
                Message(:user, [text_content("What's 2+2?")]),
                resp_msg,
            ])

            exact_match = make_evaluator(; name="exact_match") do response, expected_output
                strip(response) == strip(expected_output)
            end

            results = evaluate_agent(
                responses=[response],
                queries=["What's 2+2?"],
                expected_output=["4"],
                evaluators=LocalEvaluator(exact_match),
            )
            @test all_passed(results[1])
        end

        @testset "num_repetitions" begin
            client = EvalMockClient("consistent response")
            agent = Agent(name="test", client=client, instructions="test")
            results = evaluate_agent(
                agent=agent,
                queries=["Q1"],
                evaluators=LocalEvaluator(keyword_check("consistent")),
                num_repetitions=3,
            )
            @test eval_total(results[1]) == 3  # 1 query × 3 reps
        end

        @testset "Validation errors" begin
            @test_throws ArgumentError evaluate_agent(evaluators=LocalEvaluator())
            @test_throws ArgumentError evaluate_agent(queries="Q", evaluators=LocalEvaluator())
            @test_throws ArgumentError evaluate_agent(
                queries=["Q1", "Q2"],
                expected_output=["only one"],
                evaluators=LocalEvaluator(),
                responses=[
                    AgentResponse(messages=[Message(:assistant, [text_content("A")])]),
                    AgentResponse(messages=[Message(:assistant, [text_content("B")])]),
                ],
            )
        end

        @testset "num_repetitions validation" begin
            @test_throws ArgumentError evaluate_agent(
                agent=Agent(name="t", client=EvalMockClient(), instructions="t"),
                queries=["Q"],
                evaluators=LocalEvaluator(),
                num_repetitions=0,
            )
        end
    end

    @testset "workflow eval extraction" begin
        conversation = make_conversation([
            (:user, "Draft a launch announcement"),
            (:assistant, "Review: Draft a launch announcement"),
        ])
        result = WorkflowRunResult(
            events = [
                event_executor_invoked("reviewer"),
                event_executor_completed("reviewer", Dict("conversation" => conversation)),
            ],
            state = WF_IDLE,
        )

        agent_data = AgentFramework._extract_agent_eval_data(result)
        @test haskey(agent_data, "reviewer")
        @test length(agent_data["reviewer"]) == 1
        @test agent_data["reviewer"][1].conversation == conversation
    end

    # ═══════════════════════════════════════════════════════════════════════
    #  9. WorkflowAgent
    # ═══════════════════════════════════════════════════════════════════════

    @testset "WorkflowAgent" begin
        @testset "Construction" begin
            # Can't easily test with real workflow without full setup,
            # but test struct creation
            wa = WorkflowAgent(workflow=nothing, name="my_workflow")
            @test wa.name == "my_workflow"
            @test wa isa AbstractAgent
            @test wa.description === nothing
        end
    end

    # ═══════════════════════════════════════════════════════════════════════
    # 10. EvalNotPassedError
    # ═══════════════════════════════════════════════════════════════════════

    @testset "EvalNotPassedError" begin
        e = EvalNotPassedError("Test failure message")
        buf = IOBuffer()
        showerror(buf, e)
        @test occursin("Test failure message", String(take!(buf)))
    end

    # ═══════════════════════════════════════════════════════════════════════
    # 11. Integration: end-to-end evaluation flow
    # ═══════════════════════════════════════════════════════════════════════

    @testset "Integration — full eval flow" begin
        client = EvalMockClient("The weather in NYC is 72°F and sunny.")
        agent = Agent(name="weather_bot", client=client,
                      instructions="You are a helpful weather assistant.")

        # Multiple evaluators including keyword and custom
        length_ok = make_evaluator(; name="length_ok") do response
            length(response) > 10
        end

        results = evaluate_agent(
            agent=agent,
            queries=["What's the weather in NYC?", "Tell me about weather"],
            evaluators=[
                keyword_check("weather"),
                keyword_check("sunny"),
                length_ok,
            ],
            eval_name="Weather Bot Eval",
        )

        # All bare functions grouped into a single LocalEvaluator
        @test length(results) == 1
        @test eval_total(results[1]) == 2
        @test all_passed(results[1])

        # Verify per-item detail
        @test length(results[1].items) == 2
        for item in results[1].items
            @test item.status == "pass"
            @test length(item.scores) == 3  # 3 checks
        end
    end

    @testset "Integration — failing eval with raise_for_status" begin
        client = EvalMockClient("No relevant info here.")
        agent = Agent(name="bad_bot", client=client, instructions="test")

        results = evaluate_agent(
            agent=agent,
            queries=["What's the weather?"],
            evaluators=LocalEvaluator(keyword_check("sunny", "temperature")),
        )

        @test !all_passed(results[1])
        @test_throws EvalNotPassedError raise_for_status(results[1])
    end

end  # Evaluation Framework
