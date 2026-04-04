using AgentFramework
using Test
using Dates
using Logging

# ── Mock Client ──────────────────────────────────────────────────────────────

struct MockTelemetryClient <: AbstractChatClient end

function AgentFramework.get_response(::MockTelemetryClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    ChatResponse(
        messages = [Message(:assistant, "mock response")],
        finish_reason = STOP,
        model_id = "mock-model-v1",
        usage_details = UsageDetails(input_tokens=10, output_tokens=5, total_tokens=15),
    )
end

# ── Error client ─────────────────────────────────────────────────────────────

struct ErrorTelemetryClient <: AbstractChatClient end

function AgentFramework.get_response(::ErrorTelemetryClient, messages::Vector{Message}, options::ChatOptions)::ChatResponse
    error("LLM call failed")
end

@testset "Telemetry" begin
    # ── 1. TelemetrySpan construction and defaults ───────────────────────────
    @testset "TelemetrySpan construction and defaults" begin
        span = TelemetrySpan(name = "test.span")
        @test !isempty(span.id)
        @test span.parent_id === nothing
        @test span.name == "test.span"
        @test span.kind == :internal
        @test span.start_time <= Dates.now(Dates.UTC)
        @test span.end_time === nothing
        @test span.status == :unset
        @test isempty(span.attributes)
        @test isempty(span.events)
    end

    # ── 2. finish_span! sets end_time and status ─────────────────────────────
    @testset "finish_span! sets end_time and status" begin
        span = TelemetrySpan(name = "finish.test")
        @test span.end_time === nothing
        @test span.status == :unset

        result = finish_span!(span; status = :ok)
        @test result === span
        @test span.end_time !== nothing
        @test span.end_time >= span.start_time
        @test span.status == :ok

        span2 = TelemetrySpan(name = "finish.error")
        finish_span!(span2; status = :error)
        @test span2.status == :error
    end

    # ── 3. add_event! appends events ─────────────────────────────────────────
    @testset "add_event! appends events" begin
        span = TelemetrySpan(name = "event.test")
        @test isempty(span.events)

        add_event!(span, "started")
        @test length(span.events) == 1
        @test span.events[1]["name"] == "started"
        @test haskey(span.events[1], "time")
        @test haskey(span.events[1], "attributes")

        add_event!(span, "checkpoint"; attributes = Dict{String, Any}("step" => 3))
        @test length(span.events) == 2
        @test span.events[2]["name"] == "checkpoint"
        @test span.events[2]["attributes"]["step"] == 3
    end

    # ── 4. duration_ms calculation ───────────────────────────────────────────
    @testset "duration_ms calculation" begin
        span = TelemetrySpan(name = "duration.test")
        @test duration_ms(span) === nothing

        finish_span!(span)
        dur = duration_ms(span)
        @test dur !== nothing
        @test dur isa Integer
        @test dur >= 0
    end

    # ── 5. InMemoryTelemetryBackend — record_span! stores spans ──────────────
    @testset "InMemoryTelemetryBackend record_span!" begin
        backend = InMemoryTelemetryBackend()
        @test isempty(get_spans(backend))

        span = TelemetrySpan(name = "store.test")
        finish_span!(span)
        record_span!(backend, span)
        @test length(get_spans(backend)) == 1
        @test get_spans(backend)[1].name == "store.test"
    end

    # ── 6. InMemoryTelemetryBackend — get_spans returns copies ───────────────
    @testset "InMemoryTelemetryBackend get_spans returns copies" begin
        backend = InMemoryTelemetryBackend()
        span = TelemetrySpan(name = "copy.test")
        finish_span!(span)
        record_span!(backend, span)

        spans1 = get_spans(backend)
        spans2 = get_spans(backend)
        @test spans1 !== spans2
        @test length(spans1) == length(spans2)

        # Mutating the returned copy shouldn't affect the backend
        empty!(spans1)
        @test length(get_spans(backend)) == 1
    end

    # ── 7. InMemoryTelemetryBackend — clear_spans! empties store ─────────────
    @testset "InMemoryTelemetryBackend clear_spans!" begin
        backend = InMemoryTelemetryBackend()
        for i in 1:3
            span = TelemetrySpan(name = "clear.$i")
            finish_span!(span)
            record_span!(backend, span)
        end
        @test length(get_spans(backend)) == 3

        clear_spans!(backend)
        @test isempty(get_spans(backend))
    end

    # ── 8. LoggingTelemetryBackend — record_span! doesn't error ──────────────
    @testset "LoggingTelemetryBackend record_span! doesn't error" begin
        backend = LoggingTelemetryBackend()
        @test backend.level == Logging.Info

        backend_debug = LoggingTelemetryBackend(Logging.Debug)
        @test backend_debug.level == Logging.Debug

        span = TelemetrySpan(
            name = "log.test",
            attributes = Dict{String, Any}("key" => "value"),
        )
        finish_span!(span)

        # Should not throw
        record_span!(backend, span)
        record_span!(backend_debug, span)
    end

    # ── 9. telemetry_agent_middleware — records span on success ───────────────
    @testset "telemetry_agent_middleware success" begin
        backend = InMemoryTelemetryBackend()
        mw = telemetry_agent_middleware(backend)

        client = MockTelemetryClient()
        agent = Agent(name = "TestAgent", client = client)
        ctx = AgentContext(
            agent = agent,
            messages = [Message(role = :user, contents = [text_content("hi")])],
        )

        mw(ctx, c -> begin
            c.result = AgentResponse(
                messages = [Message(:assistant, "hello")],
                finish_reason = STOP,
                model_id = "test-model",
                usage_details = UsageDetails(input_tokens = 10, output_tokens = 5),
            )
            nothing
        end)

        spans = get_spans(backend)
        @test length(spans) == 1
        @test spans[1].name == "agent.run"
        @test spans[1].kind == :internal
        @test spans[1].status == :ok
        @test spans[1].end_time !== nothing
        @test spans[1].attributes[GenAIConventions.AGENT_NAME] == "TestAgent"
        @test spans[1].attributes[GenAIConventions.OPERATION_NAME] == "chat"
        @test spans[1].attributes["message_count"] == 1
        @test spans[1].attributes[GenAIConventions.RESPONSE_MODEL] == "test-model"
        @test spans[1].attributes[GenAIConventions.RESPONSE_FINISH_REASONS] == "STOP"
        @test spans[1].attributes[GenAIConventions.USAGE_INPUT_TOKENS] == 10
        @test spans[1].attributes[GenAIConventions.USAGE_OUTPUT_TOKENS] == 5
    end

    # ── 10. telemetry_agent_middleware — records error span on exception ──────
    @testset "telemetry_agent_middleware error" begin
        backend = InMemoryTelemetryBackend()
        mw = telemetry_agent_middleware(backend)

        client = MockTelemetryClient()
        agent = Agent(name = "ErrorAgent", client = client)
        ctx = AgentContext(
            agent = agent,
            messages = [Message(role = :user, contents = [text_content("hi")])],
        )

        @test_throws ErrorException mw(ctx, c -> error("test failure"))

        spans = get_spans(backend)
        @test length(spans) == 1
        @test spans[1].name == "agent.run"
        @test spans[1].status == :error
        @test spans[1].end_time !== nothing
        @test length(spans[1].events) == 1
        @test spans[1].events[1]["name"] == "exception"
        @test spans[1].events[1]["attributes"]["type"] == "ErrorException"
        @test contains(spans[1].events[1]["attributes"]["message"], "test failure")
    end

    # ── 11. telemetry_chat_middleware — records span with model info ──────────
    @testset "telemetry_chat_middleware with model info" begin
        backend = InMemoryTelemetryBackend()
        mw = telemetry_chat_middleware(backend)

        tool = FunctionTool(name = "calc", description = "Calculate", func = identity)
        ctx = ChatContext(
            messages = [Message(role = :user, contents = [text_content("hello")])],
            options = ChatOptions(
                model = "gpt-4",
                temperature = 0.7,
                max_tokens = 100,
                tools = [tool],
            ),
        )

        mw(ctx, c -> nothing)

        spans = get_spans(backend)
        @test length(spans) == 1
        @test spans[1].name == "chat.completion"
        @test spans[1].kind == :client
        @test spans[1].status == :ok
        @test spans[1].attributes[GenAIConventions.REQUEST_MODEL] == "gpt-4"
        @test spans[1].attributes[GenAIConventions.REQUEST_TEMPERATURE] == 0.7
        @test spans[1].attributes[GenAIConventions.REQUEST_MAX_TOKENS] == 100
        @test spans[1].attributes["gen_ai.request.tool_count"] == 1
        @test spans[1].attributes["message_count"] == 1
    end

    # ── 12. telemetry_function_middleware — records tool invocation span ──────
    @testset "telemetry_function_middleware" begin
        backend = InMemoryTelemetryBackend()
        mw = telemetry_function_middleware(backend)

        tool = FunctionTool(name = "get_weather", description = "Get weather", func = identity)
        ctx = FunctionInvocationContext(
            tool = tool,
            arguments = Dict{String, Any}("city" => "London"),
            call_id = "call_abc123",
        )

        mw(ctx, c -> "sunny")

        spans = get_spans(backend)
        @test length(spans) == 1
        @test spans[1].name == "tool.invoke"
        @test spans[1].kind == :internal
        @test spans[1].status == :ok
        @test spans[1].attributes[GenAIConventions.TOOL_NAME] == "get_weather"
        @test spans[1].attributes[GenAIConventions.TOOL_CALL_ID] == "call_abc123"
    end

    # ── 13. instrument! adds middleware to agent ─────────────────────────────
    @testset "instrument! adds middleware" begin
        backend = InMemoryTelemetryBackend()
        client = MockTelemetryClient()
        agent = Agent(name = "InstrumentTest", client = client)

        @test isempty(agent.agent_middlewares)
        @test isempty(agent.chat_middlewares)
        @test isempty(agent.function_middlewares)

        result = instrument!(agent, backend)
        @test result === agent
        @test length(agent.agent_middlewares) == 1
        @test length(agent.chat_middlewares) == 1
        @test length(agent.function_middlewares) == 1

        # Instrument again — should prepend additional middleware
        instrument!(agent, backend)
        @test length(agent.agent_middlewares) == 2
        @test length(agent.chat_middlewares) == 2
        @test length(agent.function_middlewares) == 2
    end

    # ── 14. GenAIConventions constants exist ──────────────────────────────────
    @testset "GenAIConventions constants" begin
        @test GenAIConventions.SYSTEM == "gen_ai.system"
        @test GenAIConventions.OPERATION_NAME == "gen_ai.operation.name"
        @test GenAIConventions.REQUEST_MODEL == "gen_ai.request.model"
        @test GenAIConventions.RESPONSE_MODEL == "gen_ai.response.model"
        @test GenAIConventions.REQUEST_TEMPERATURE == "gen_ai.request.temperature"
        @test GenAIConventions.REQUEST_TOP_P == "gen_ai.request.top_p"
        @test GenAIConventions.REQUEST_MAX_TOKENS == "gen_ai.request.max_tokens"
        @test GenAIConventions.RESPONSE_FINISH_REASONS == "gen_ai.response.finish_reasons"
        @test GenAIConventions.USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
        @test GenAIConventions.USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
        @test GenAIConventions.USAGE_TOTAL_TOKENS == "gen_ai.usage.total_tokens"
        @test GenAIConventions.AGENT_NAME == "gen_ai.agent.name"
        @test GenAIConventions.AGENT_ID == "gen_ai.agent.id"
        @test GenAIConventions.TOOL_NAME == "gen_ai.tool.name"
        @test GenAIConventions.TOOL_CALL_ID == "gen_ai.tool.call_id"
    end

    # ── 15. Thread safety — concurrent span recording ────────────────────────
    @testset "Thread safety concurrent recording" begin
        backend = InMemoryTelemetryBackend()
        n = 100

        tasks = map(1:n) do i
            Threads.@spawn begin
                span = TelemetrySpan(name = "concurrent.$i")
                finish_span!(span)
                record_span!(backend, span)
            end
        end

        for t in tasks
            wait(t)
        end

        spans = get_spans(backend)
        @test length(spans) == n
        names = Set(s.name for s in spans)
        @test length(names) == n
    end

    # ── Integration: instrument! + run_agent ─────────────────────────────────
    @testset "instrument! integration with run_agent" begin
        backend = InMemoryTelemetryBackend()
        client = MockTelemetryClient()
        agent = Agent(name = "IntegrationAgent", client = client)
        instrument!(agent, backend)

        response = run_agent(agent, "Hello!")
        @test response.text == "mock response"

        spans = get_spans(backend)
        # Should have at least agent + chat spans
        @test length(spans) >= 2
        span_names = [s.name for s in spans]
        @test "agent.run" in span_names
        @test "chat.completion" in span_names
        @test all(s -> s.status == :ok, spans)
    end
end
