#!/usr/bin/env julia
#
# Integration test suite for AgentFramework.jl with Ollama backend + Mem0.jl memory
# Run: julia --project=. test/integration_ollama_mem0.jl
#

using Test
using JSON3
using HTTP
using Base64
using Random

# Load both packages
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "Mem0.jl"))
using AgentFramework
using Mem0

const MODEL = "qwen3:8b"
const EMBED_MODEL = "nomic-embed-text"

println("=" ^ 70)
println("Integration Tests: AgentFramework.jl + Ollama + Mem0.jl")
println("Model: $MODEL | Embeddings: $EMBED_MODEL")
println("=" ^ 70)
println()

# ─── Helper: extract text, stripping <think>...</think> blocks ─────────────
function clean_response(text::String)
    cleaned = replace(text, r"<think>.*?</think>"s => "")
    return strip(cleaned)
end

function clean_response(response::AgentResponse)
    return clean_response(get_text(response))
end

# ─── 1. Basic Agent Chat ────────────────────────────────────────────────────

@testset "1. Basic Ollama Agent" begin
    client = OllamaChatClient(model=MODEL)
    agent = Agent(
        name = "basic_agent",
        client = client,
        instructions = "You are a helpful assistant. Be concise. Answer in one sentence. Do NOT wrap your answer in <think> tags.",
    )

    @testset "Simple question" begin
        response = run_agent(agent, "What is 2 + 2? Just give the number.")
        text = clean_response(response)
        println("  Q: What is 2+2?")
        println("  A: $text")
        @test !isempty(text)
        @test occursin("4", text)
    end

    @testset "Follow-up with session" begin
        history = InMemoryHistoryProvider()
        agent_with_history = Agent(
            name = "history_agent",
            client = client,
            instructions = "You are a helpful assistant. Be concise. Answer in one sentence.",
            context_providers = [history],
        )
        session = AgentSession()
        r1 = run_agent(agent_with_history, "My name is Julia. Remember that."; session=session)
        t1 = clean_response(r1)
        println("  Q: My name is Julia.")
        println("  A: $t1")

        r2 = run_agent(agent_with_history, "What is my name?"; session=session)
        t2 = clean_response(r2)
        println("  Q: What is my name?")
        println("  A: $t2")
        @test occursin("julia", lowercase(t2))
    end

    @testset "System instructions respected" begin
        pirate = Agent(
            name = "pirate",
            client = client,
            instructions = "You are a pirate. Every response must contain 'arr' or 'matey'. Be very brief.",
        )
        response = run_agent(pirate, "Hello there")
        text = clean_response(response)
        println("  Pirate: $text")
        @test occursin("arr", lowercase(text)) || occursin("matey", lowercase(text)) ||
              occursin("ahoy", lowercase(text)) || occursin("ye", lowercase(text))
    end
end

# ─── 2. Tool Calling ────────────────────────────────────────────────────────

@testset "2. Tool Calling" begin
    weather_calls = Ref(0)
    calc_calls = Ref(0)

    @tool function get_weather(city::String)
        "Get the current weather for a city"
        weather_calls[] += 1
        return "Sunny, 22°C in $city"
    end

    @tool function calculator(expression::String)
        "Evaluate a simple math expression like '2 + 3' or '10 * 5'"
        calc_calls[] += 1
        result = eval(Meta.parse(expression))
        return string(result)
    end

    client = OllamaChatClient(model=MODEL)

    @testset "Single tool call" begin
        agent = Agent(
            name = "weather_agent",
            client = client,
            instructions = "You are a weather assistant. Use the get_weather tool to answer weather questions. Be concise.",
            tools = [get_weather],
        )
        response = run_agent(agent, "What's the weather in London?")
        text = clean_response(response)
        println("  Q: Weather in London?")
        println("  A: $text")
        println("  Tool calls made: $(weather_calls[])")
        @test weather_calls[] >= 1
        @test occursin("London", text) || occursin("22", text) || occursin("unny", text)
    end

    @testset "Multiple tools available" begin
        agent = Agent(
            name = "multi_tool_agent",
            client = client,
            instructions = "You help with weather and math. Use get_weather for weather, calculator for math. Be concise.",
            tools = [get_weather, calculator],
        )
        weather_calls[] = 0
        calc_calls[] = 0

        response = run_agent(agent, "What's the weather in Paris?")
        text = clean_response(response)
        println("  Multi-tool weather: $text (calls: weather=$(weather_calls[]) calc=$(calc_calls[]))")
        @test weather_calls[] >= 1

        response2 = run_agent(agent, "What is 7 * 8?")
        text2 = clean_response(response2)
        println("  Multi-tool math: $text2 (calls: weather=$(weather_calls[]) calc=$(calc_calls[]))")
        @test calc_calls[] >= 1
        @test occursin("56", text2)
    end

    @testset "Tool lifecycle (max_invocations)" begin
        limited = FunctionTool(
            name = "limited_tool",
            description = "A tool that can only be called twice",
            func = (x) -> "result",
            parameters = Dict{String,Any}("type" => "object",
                "properties" => Dict{String,Any}("x" => Dict{String,Any}("type" => "string")),
                "required" => ["x"]),
            max_invocations = 2,
        )
        @test invoke_tool(limited, Dict{String,Any}("x" => "a")) == "result"
        @test invoke_tool(limited, Dict{String,Any}("x" => "b")) == "result"
        @test_throws ToolExecutionError invoke_tool(limited, Dict{String,Any}("x" => "c"))
        reset_invocation_count!(limited)
        @test invoke_tool(limited, Dict{String,Any}("x" => "d")) == "result"
        println("  Tool lifecycle (max_invocations=2): ✓")
    end
end

# ─── 3. Streaming ───────────────────────────────────────────────────────────

@testset "3. Streaming" begin
    client = OllamaChatClient(model=MODEL)
    agent = Agent(
        name = "streaming_agent",
        client = client,
        instructions = "You are helpful. Be concise, one sentence max.",
    )

    @testset "Stream collects full response" begin
        chunks = String[]
        stream = run_agent_streaming(agent, "Say hello in French.")
        for update in stream
            txt = get_text(update)
            if !isempty(txt)
                push!(chunks, txt)
            end
        end
        full_text = join(chunks)
        cleaned = clean_response(full_text)
        println("  Streaming: $(length(chunks)) chunks")
        println("  Full: $cleaned")
        @test !isempty(cleaned)
        @test occursin("bonjour", lowercase(cleaned)) || occursin("salut", lowercase(cleaned)) ||
              length(cleaned) > 5
    end

    @testset "Stream with tool calls" begin
        @tool function greet_in(language::String)
            "Greet the user in the specified language"
            return language == "Spanish" ? "¡Hola!" : "Hello!"
        end

        tool_agent = Agent(
            name = "stream_tool_agent",
            client = client,
            instructions = "Use the greet_in tool to greet users. Be concise.",
            tools = [greet_in],
        )

        chunks = String[]
        stream = run_agent_streaming(tool_agent, "Greet me in Spanish")
        for update in stream
            txt = get_text(update)
            !isempty(txt) && push!(chunks, txt)
        end
        full = clean_response(join(chunks))
        println("  Stream+tools: $full")
        @test !isempty(full)
    end
end

# ─── 4. Structured Output ──────────────────────────────────────────────────

@testset "4. Structured Output" begin
    client = OllamaChatClient(model=MODEL)
    agent = Agent(
        name = "structured_agent",
        client = client,
        instructions = "You extract structured data. Return ONLY valid JSON, no markdown fences, no explanation.",
    )

    @testset "JSON extraction" begin
        response = run_agent(agent,
            """Extract the person's name and age from this text as JSON with keys "name" and "age":
            "John Smith is 42 years old and lives in Seattle."
            Return ONLY the JSON object, nothing else.""")
        text = clean_response(response)
        println("  Structured: $text")
        # Try to parse as JSON, stripping fences if needed
        stripped = replace(text, r"```json\s*" => "")
        stripped = replace(stripped, r"```\s*" => "")
        stripped = strip(stripped)
        data = JSON3.read(stripped, Dict{String,Any})
        @test haskey(data, "name")
        @test haskey(data, "age")
        @test occursin("John", string(data["name"]))
    end

    @testset "Schema-based structured output" begin
        Base.@kwdef struct CityInfo
            name::String
            country::String
            population::Int
        end

        schema = schema_from_type(CityInfo)
        @test haskey(schema, "properties")
        @test haskey(schema["properties"], "name")
        println("  Schema for CityInfo: $(JSON3.write(schema))")

        format = response_format_for(CityInfo)
        response = run_agent(agent,
            "Provide info about Tokyo as JSON with keys: name, country, population (integer)";
            options=ChatOptions(response_format=format))
        text = clean_response(response)
        println("  Schema response: $text")
        result = parse_structured(CityInfo, text)
        @test result.value.name isa String
        @test !isempty(result.value.name)
        println("  Parsed: $(result.value)")
    end
end

# ─── 5. Multi-agent Handoff ─────────────────────────────────────────────────

@testset "5. Multi-agent" begin
    client = OllamaChatClient(model=MODEL)

    @testset "as_tool delegation" begin
        math_agent = Agent(
            name = "math_expert",
            client = client,
            instructions = "You are a math expert. Solve math problems. Be concise, just give the answer.",
        )

        main_agent = Agent(
            name = "coordinator",
            client = client,
            instructions = "You coordinate tasks. Use the math_expert tool for math questions. Be concise.",
            tools = [as_tool(math_agent; description="Ask the math expert to solve a math problem")],
        )

        response = run_agent(main_agent, "What is the square root of 144?")
        text = clean_response(response)
        println("  Multi-agent Q: sqrt(144)?")
        println("  Multi-agent A: $text")
        @test occursin("12", text)
    end

    @testset "HandoffTool" begin
        billing = Agent(
            name = "billing_agent",
            client = client,
            instructions = "You are the billing department. Answer billing questions. Be concise.",
        )

        handoff = HandoffTool(
            name = "transfer_to_billing",
            description = "Transfer to billing department for payment questions",
            target = billing,
        )

        support = Agent(
            name = "support_agent",
            client = client,
            instructions = "You are customer support. For billing questions, use transfer_to_billing. Be concise.",
            tools = [handoff],
        )

        response = run_agent(support, "I have a question about my invoice")
        text = clean_response(response)
        println("  Handoff Q: billing question")
        println("  Handoff A: $text")
        @test !isempty(text)
    end
end

# ─── 6. Middleware Pipeline ─────────────────────────────────────────────────

@testset "6. Middleware" begin
    client = OllamaChatClient(model=MODEL)

    @testset "Chat middleware intercepts" begin
        log = String[]

        function logging_middleware(ctx::ChatContext, next::Function)
            push!(log, "before:$(length(ctx.messages))msgs")
            result = next(ctx)
            push!(log, "after:response")
            return result
        end

        agent = Agent(
            name = "middleware_test",
            client = client,
            instructions = "Be concise.",
            chat_middlewares = [logging_middleware],
        )

        response = run_agent(agent, "Say hi")
        text = clean_response(response)
        println("  Middleware log: $log")
        println("  Response: $text")
        @test length(log) >= 2
        @test startswith(log[1], "before:")
        @test log[end] == "after:response"
    end

    @testset "Terminate pipeline" begin
        function block_middleware(ctx::ChatContext, next::Function)
            terminate_pipeline("Blocked by middleware")
        end

        agent = Agent(
            name = "blocked_agent",
            client = client,
            instructions = "You won't see this.",
            chat_middlewares = [block_middleware],
        )

        # The pipeline should terminate gracefully
        response = run_agent(agent, "Try to chat")
        text = get_text(response)
        println("  Terminated: '$text'")
        @test occursin("Blocked", text) || isempty(text) || text == "Blocked by middleware"
    end
end

# ─── 7. Session State & Context ─────────────────────────────────────────────

@testset "7. Sessions & Context" begin
    client = OllamaChatClient(model=MODEL)

    @testset "Session state persists" begin
        session = AgentSession(id="test_session_state")
        session.state["counter"] = 0

        agent = Agent(
            name = "stateful_agent",
            client = client,
            instructions = "Just say OK.",
        )

        run_agent(agent, "First"; session=session)
        session.state["counter"] += 1
        run_agent(agent, "Second"; session=session)
        session.state["counter"] += 1

        @test session.state["counter"] == 2
        @test session.id == "test_session_state"
        println("  Session state counter: $(session.state["counter"]) ✓")
    end

    @testset "InMemoryHistoryProvider" begin
        history = InMemoryHistoryProvider()
        agent = Agent(
            name = "history_agent",
            client = client,
            instructions = "Be concise.",
            context_providers = [history],
        )
        session = AgentSession()

        r1 = run_agent(agent, "My favorite number is 42."; session=session)
        r2 = run_agent(agent, "What is my favorite number?"; session=session)
        text = clean_response(r2)
        println("  History recall: $text")
        @test occursin("42", text)
    end
end

# ─── 8. Mem0.jl Memory Integration ──────────────────────────────────────────

@testset "8. Mem0.jl Integration" begin
    config = Mem0.MemoryConfig(
        llm = Mem0.LlmConfig(
            provider = "ollama",
            config = Dict{String,Any}("model" => MODEL, "temperature" => 0.1, "max_tokens" => 1000),
        ),
        embedder = Mem0.EmbedderConfig(
            provider = "ollama",
            config = Dict{String,Any}("model" => EMBED_MODEL, "embedding_dims" => 768),
        ),
        vector_store = Mem0.VectorStoreConfig(
            provider = "in_memory",
            config = Dict{String,Any}(
                "collection_name" => "integration_test",
                "embedding_model_dims" => 768,
            ),
        ),
        version = "v1.1",
    )
    mem = Mem0.Memory(config = config)

    @testset "Add and search memories" begin
        println("  Adding memories...")
        r1 = Mem0.add(mem, "I love programming in Julia"; user_id="test_user")
        println("  Added: $(length(r1["results"])) memories from statement 1")

        r2 = Mem0.add(mem, "My favorite color is blue"; user_id="test_user")
        println("  Added: $(length(r2["results"])) memories from statement 2")

        r3 = Mem0.add(mem, "I work at a university as a researcher"; user_id="test_user")
        println("  Added: $(length(r3["results"])) memories from statement 3")

        results = Mem0.search(mem, "What programming language?"; user_id="test_user")
        search_items = results["results"]
        println("  Search results: $(length(search_items))")
        for (i, r) in enumerate(search_items)
            println("    [$i] $(r["memory"]) (score: $(round(r["score"], digits=3)))")
        end
        @test length(search_items) >= 1

        all_result = Mem0.get_all(mem; user_id="test_user")
        all_items = all_result["results"]
        println("  Total memories: $(length(all_items))")
        @test length(all_items) >= 1
    end

    @testset "Memory-augmented agent" begin
        client = OllamaChatClient(model=MODEL)

        Mem0.add(mem, "The user's name is Dr. Simon Frost"; user_id="agent_user")
        Mem0.add(mem, "The user studies infectious disease modeling"; user_id="agent_user")

        agent = Agent(
            name = "memory_agent",
            client = client,
            instructions = "You are a helpful personal assistant. Use the context about the user to personalize responses. Be concise.",
        )

        all_result = Mem0.get_all(mem; user_id="agent_user")
        memories = all_result["results"]
        memory_text = if !isempty(memories)
            facts = join([m["memory"] for m in memories], "\n- ")
            "Known facts about the user:\n- $facts"
        else
            ""
        end

        session = AgentSession()
        messages = Message[]
        !isempty(memory_text) && push!(messages, Message(role=:system, contents=[text_content(memory_text)]))
        push!(messages, Message(role=:user, contents=[text_content("What do you know about me?")]))

        response = run_agent(agent, messages; session=session)
        text = clean_response(response)
        println("  Memory-agent: $text")
        @test !isempty(text)
        has_ctx = occursin("Simon", text) || occursin("Frost", text) ||
                  occursin("infectious", lowercase(text)) || occursin("disease", lowercase(text))
        @test has_ctx
    end
end

# ─── 9. Mem0.jl + Neo4j Graph Memory ────────────────────────────────────────

@testset "9. Mem0 + Neo4j Graph Memory" begin
    neo4j_ok = try
        resp = HTTP.post("http://127.0.0.1:7474/db/neo4j/tx/commit",
            ["Content-Type" => "application/json",
             "Authorization" => "Basic " * Base64.base64encode("neo4j:password")],
            JSON3.write(Dict("statements" => [Dict("statement" => "RETURN 1")]));
            status_exception=false)
        resp.status in 200:299
    catch
        false
    end

    if !neo4j_ok
        @info "Neo4j not available — skipping graph memory tests"
        @test_skip "Neo4j unavailable"
    else
        using Random

        config = Mem0.MemoryConfig(
            llm = Mem0.LlmConfig(
                provider = "ollama",
                config = Dict{String,Any}("model" => MODEL, "temperature" => 0.1, "max_tokens" => 1000),
            ),
            embedder = Mem0.EmbedderConfig(
                provider = "ollama",
                config = Dict{String,Any}("model" => EMBED_MODEL, "embedding_dims" => 768),
            ),
            vector_store = Mem0.VectorStoreConfig(
                provider = "in_memory",
                config = Dict{String,Any}(
                    "collection_name" => "neo4j_integ_test",
                    "embedding_model_dims" => 768,
                ),
            ),
            graph_store = Mem0.GraphStoreConfig(
                provider = "neo4j",
                config = Dict{String,Any}(
                    "url" => "http://127.0.0.1:7474",
                    "username" => "neo4j",
                    "password" => "password",
                    "database" => "neo4j",
                ),
                threshold = 0.7,
            ),
            version = "v1.1",
        )
        mem = Mem0.Memory(config = config)
        test_uid = "neo4j_integ_$(randstring(6))"

        @testset "Graph memory add & search" begin
            println("  Adding memories with graph...")
            r1 = Mem0.add(mem, "Alice works at Microsoft Research on AI safety."; user_id=test_uid)
            println("  Added: $(length(r1["results"])) vector memories, relations: $(get(r1, "relations", []))")

            r2 = Mem0.add(mem, "Bob is Alice's colleague and works on machine learning."; user_id=test_uid)
            println("  Added: $(length(r2["results"])) vector memories")

            println("  Searching...")
            results = Mem0.search(mem, "Who works at Microsoft?"; user_id=test_uid)
            search_items = results["results"]
            println("  Search: $(length(search_items)) results")
            for (i, r) in enumerate(search_items)
                println("    [$i] $(r["memory"]) (score: $(round(r["score"], digits=3)))")
            end
            @test length(search_items) >= 1

            # Check graph relations
            graph_rels = Mem0.get_all_graph(mem.graph, Dict{String,Any}("user_id" => test_uid))
            println("  Graph relations: $(length(graph_rels))")
            for r in graph_rels
                src = r["source"]; rel = r["relationship"]; dst = r["destination"]
                println("    $src --[$rel]--> $dst")
            end
            @test length(graph_rels) >= 1
        end

        # Cleanup
        try
            Mem0.delete_all_graph!(mem.graph, Dict{String,Any}("user_id" => test_uid))
        catch e
            @warn "Neo4j cleanup failed" exception=e
        end
    end
end

# ─── 10. Workflow with Ollama ───────────────────────────────────────────────

@testset "10. Workflow Integration" begin
    client = OllamaChatClient(model=MODEL)

    @testset "Sequential workflow" begin
        researcher = Agent(
            name = "researcher",
            client = client,
            instructions = "You research topics. Given a topic, provide 3 key facts as bullet points. Be concise.",
        )

        summarizer = Agent(
            name = "summarizer",
            client = client,
            instructions = "You summarize text. Given research notes, write a one-paragraph summary.",
        )

        wf = WorkflowBuilder(
            name = "research_pipeline",
            start = agent_executor("research", researcher),
        )
        add_executor(wf, agent_executor("summarize", summarizer))
        add_edge(wf, "research", "summarize")
        add_output(wf, "summarize")
        workflow = build(wf)

        result = run_workflow(workflow, "Tell me about the Julia programming language")
        outputs = get_outputs(result)
        println("  Workflow state: $(result.state)")
        if !isempty(outputs)
            text = clean_response(string(first(outputs)))
            println("  Workflow output (first 200 chars): $(first(text, 200))")
            @test !isempty(text)
        else
            println("  Workflow produced $(length(result.events)) events")
            @test !isempty(result.events)
        end
    end
end

# ─── 11. Resilience ────────────────────────────────────────────────────────

@testset "11. Resilience" begin
    client = OllamaChatClient(model=MODEL)

    @testset "with_retry! middleware" begin
        agent = Agent(
            name = "resilient_agent",
            client = client,
            instructions = "Say 'pong' when I say 'ping'. Nothing else.",
        )
        with_retry!(agent)

        response = run_agent(agent, "ping")
        text = clean_response(response)
        println("  Resilience: $text")
        @test occursin("pong", lowercase(text))
    end
end

# ─── 12. Edge Cases & Error Handling ────────────────────────────────────────

@testset "12. Edge Cases" begin
    client = OllamaChatClient(model=MODEL)

    @testset "Empty input" begin
        agent = Agent(name="empty_test", client=client, instructions="Say hello if no question is asked.")
        response = run_agent(agent, "")
        text = get_text(response)
        @test text isa String
        println("  Empty input response: $(first(clean_response(text), 80))")
    end

    @testset "Long input" begin
        agent = Agent(name="long_test", client=client, instructions="Summarize in one sentence.")
        long_text = "Julia is a programming language. " ^ 50
        response = run_agent(agent, long_text)
        text = clean_response(response)
        @test !isempty(text)
        println("  Long input summary: $(first(text, 100))")
    end

    @testset "Multiple messages input" begin
        agent = Agent(name="multi_msg", client=client, instructions="Be helpful and concise.")
        msgs = [
            Message(role=:user, contents=[text_content("My name is Bob")]),
            Message(role=:assistant, contents=[text_content("Nice to meet you, Bob!")]),
            Message(role=:user, contents=[text_content("What's my name?")]),
        ]
        response = run_agent(agent, msgs)
        text = clean_response(response)
        println("  Multi-message: $text")
        @test occursin("Bob", text) || occursin("bob", lowercase(text))
    end
end

println()
println("=" ^ 70)
println("Integration tests complete!")
println("=" ^ 70)
