# Integration tests for AgentFramework.jl with Ollama
# Run with: AGENTFRAMEWORK_INTEGRATION=true julia --project=. test/runtests.jl
# Or directly: julia --project=. test/test_ollama_integration.jl

using AgentFramework
using Test
import HTTP, JSON3

# Find first available model
function get_test_model()
    model = get(ENV, "AGENTFRAMEWORK_TEST_MODEL", "")
    if !isempty(model)
        return model
    end
    # Try to detect available models via curl (more reliable than HTTP.jl with Ollama)
    try
        output = read(`curl -s --max-time 5 http://localhost:11434/api/tags`, String)
        data = JSON3.read(output, Dict{String, Any})
        models = get(data, "models", Any[])
        model_names = String[]
        for m in models
            m_dict = m isa Dict ? m : Dict{String, Any}(string(k) => v for (k, v) in pairs(m))
            push!(model_names, string(get(m_dict, "name", "")))
        end
        # Prefer fast, tool-capable models
        for preferred in ["gemma3:latest", "qwen3:8b", "phi4-mini:latest", "llama3.2:3b", "mistral:latest", "llama3:8b"]
            if preferred in model_names
                return preferred
            end
        end
        if !isempty(model_names)
            return first(model_names)
        end
    catch e
        @warn "Failed to detect Ollama models" exception=e
    end
    return "gemma3:latest"
end

# Models known NOT to support tool calling
const TOOL_UNSUPPORTED_MODELS = Set(["llama3:8b", "llama3:latest", "llama3.2:3b", "gemma3:latest"])

const TEST_MODEL = get_test_model()

@testset "Ollama Integration (model=$TEST_MODEL)" begin
    client = OllamaChatClient(model=TEST_MODEL)

    @testset "Basic chat" begin
        response = get_response(client, [Message(:user, "Say exactly: HELLO WORLD")], ChatOptions())
        @test !isempty(response.text)
        @test length(response.messages) >= 1
        @test response.messages[1].role == :assistant
        @info "Basic chat response: $(first(response.text, 100))"
    end

    @testset "Chat with system message" begin
        messages = [
            Message(:system, "You are a calculator. Only respond with numbers."),
            Message(:user, "What is 2 + 2?"),
        ]
        response = get_response(client, messages, ChatOptions(temperature=0.0))
        @test !isempty(response.text)
        @info "Calculator response: $(response.text)"
    end

    @testset "Streaming chat" begin
        channel = get_response_streaming(client, [Message(:user, "Count from 1 to 5")], ChatOptions())
        updates = ChatResponseUpdate[]
        for update in channel
            push!(updates, update)
        end
        @test !isempty(updates)
        response = ChatResponse(updates)
        @test !isempty(response.text)
        @info "Streaming response: $(first(response.text, 100))"
    end

    @testset "Agent basic run" begin
        agent = Agent(
            name = "TestAgent",
            instructions = "You are a helpful assistant. Keep responses brief.",
            client = client,
            options = ChatOptions(temperature=0.0),
        )
        response = run_agent(agent, "What is the capital of France? Answer in one word.")
        @test !isempty(response.text)
        @info "Agent response: $(first(response.text, 100))"
    end

    @testset "Agent with tools" begin
        if TEST_MODEL in TOOL_UNSUPPORTED_MODELS
            @info "Skipping tool test: $TEST_MODEL does not support tools"
            @test_skip true
        else
            add_tool = FunctionTool(
            name = "add_numbers",
            description = "Add two numbers together and return the result",
            func = (a, b) -> a + b,
            parameters = Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "a" => Dict{String, Any}("type" => "number", "description" => "First number"),
                    "b" => Dict{String, Any}("type" => "number", "description" => "Second number"),
                ),
                "required" => ["a", "b"],
            ),
        )

        agent = Agent(
            name = "MathAgent",
            instructions = "You are a math assistant. Use the add_numbers tool to compute additions. Respond with just the numeric result.",
            client = client,
            tools = [add_tool],
            options = ChatOptions(temperature=0.0),
        )

        response = run_agent(agent, "What is 17 + 25?")
        @test !isempty(response.text)
        @info "Tool agent response: $(first(response.text, 200))"
        end
    end

    @testset "Agent streaming run" begin
        agent = Agent(
            name = "StreamAgent",
            instructions = "You are helpful. Keep responses brief.",
            client = client,
            options = ChatOptions(temperature=0.0),
        )

        stream = run_agent_streaming(agent, "Say hello in exactly 3 words.")
        collected = String[]
        for update in stream
            txt = get_text(update)
            if !isempty(txt)
                push!(collected, txt)
            end
        end
        full_text = join(collected)
        @test !isempty(full_text)
        @info "Streaming agent response: $full_text"
    end

    @testset "Agent with session continuity" begin
        history = InMemoryHistoryProvider()
        agent = Agent(
            name = "SessionAgent",
            instructions = "You are helpful. Keep responses brief.",
            client = client,
            context_providers = [history],
            options = ChatOptions(temperature=0.0),
        )
        session = AgentSession(id="integration-test")

        # First turn
        r1 = run_agent(agent, "My name is Alice."; session=session)
        @test !isempty(r1.text)

        # Second turn — should have context from first
        r2 = run_agent(agent, "What is my name?"; session=session)
        @test !isempty(r2.text)
        @info "Session continuity response: $(r2.text[1:min(200, length(r2.text))])"
    end

    @testset "Agent with middleware" begin
        call_log = String[]

        mw = function(ctx::AgentContext, next)
            push!(call_log, "before")
            result = next(ctx)
            push!(call_log, "after")
            return result
        end

        agent = Agent(
            client = client,
            agent_middlewares = [mw],
            options = ChatOptions(temperature=0.0),
        )

        response = run_agent(agent, "Hi")
        @test !isempty(response.text)
        @test call_log == ["before", "after"]
    end
end
