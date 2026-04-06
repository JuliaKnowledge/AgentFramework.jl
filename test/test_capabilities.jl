using AgentFramework
using Test

@testset "Capabilities" begin
    # ── Default capabilities (unknown client) ────────────────────────────────
    @testset "Default capabilities" begin
        struct _TestBareChatClient <: AbstractChatClient end
        client = _TestBareChatClient()
        @test !supports_embeddings(client)
        @test !supports_image_generation(client)
        @test !supports_code_interpreter(client)
        @test !supports_file_search(client)
        @test !supports_web_search(client)
        @test !supports_streaming(client)
        @test !supports_structured_output(client)
        @test !supports_tool_calling(client)
        @test isempty(list_capabilities(client))
    end

    # ── OllamaChatClient capabilities ────────────────────────────────────────
    @testset "OllamaChatClient capabilities" begin
        client = OllamaChatClient(model="test", base_url="http://localhost:11434")
        @test supports_streaming(client)
        @test supports_tool_calling(client)
        @test supports_embeddings(client)
        @test !supports_image_generation(client)
        @test !supports_code_interpreter(client)
        @test !supports_file_search(client)
        @test !supports_web_search(client)
        @test !supports_structured_output(client)
    end

    # ── OpenAIChatClient capabilities ────────────────────────────────────────
    @testset "OpenAIChatClient capabilities" begin
        client = OpenAIChatClient(model="gpt-4o", api_key="test-key")
        @test supports_streaming(client)
        @test supports_tool_calling(client)
        @test supports_structured_output(client)
        @test supports_embeddings(client)
        @test supports_image_generation(client)
        @test !supports_code_interpreter(client)
        @test !supports_file_search(client)
        @test !supports_web_search(client)
    end

    # ── AzureOpenAIChatClient capabilities ───────────────────────────────────
    @testset "AzureOpenAIChatClient capabilities" begin
        client = AzureOpenAIChatClient(model="gpt-4o", endpoint="https://test.openai.azure.com", api_key="test-key")
        @test supports_streaming(client)
        @test supports_tool_calling(client)
        @test supports_structured_output(client)
        @test supports_embeddings(client)
        @test supports_image_generation(client)
        @test !supports_code_interpreter(client)
        @test !supports_file_search(client)
        @test !supports_web_search(client)
    end

    @testset "FoundryChatClient capabilities" begin
        client = FoundryChatClient(
            model = "gpt-5.3",
            project_endpoint = "https://acct.services.ai.azure.com/api/projects/_project",
            token_provider = () -> "token",
        )
        @test supports_streaming(client)
        @test supports_tool_calling(client)
        @test supports_structured_output(client)
        @test !supports_embeddings(client)
        @test !supports_image_generation(client)
        @test !supports_code_interpreter(client)
        @test !supports_file_search(client)
        @test !supports_web_search(client)
    end

    # ── has_capability dispatch ──────────────────────────────────────────────
    @testset "has_capability dispatch" begin
        ollama = OllamaChatClient(model="test", base_url="http://localhost:11434")
        @test has_capability(ollama, streaming_capability)
        @test has_capability(ollama, tool_calling_capability)
        @test has_capability(ollama, embedding_capability)
        @test !has_capability(ollama, image_generation_capability)

        openai = OpenAIChatClient(model="gpt-4o", api_key="test-key")
        @test has_capability(openai, streaming_capability)
        @test has_capability(openai, embedding_capability)
        @test !has_capability(openai, web_search_capability)
    end

    # ── list_capabilities ────────────────────────────────────────────────────
    @testset "list_capabilities" begin
        ollama = OllamaChatClient(model="test", base_url="http://localhost:11434")
        caps = list_capabilities(ollama)
        @test :streaming in caps
        @test :tool_calling in caps
        @test :embeddings in caps
        @test length(caps) == 3

        openai = OpenAIChatClient(model="gpt-4o", api_key="test-key")
        caps = list_capabilities(openai)
        @test :streaming in caps
        @test :tool_calling in caps
        @test :structured_output in caps
        @test :embeddings in caps
        @test :image_generation in caps
        @test length(caps) == 5
    end

    # ── require_capability ───────────────────────────────────────────────────
    @testset "require_capability throws for missing" begin
        ollama = OllamaChatClient(model="test", base_url="http://localhost:11434")
        @test_throws ChatClientError require_capability(ollama, image_generation_capability, "image_generation")

        err = try
            require_capability(ollama, image_generation_capability, "image_generation")
        catch e
            e
        end
        @test occursin("does not support image_generation", err.message)
    end

    @testset "require_capability passes for present" begin
        ollama = OllamaChatClient(model="test", base_url="http://localhost:11434")
        @test require_capability(ollama, streaming_capability, "streaming") === nothing

        openai = OpenAIChatClient(model="gpt-4o", api_key="test-key")
        @test require_capability(openai, embedding_capability, "embeddings") === nothing
    end

    # ── Custom client capabilities ───────────────────────────────────────────
    @testset "Custom client capabilities" begin
        struct _TestCustomClient <: AbstractChatClient end
        AgentFramework.web_search_capability(::Type{_TestCustomClient}) = HasWebSearch()
        AgentFramework.streaming_capability(::Type{_TestCustomClient}) = HasStreaming()

        client = _TestCustomClient()
        @test supports_web_search(client)
        @test supports_streaming(client)
        @test !supports_embeddings(client)
        @test !supports_tool_calling(client)

        caps = list_capabilities(client)
        @test :web_search in caps
        @test :streaming in caps
        @test length(caps) == 2
    end

    # ── Trait type hierarchy ─────────────────────────────────────────────────
    @testset "Trait type hierarchy" begin
        @test HasEmbeddings() isa Capability
        @test HasImageGeneration() isa Capability
        @test HasCodeInterpreter() isa Capability
        @test HasFileSearch() isa Capability
        @test HasWebSearch() isa Capability
        @test HasStreaming() isa Capability
        @test HasStructuredOutput() isa Capability
        @test HasToolCalling() isa Capability
        @test !(NoCapability() isa Capability)
    end
end
