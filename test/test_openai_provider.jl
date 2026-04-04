@testset "OpenAI Provider" begin

    # ── OpenAIChatClient Construction ────────────────────────────────────────

    @testset "OpenAIChatClient defaults" begin
        client = OpenAIChatClient(api_key="sk-test-key")
        @test client.model == "gpt-4o"
        @test client.base_url == "https://api.openai.com/v1"
        @test client.api_key == "sk-test-key"
        @test client.organization === nothing
        @test client.read_timeout == 120
        @test isempty(client.options)
    end

    @testset "OpenAIChatClient custom fields" begin
        client = OpenAIChatClient(
            model="gpt-4o-mini",
            api_key="sk-custom",
            base_url="http://localhost:8080/v1",
            organization="org-123",
            read_timeout=60,
            options=Dict{String, Any}("seed" => 42),
        )
        @test client.model == "gpt-4o-mini"
        @test client.base_url == "http://localhost:8080/v1"
        @test client.organization == "org-123"
        @test client.read_timeout == 60
        @test client.options["seed"] == 42
    end

    @testset "OpenAIChatClient env var fallback" begin
        old_key = get(ENV, "OPENAI_API_KEY", nothing)
        try
            ENV["OPENAI_API_KEY"] = "sk-from-env"
            client = OpenAIChatClient()
            # api_key field is empty, but _resolve_api_key should find ENV
            @test client.api_key == ""
            key = AgentFramework._resolve_api_key(client)
            @test key == "sk-from-env"
        finally
            if old_key === nothing
                delete!(ENV, "OPENAI_API_KEY")
            else
                ENV["OPENAI_API_KEY"] = old_key
            end
        end
    end

    @testset "OpenAIChatClient explicit key overrides env" begin
        old_key = get(ENV, "OPENAI_API_KEY", nothing)
        try
            ENV["OPENAI_API_KEY"] = "sk-from-env"
            client = OpenAIChatClient(api_key="sk-explicit")
            key = AgentFramework._resolve_api_key(client)
            @test key == "sk-explicit"
        finally
            if old_key === nothing
                delete!(ENV, "OPENAI_API_KEY")
            else
                ENV["OPENAI_API_KEY"] = old_key
            end
        end
    end

    @testset "OpenAIChatClient missing key throws" begin
        old_key = get(ENV, "OPENAI_API_KEY", nothing)
        try
            delete!(ENV, "OPENAI_API_KEY")
            client = OpenAIChatClient()
            @test_throws ChatClientError AgentFramework._resolve_api_key(client)
        finally
            if old_key !== nothing
                ENV["OPENAI_API_KEY"] = old_key
            end
        end
    end

    # ── AzureOpenAIChatClient Construction ───────────────────────────────────

    @testset "AzureOpenAIChatClient defaults" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com",
            api_key="azure-test-key",
        )
        @test client.model == "gpt-4o"
        @test client.endpoint == "https://myresource.openai.azure.com"
        @test client.api_key == "azure-test-key"
        @test client.credential === nothing
        @test client.token_provider === nothing
        @test client.token_scope == AgentFramework.DEFAULT_AZURE_OPENAI_TOKEN_SCOPE
        @test client.api_version == "2024-06-01"
        @test client.read_timeout == 120
        @test isempty(client.options)
    end

    @testset "AzureOpenAIChatClient custom api_version" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com",
            api_key="key",
            api_version="2024-10-01-preview",
        )
        @test client.api_version == "2024-10-01-preview"
    end

    @testset "AzureOpenAIChatClient env var fallback" begin
        old_key = get(ENV, "AZURE_OPENAI_API_KEY", nothing)
        try
            ENV["AZURE_OPENAI_API_KEY"] = "azure-from-env"
            client = AzureOpenAIChatClient(
                model="gpt-4o",
                endpoint="https://myresource.openai.azure.com",
            )
            @test client.api_key == ""
            key = AgentFramework._resolve_api_key(client)
            @test key == "azure-from-env"
        finally
            if old_key === nothing
                delete!(ENV, "AZURE_OPENAI_API_KEY")
            else
                ENV["AZURE_OPENAI_API_KEY"] = old_key
            end
        end
    end

    @testset "AzureOpenAIChatClient missing key throws" begin
        old_key = get(ENV, "AZURE_OPENAI_API_KEY", nothing)
        try
            delete!(ENV, "AZURE_OPENAI_API_KEY")
            client = AzureOpenAIChatClient(
                model="gpt-4o",
                endpoint="https://myresource.openai.azure.com",
            )
            @test_throws ChatClientError AgentFramework._resolve_api_key(client)
        finally
            if old_key !== nothing
                ENV["AZURE_OPENAI_API_KEY"] = old_key
            end
        end
    end

    # ── URL Construction ─────────────────────────────────────────────────────

    @testset "OpenAI URL construction" begin
        client = OpenAIChatClient(api_key="sk-test")
        url = AgentFramework._chat_completions_url(client)
        @test url == "https://api.openai.com/v1/chat/completions"
    end

    @testset "OpenAI URL with custom base_url" begin
        client = OpenAIChatClient(api_key="sk-test", base_url="http://localhost:8080/v1")
        url = AgentFramework._chat_completions_url(client)
        @test url == "http://localhost:8080/v1/chat/completions"
    end

    @testset "OpenAI URL strips trailing slash" begin
        client = OpenAIChatClient(api_key="sk-test", base_url="https://api.openai.com/v1/")
        url = AgentFramework._chat_completions_url(client)
        @test url == "https://api.openai.com/v1/chat/completions"
    end

    @testset "Azure URL construction" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com",
            api_key="key",
        )
        url = AgentFramework._chat_completions_url(client)
        @test url == "https://myresource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-06-01"
    end

    @testset "Azure URL with custom api_version" begin
        client = AzureOpenAIChatClient(
            model="my-deployment",
            endpoint="https://myresource.openai.azure.com",
            api_key="key",
            api_version="2024-10-01-preview",
        )
        url = AgentFramework._chat_completions_url(client)
        @test url == "https://myresource.openai.azure.com/openai/deployments/my-deployment/chat/completions?api-version=2024-10-01-preview"
    end

    @testset "Azure URL strips trailing slash from endpoint" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com/",
            api_key="key",
        )
        url = AgentFramework._chat_completions_url(client)
        @test url == "https://myresource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-06-01"
    end

    # ── Header Construction ──────────────────────────────────────────────────

    @testset "OpenAI headers - Bearer token" begin
        client = OpenAIChatClient(api_key="sk-test-key-123")
        headers = AgentFramework._build_headers(client)
        header_dict = Dict(headers)
        @test header_dict["Authorization"] == "Bearer sk-test-key-123"
        @test header_dict["Content-Type"] == "application/json"
        @test header_dict["Connection"] == "close"
        @test !haskey(header_dict, "OpenAI-Organization")
    end

    @testset "OpenAI headers - with organization" begin
        client = OpenAIChatClient(api_key="sk-test", organization="org-abc")
        headers = AgentFramework._build_headers(client)
        header_dict = Dict(headers)
        @test header_dict["OpenAI-Organization"] == "org-abc"
    end

    @testset "Azure headers - api-key" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com",
            api_key="azure-key-456",
        )
        headers = AgentFramework._build_headers(client)
        header_dict = Dict(headers)
        @test header_dict["api-key"] == "azure-key-456"
        @test header_dict["Content-Type"] == "application/json"
        @test header_dict["Connection"] == "close"
        @test !haskey(header_dict, "Authorization")
    end

    @testset "Azure headers - token provider" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com",
            token_provider=() -> "provider-token",
        )
        headers = AgentFramework._build_headers(client)
        header_dict = Dict(headers)
        @test header_dict["Authorization"] == "Bearer provider-token"
        @test !haskey(header_dict, "api-key")
    end

    # ── Curl Header Construction ─────────────────────────────────────────────

    @testset "OpenAI curl headers" begin
        client = OpenAIChatClient(api_key="sk-curl-test")
        curl_headers = AgentFramework._build_curl_headers(client)
        @test "-H" in curl_headers
        @test "Content-Type: application/json" in curl_headers
        @test "Authorization: Bearer sk-curl-test" in curl_headers
    end

    @testset "OpenAI curl headers with organization" begin
        client = OpenAIChatClient(api_key="sk-curl-test", organization="org-xyz")
        curl_headers = AgentFramework._build_curl_headers(client)
        @test "OpenAI-Organization: org-xyz" in curl_headers
    end

    @testset "Azure curl headers" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com",
            api_key="azure-curl-key",
        )
        curl_headers = AgentFramework._build_curl_headers(client)
        @test "api-key: azure-curl-key" in curl_headers
        @test "Content-Type: application/json" in curl_headers
    end

    @testset "Azure curl headers with token provider" begin
        client = AzureOpenAIChatClient(
            model="gpt-4o",
            endpoint="https://myresource.openai.azure.com",
            token_provider=() -> "provider-token",
        )
        curl_headers = AgentFramework._build_curl_headers(client)
        @test "Authorization: Bearer provider-token" in curl_headers
        @test !("api-key: provider-token" in curl_headers)
    end

    # ── Show Method ──────────────────────────────────────────────────────────

    @testset "show methods" begin
        client1 = OpenAIChatClient(api_key="sk-test")
        @test sprint(show, client1) == "OpenAIChatClient(\"gpt-4o\")"

        client2 = AzureOpenAIChatClient(
            model="my-deploy",
            endpoint="https://myresource.openai.azure.com",
            api_key="key",
        )
        @test sprint(show, client2) == "AzureOpenAIChatClient(\"my-deploy\")"
    end

    # ── Request Body Construction ────────────────────────────────────────────

    @testset "request body includes model and messages" begin
        client = OpenAIChatClient(api_key="sk-test", model="gpt-4o-mini")
        msgs = [Message(role=:user, contents=[text_content("Hello")])]
        opts = ChatOptions()
        body = AgentFramework._build_request_body(client, msgs, opts; stream=false)
        @test body["model"] == "gpt-4o-mini"
        @test body["stream"] == false
        @test length(body["messages"]) == 1
        @test body["messages"][1]["role"] == "user"
        @test body["messages"][1]["content"] == "Hello"
    end

    @testset "request body respects ChatOptions model override" begin
        client = OpenAIChatClient(api_key="sk-test", model="gpt-4o")
        msgs = [Message(role=:user, contents=[text_content("Hi")])]
        opts = ChatOptions(model="gpt-4o-mini")
        body = AgentFramework._build_request_body(client, msgs, opts; stream=false)
        @test body["model"] == "gpt-4o-mini"
    end

    @testset "request body includes client options" begin
        client = OpenAIChatClient(
            api_key="sk-test",
            options=Dict{String, Any}("seed" => 42, "frequency_penalty" => 0.5),
        )
        msgs = [Message(role=:user, contents=[text_content("Hi")])]
        opts = ChatOptions()
        body = AgentFramework._build_request_body(client, msgs, opts; stream=false)
        @test body["seed"] == 42
        @test body["frequency_penalty"] == 0.5
    end

    @testset "request body includes optional ChatOptions fields" begin
        client = OpenAIChatClient(api_key="sk-test")
        msgs = [Message(role=:user, contents=[text_content("Hi")])]
        opts = ChatOptions(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["END"],
        )
        body = AgentFramework._build_request_body(client, msgs, opts; stream=false)
        @test body["temperature"] == 0.7
        @test body["top_p"] == 0.9
        @test body["max_tokens"] == 100
        @test body["stop"] == ["END"]
    end

    # ── Exports ──────────────────────────────────────────────────────────────

    @testset "types are exported" begin
        @test isdefined(AgentFramework, :OpenAIChatClient)
        @test isdefined(AgentFramework, :AzureOpenAIChatClient)
        @test OpenAIChatClient <: AbstractChatClient
        @test AzureOpenAIChatClient <: AbstractChatClient
    end
end
