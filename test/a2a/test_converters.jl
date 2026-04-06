@testset "A2A converters" begin
    @testset "message_to_a2a_dict preserves parts and metadata" begin
        message = Message(
            role = :assistant,
            contents = [
                text_content("Hello"),
                data_content("YWJj", "text/plain"),
                uri_content("https://example.com/file.txt"; media_type = "text/plain"),
                hosted_file_content("hosted://storage/document.pdf"),
            ],
            additional_properties = Dict{String, Any}(
                "context_id" => "ctx-123",
                "trace_id" => "trace-456",
                "__internal" => "skip",
            ),
        )

        payload = message_to_a2a_dict(message)
        @test payload["role"] == "agent"
        @test payload["contextId"] == "ctx-123"
        @test payload["metadata"] == Dict{String, Any}("trace_id" => "trace-456")
        @test [part["kind"] for part in payload["parts"]] == ["text", "file", "file", "file"]
        @test payload["parts"][2]["file"]["bytes"] == "YWJj"
        @test payload["parts"][3]["file"]["uri"] == "https://example.com/file.txt"
        @test payload["parts"][4]["file"]["uri"] == "hosted://storage/document.pdf"
    end

    @testset "a2a_message_to_message parses data and file parts" begin
        payload = Dict{String, Any}(
            "kind" => "message",
            "role" => "agent",
            "messageId" => "msg-1",
            "contextId" => "ctx-1",
            "metadata" => Dict{String, Any}("trace_id" => "trace-1"),
            "parts" => Any[
                Dict{String, Any}("kind" => "data", "data" => Dict{String, Any}("key" => "value", "number" => 42)),
                Dict{String, Any}("kind" => "file", "file" => Dict{String, Any}("uri" => "https://example.com/file.pdf", "mimeType" => "application/pdf")),
                Dict{String, Any}("kind" => "file", "file" => Dict{String, Any}("bytes" => "YQ==", "mimeType" => "text/plain")),
            ],
        )

        message = a2a_message_to_message(payload)
        @test message.role == :assistant
        @test message.message_id == "msg-1"
        @test message.additional_properties["trace_id"] == "trace-1"
        @test message.additional_properties["context_id"] == "ctx-1"
        @test JSON3.read(get_text(message.contents[1]), Dict{String, Any}) == Dict{String, Any}("key" => "value", "number" => 42)
        @test content_type_string(message.contents[2].type) == "uri"
        @test message.contents[2].uri == "https://example.com/file.pdf"
        @test content_type_string(message.contents[3].type) == "data"
        @test message.contents[3].text == "YQ=="
    end

    @testset "a2a_task_from_dict and task_to_response parse terminal artifacts" begin
        payload = Dict{String, Any}(
            "kind" => "task",
            "id" => "task-1",
            "contextId" => "ctx-1",
            "status" => Dict{String, Any}("state" => "completed"),
            "artifacts" => Any[
                Dict{String, Any}(
                    "artifactId" => "art-1",
                    "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "First result")],
                ),
                Dict{String, Any}(
                    "artifactId" => "art-2",
                    "parts" => Any[Dict{String, Any}("kind" => "text", "text" => "Second result")],
                ),
            ],
        )

        task = a2a_task_from_dict(payload)
        response = task_to_response(task)

        @test task.id == "task-1"
        @test task.context_id == "ctx-1"
        @test task.status.state == A2A_TASK_COMPLETED
        @test response.response_id == "task-1"
        @test response.finish_reason == STOP
        @test response.continuation_token === nothing
        @test length(response.messages) == 2
        @test response.messages[1].text == "First result"
        @test response.messages[2].text == "Second result"
    end
end
