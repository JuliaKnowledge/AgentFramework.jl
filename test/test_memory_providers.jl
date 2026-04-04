using AgentFramework
using Test
using Dates

@testset "Memory Providers" begin

    # ── FileHistoryProvider ──────────────────────────────────────────────────

    @testset "FileHistoryProvider construction" begin
        provider = FileHistoryProvider(directory="/tmp/test_history")
        @test provider.source_id == "file_history"
        @test provider.directory == "/tmp/test_history"
        @test provider.max_messages == 100
    end

    @testset "FileHistoryProvider show" begin
        provider = FileHistoryProvider(directory="/tmp/test_dir")
        io = IOBuffer()
        show(io, provider)
        @test String(take!(io)) == "FileHistoryProvider(\"/tmp/test_dir\")"
    end

    @testset "FileHistoryProvider read/write cycle" begin
        dir = mktempdir()
        try
            provider = FileHistoryProvider(directory=dir, max_messages=50)

            # Initially empty
            msgs = get_messages(provider, "session-1")
            @test isempty(msgs)

            # Save messages
            save_messages!(provider, "session-1", [
                Message(:user, "Hello"),
                Message(:assistant, "Hi there!"),
            ])

            # Read them back
            msgs = get_messages(provider, "session-1")
            @test length(msgs) == 2
            @test msgs[1].role == :user
            @test msgs[1].text == "Hello"
            @test msgs[2].role == :assistant
            @test msgs[2].text == "Hi there!"

            # Verify file exists
            @test isfile(joinpath(dir, "session-1.json"))
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "FileHistoryProvider append messages" begin
        dir = mktempdir()
        try
            provider = FileHistoryProvider(directory=dir)

            save_messages!(provider, "s1", [Message(:user, "first")])
            save_messages!(provider, "s1", [Message(:assistant, "response")])
            save_messages!(provider, "s1", [Message(:user, "second")])

            msgs = get_messages(provider, "s1")
            @test length(msgs) == 3
            @test msgs[1].text == "first"
            @test msgs[2].text == "response"
            @test msgs[3].text == "second"
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "FileHistoryProvider max_messages truncation" begin
        dir = mktempdir()
        try
            provider = FileHistoryProvider(directory=dir, max_messages=3)

            # Save 5 messages in batches
            save_messages!(provider, "s1", [
                Message(:user, "msg1"),
                Message(:assistant, "msg2"),
            ])
            save_messages!(provider, "s1", [
                Message(:user, "msg3"),
                Message(:assistant, "msg4"),
                Message(:user, "msg5"),
            ])

            msgs = get_messages(provider, "s1")
            @test length(msgs) == 3
            @test msgs[1].text == "msg3"
            @test msgs[2].text == "msg4"
            @test msgs[3].text == "msg5"
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "FileHistoryProvider session isolation" begin
        dir = mktempdir()
        try
            provider = FileHistoryProvider(directory=dir)

            save_messages!(provider, "session-a", [Message(:user, "from A")])
            save_messages!(provider, "session-b", [Message(:user, "from B")])

            @test length(get_messages(provider, "session-a")) == 1
            @test length(get_messages(provider, "session-b")) == 1
            @test get_messages(provider, "session-a")[1].text == "from A"
            @test get_messages(provider, "session-b")[1].text == "from B"
            @test isempty(get_messages(provider, "session-c"))
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "FileHistoryProvider creates directory" begin
        dir = mktempdir()
        subdir = joinpath(dir, "nested", "deep")
        try
            provider = FileHistoryProvider(directory=subdir)
            @test !isdir(subdir)

            save_messages!(provider, "s1", [Message(:user, "hello")])
            @test isdir(subdir)
            @test isfile(joinpath(subdir, "s1.json"))
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "FileHistoryProvider before_run!/after_run!" begin
        dir = mktempdir()
        try
            provider = FileHistoryProvider(directory=dir)
            session = AgentSession(id="test-session")

            # Pre-populate some history
            save_messages!(provider, session.id, [
                Message(:user, "old question"),
                Message(:assistant, "old answer"),
            ])

            # Simulate before_run!
            ctx = SessionContext(
                session_id=session.id,
                input_messages=[Message(:user, "new question")],
            )
            state = Dict{String, Any}()

            before_run!(provider, nothing, session, ctx, state)

            # History should be in context_messages
            @test haskey(ctx.context_messages, "file_history")
            @test length(ctx.context_messages["file_history"]) == 2

            # Simulate a response
            response_msg = Message(:assistant, "new answer")
            ctx.response = (messages=[response_msg],)

            after_run!(provider, nothing, session, ctx, state)

            # Now should have 4 messages total
            msgs = get_messages(provider, session.id)
            @test length(msgs) == 4
            @test msgs[3].text == "new question"
            @test msgs[4].text == "new answer"
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    # ── DBInterfaceHistoryProvider ───────────────────────────────────────────

    @testset "DBInterfaceHistoryProvider construction" begin
        provider = DBInterfaceHistoryProvider()
        @test provider.source_id == "db_history"
        @test provider.conn === nothing
        @test provider.table_name == "agent_messages"
        @test provider.max_messages == 100
        @test provider.auto_create_table == true
        @test provider._table_created == false
    end

    @testset "DBInterfaceHistoryProvider custom fields" begin
        provider = DBInterfaceHistoryProvider(
            source_id="custom_db",
            table_name="chat_history",
            max_messages=50,
            auto_create_table=false,
        )
        @test provider.source_id == "custom_db"
        @test provider.table_name == "chat_history"
        @test provider.max_messages == 50
        @test provider.auto_create_table == false
    end

    @testset "DBInterfaceHistoryProvider get_create_table_sql" begin
        provider = DBInterfaceHistoryProvider(table_name="my_messages")
        sql = get_create_table_sql(provider)
        @test occursin("CREATE TABLE IF NOT EXISTS my_messages", sql)
        @test occursin("session_id TEXT NOT NULL", sql)
        @test occursin("role TEXT NOT NULL", sql)
        @test occursin("content TEXT NOT NULL", sql)
        @test occursin("created_at TEXT NOT NULL", sql)
        @test occursin("id INTEGER PRIMARY KEY AUTOINCREMENT", sql)
    end

    @testset "DBInterfaceHistoryProvider show" begin
        provider = DBInterfaceHistoryProvider(table_name="history")
        io = IOBuffer()
        show(io, provider)
        @test String(take!(io)) == "DBInterfaceHistoryProvider(table=\"history\")"
    end

    # ── RedisHistoryProvider ─────────────────────────────────────────────────

    @testset "RedisHistoryProvider construction" begin
        provider = RedisHistoryProvider()
        @test provider.source_id == "redis_history"
        @test provider.conn === nothing
        @test provider.prefix == "agentframework:history:"
        @test provider.max_messages == 100
        @test provider.ttl == 86400
    end

    @testset "RedisHistoryProvider custom fields" begin
        provider = RedisHistoryProvider(
            source_id="custom_redis",
            prefix="myapp:history:",
            max_messages=200,
            ttl=3600,
        )
        @test provider.source_id == "custom_redis"
        @test provider.prefix == "myapp:history:"
        @test provider.max_messages == 200
        @test provider.ttl == 3600
    end

    @testset "RedisHistoryProvider show" begin
        provider = RedisHistoryProvider(prefix="test:")
        io = IOBuffer()
        show(io, provider)
        @test String(take!(io)) == "RedisHistoryProvider(prefix=\"test:\")"
    end

    @testset "RedisHistoryProvider key format" begin
        provider = RedisHistoryProvider(prefix="myprefix:")
        key = AgentFramework._redis_key(provider, "session-123")
        @test key == "myprefix:session-123"
    end

    # ── Message serialization helpers ────────────────────────────────────────

    @testset "Message serialize/deserialize roundtrip" begin
        msg = Message(:user, "Hello, world!")
        json = AgentFramework._serialize_message(msg)
        restored = AgentFramework._deserialize_message(json)
        @test restored.role == :user
        @test restored.text == "Hello, world!"
    end

    @testset "Message serialize/deserialize with metadata" begin
        msg = Message(
            role=:assistant,
            contents=[text_content("response text")],
            author_name="agent-1",
        )
        json = AgentFramework._serialize_message(msg)
        restored = AgentFramework._deserialize_message(json)
        @test restored.role == :assistant
        @test restored.text == "response text"
        @test restored.author_name == "agent-1"
    end

    # ── Type hierarchy ───────────────────────────────────────────────────────

    @testset "Type hierarchy" begin
        @test DBInterfaceHistoryProvider <: BaseHistoryProvider
        @test DBInterfaceHistoryProvider <: AbstractHistoryProvider
        @test DBInterfaceHistoryProvider <: AbstractContextProvider

        @test RedisHistoryProvider <: BaseHistoryProvider
        @test RedisHistoryProvider <: AbstractHistoryProvider
        @test RedisHistoryProvider <: AbstractContextProvider

        @test FileHistoryProvider <: BaseHistoryProvider
        @test FileHistoryProvider <: AbstractHistoryProvider
        @test FileHistoryProvider <: AbstractContextProvider
    end
end
