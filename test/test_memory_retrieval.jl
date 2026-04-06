using AgentFramework
using SQLite
using Test

@testset "Memory Retrieval" begin
    @testset "MemoryRecord procedural metadata is searchable" begin
        record = MemoryRecord(
            scope = "user-1",
            kind = :subtask,
            role = :assistant,
            content = "Use the spreadsheet filter before you build the pivot table.",
            metadata = Dict{String, Any}(
                "task" => "prepare quarterly revenue report",
                "subtask" => "filter revenue rows",
                "agent_name" => "excel",
                "reflection" => "Filtering first reduces downstream tool calls",
                "tags" => ["excel", "reporting"],
            ),
        )

        text = AgentFramework._memory_search_text(record)
        @test occursin("subtask", text)
        @test occursin("quarterly revenue report", text)
        @test occursin("excel", text)
    end

    @testset "InMemoryMemoryStore search and scope isolation" begin
        store = InMemoryMemoryStore()
        add_memories!(store, [
            MemoryRecord(
                scope = "user-1",
                kind = :procedural,
                role = :assistant,
                content = "Draft the email first, then attach the spreadsheet.",
                metadata = Dict{String, Any}("task" => "send budget update", "tags" => ["email", "budget"]),
            ),
            MemoryRecord(
                scope = "user-2",
                kind = :procedural,
                role = :assistant,
                content = "Always verify calendar conflicts before confirming the meeting.",
                metadata = Dict{String, Any}("task" => "schedule leadership review", "tags" => ["calendar"]),
            ),
        ])

        results = search_memories(store, "budget email"; scope="user-1", limit=3)
        @test length(results) == 1
        @test results[1].record.scope == "user-1"
        @test occursin("attach the spreadsheet", results[1].record.content)

        @test isempty(search_memories(store, "budget email"; scope="user-2", limit=3))
    end

    @testset "FileMemoryStore persists procedural memories" begin
        dir = mktempdir()
        try
            store = FileMemoryStore(directory=dir)
            add_memories!(store, [
                MemoryRecord(
                    scope = "user-1",
                    kind = :subtask,
                    role = :assistant,
                    content = "Open the workbook, filter on the current quarter, then export the chart.",
                    metadata = Dict{String, Any}("subtask" => "quarterly chart export", "tags" => ["excel", "chart"]),
                ),
            ])

            reopened = FileMemoryStore(directory=dir)
            results = search_memories(reopened, "quarter chart"; scope="user-1", limit=3)
            @test length(results) == 1
            @test results[1].record.kind == :subtask
            @test results[1].record.metadata["subtask"] == "quarterly chart export"
        finally
            rm(dir; recursive=true, force=true)
        end
    end

    @testset "SQLiteMemoryStore supports full-text search" begin
        db_path = tempname() * ".sqlite"
        store = SQLiteMemoryStore(db_path)
        try
            add_memories!(store, [
                MemoryRecord(
                    scope = "user-1",
                    kind = :full_task,
                    role = :assistant,
                    content = "Plan the budget review as: gather spreadsheet data, summarize changes, then email finance.",
                    metadata = Dict{String, Any}("plan" => "gather -> summarize -> email", "tags" => ["budget", "finance"]),
                ),
                MemoryRecord(
                    scope = "user-1",
                    kind = :subtask,
                    role = :assistant,
                    content = "Use conditional formatting to highlight negative deltas.",
                    metadata = Dict{String, Any}("subtask" => "format negative deltas", "tools" => ["excel"]),
                ),
            ])

            full_task = search_memories(store, "finance budget"; scope="user-1", limit=5)
            @test length(full_task) == 1
            @test full_task[1].record.kind == :full_task

            subtask = search_memories(store, "negative deltas excel"; scope="user-1", limit=5)
            @test length(subtask) == 1
            @test subtask[1].record.kind == :subtask

            @test length(get_memories(store; scope="user-1")) == 2
            clear_memories!(store; scope="user-1")
            @test isempty(get_memories(store; scope="user-1"))
        finally
            rm(db_path; force=true)
        end
    end

    @testset "MemoryContextProvider shares memory by user scope" begin
        store = InMemoryMemoryStore()
        provider = MemoryContextProvider(store=store, source_id="memory_ctx", max_results=3)

        session1 = AgentSession(id="session-1", user_id="shared-user")
        ctx1 = SessionContext(
            session_id=session1.id,
            input_messages=[Message(:user, "How should I filter the spreadsheet before I send the update?")],
        )
        ctx1.response = (messages=[Message(:assistant, "Filter the spreadsheet first, then draft the update email.")],)
        state1 = Dict{String, Any}()
        after_run!(provider, nothing, session1, ctx1, state1)

        session2 = AgentSession(id="session-2", user_id="shared-user")
        ctx2 = SessionContext(
            session_id=session2.id,
            input_messages=[Message(:user, "What was the spreadsheet-and-email workflow again?")],
        )
        state2 = Dict{String, Any}()
        before_run!(provider, nothing, session2, ctx2, state2)

        @test haskey(ctx2.context_messages, "memory_ctx")
        injected = only(ctx2.context_messages["memory_ctx"])
        @test occursin("draft the update email", injected.text)
        @test state2["last_result_count"] >= 1
    end

    @testset "RDFMemoryStore availability stays explicit" begin
        result = try
            RDFMemoryStore()
        catch ex
            ex
        end
        @test result isa RDFMemoryStore || result isa AgentError
        if result isa AgentError
            @test occursin("RDFLib.jl", result.message)
        end
    end
end
