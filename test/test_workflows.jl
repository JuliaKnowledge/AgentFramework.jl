using AgentFramework
using Test

@testset "Workflows" begin

    @testset "Workflow types" begin
        @testset "WorkflowMessage" begin
            msg = WorkflowMessage(data="hello", source_id="exec1")
            @test msg.data == "hello"
            @test msg.source_id == "exec1"
            @test msg.target_id === nothing
            @test msg.type == STANDARD_MESSAGE
        end

        @testset "WorkflowEvent factories" begin
            e1 = event_started()
            @test e1.type == EVT_STARTED

            e2 = event_status(WF_IN_PROGRESS)
            @test e2.type == EVT_STATUS
            @test e2.state == WF_IN_PROGRESS

            e3 = event_output("exec1", "result")
            @test e3.type == EVT_OUTPUT
            @test e3.executor_id == "exec1"
            @test e3.data == "result"

            e4 = event_superstep_started(3)
            @test e4.iteration == 3

            e5 = event_executor_invoked("exec1")
            @test e5.executor_id == "exec1"

            # Display
            s = sprint(show, e2)
            @test contains(s, "EVT_STATUS")
        end

        @testset "WorkflowRunResult" begin
            events = [
                event_started(),
                event_output("e1", "out1"),
                event_output("e2", "out2"),
                event_status(WF_IDLE),
            ]
            result = WorkflowRunResult(events=events, state=WF_IDLE)
            @test get_outputs(result) == ["out1", "out2"]
            @test get_final_state(result) == WF_IDLE
            @test isempty(get_request_info_events(result))
        end
    end

    @testset "Edges" begin
        @testset "Direct edge" begin
            eg = direct_edge("a", "b")
            @test eg.kind == DIRECT_EDGE
            @test length(eg.edges) == 1
            @test eg.edges[1].source_id == "a"
            @test eg.edges[1].target_id == "b"

            # Routing
            msgs = [WorkflowMessage(data="hello", source_id="a")]
            routed = route_messages(eg, msgs)
            @test haskey(routed, "b")
            @test routed["b"] == ["hello"]
        end

        @testset "Direct edge with condition" begin
            eg = direct_edge("a", "b"; condition = d -> d isa String && length(d) > 3)
            msgs_pass = [WorkflowMessage(data="hello", source_id="a")]
            msgs_fail = [WorkflowMessage(data="hi", source_id="a")]

            @test !isempty(route_messages(eg, msgs_pass))
            @test isempty(route_messages(eg, msgs_fail))
        end

        @testset "Fan-out edge" begin
            eg = fan_out_edge("src", ["t1", "t2", "t3"])
            @test eg.kind == FAN_OUT_EDGE
            @test length(eg.edges) == 3

            msgs = [WorkflowMessage(data="broadcast", source_id="src")]
            routed = route_messages(eg, msgs)
            @test length(routed) == 3
            @test routed["t1"] == ["broadcast"]
            @test routed["t2"] == ["broadcast"]
            @test routed["t3"] == ["broadcast"]
        end

        @testset "Fan-out with selection" begin
            eg = fan_out_edge("src", ["fast", "slow"];
                selection_func = (data, ids) -> data == "urgent" ? ["fast"] : ["slow"]
            )

            urgent = [WorkflowMessage(data="urgent", source_id="src")]
            normal = [WorkflowMessage(data="normal", source_id="src")]

            r1 = route_messages(eg, urgent)
            @test haskey(r1, "fast") && !haskey(r1, "slow")

            r2 = route_messages(eg, normal)
            @test haskey(r2, "slow") && !haskey(r2, "fast")
        end

        @testset "Fan-in edge" begin
            eg = fan_in_edge(["s1", "s2"], "target")
            @test eg.kind == FAN_IN_EDGE

            # Both sources present → delivers
            msgs_complete = [
                WorkflowMessage(data="from_s1", source_id="s1"),
                WorkflowMessage(data="from_s2", source_id="s2"),
            ]
            routed = route_messages(eg, msgs_complete)
            @test haskey(routed, "target")
            @test length(routed["target"]) == 2

            # Only one source → doesn't deliver
            msgs_partial = [WorkflowMessage(data="from_s1", source_id="s1")]
            routed2 = route_messages(eg, msgs_partial)
            @test isempty(routed2)
        end

        @testset "Edge display" begin
            e = Edge(source_id="a", target_id="b", condition_name="is_valid")
            s = sprint(show, e)
            @test contains(s, "a")
            @test contains(s, "b")
        end
    end

    @testset "Executor" begin
        @testset "ExecutorSpec construction" begin
            spec = ExecutorSpec(
                id = "upper",
                description = "Uppercases text",
                handler = (msg, ctx) -> send_message(ctx, uppercase(msg)),
            )
            @test spec.id == "upper"
            @test contains(sprint(show, spec), "upper")
        end

        @testset "WorkflowContext messaging" begin
            ctx = WorkflowContext(executor_id="test", _state=Dict{String, Any}())

            send_message(ctx, "hello")
            send_message(ctx, "world"; target_id="specific")

            @test length(ctx._sent_messages) == 2
            @test ctx._sent_messages[1].data == "hello"
            @test ctx._sent_messages[1].target_id === nothing
            @test ctx._sent_messages[2].target_id == "specific"
        end

        @testset "WorkflowContext yield_output" begin
            ctx = WorkflowContext(executor_id="test", _state=Dict{String, Any}())

            yield_output(ctx, "result1")
            yield_output(ctx, 42)

            @test length(ctx._yielded_outputs) == 2
            @test ctx._yielded_outputs[1] == "result1"
            @test length(ctx._events) == 2
            @test ctx._events[1].type == EVT_OUTPUT
        end

        @testset "WorkflowContext state" begin
            state = Dict{String, Any}("key1" => "val1")
            ctx = WorkflowContext(executor_id="test", _state=state)

            @test get_state(ctx, "key1") == "val1"
            @test get_state(ctx, "missing", "default") == "default"

            set_state!(ctx, "key2", 42)
            @test get_state(ctx, "key2") == 42
        end

        @testset "execute_handler" begin
            spec = ExecutorSpec(
                id = "doubler",
                handler = (msg, ctx) -> begin
                    send_message(ctx, msg * 2)
                    yield_output(ctx, "processed: $(msg * 2)")
                end,
            )

            state = Dict{String, Any}()
            ctx = execute_handler(spec, 21, ["input"], state)

            @test length(ctx._sent_messages) == 1
            @test ctx._sent_messages[1].data == 42
            @test length(ctx._yielded_outputs) == 1
            @test ctx._yielded_outputs[1] == "processed: 42"
        end
    end

    @testset "Workflow Builder" begin
        @testset "Basic build" begin
            start = ExecutorSpec(id="start", handler=(msg, ctx) -> send_message(ctx, uppercase(msg)))
            finish = ExecutorSpec(id="finish", handler=(msg, ctx) -> yield_output(ctx, "done: $msg"))

            wf = build(
                add_output(
                    add_edge(
                        add_executor(
                            WorkflowBuilder(name="test", start=start),
                            finish
                        ),
                        "start", "finish"
                    ),
                    "finish"
                )
            )

            @test wf.name == "test"
            @test length(wf.executors) == 2
            @test length(wf.edge_groups) == 1
            @test wf.start_executor_id == "start"
        end

        @testset "Validation errors" begin
            start = ExecutorSpec(id="start", handler=(msg, ctx) -> nothing)

            # Edge referencing unknown executor
            builder = WorkflowBuilder(name="test", start=start)
            add_edge(builder, "start", "nonexistent")
            @test_throws WorkflowError build(builder)

            # Duplicate executor
            builder2 = WorkflowBuilder(name="test", start=start)
            @test_throws WorkflowError add_executor(builder2, ExecutorSpec(id="start", handler=(m,c)->nothing))
        end

        @testset "Fan-out builder" begin
            start = ExecutorSpec(id="src", handler=(msg, ctx) -> send_message(ctx, msg))
            t1 = ExecutorSpec(id="t1", handler=(msg, ctx) -> yield_output(ctx, "t1:$msg"))
            t2 = ExecutorSpec(id="t2", handler=(msg, ctx) -> yield_output(ctx, "t2:$msg"))

            builder = WorkflowBuilder(name="fanout", start=start)
            add_executor(builder, t1)
            add_executor(builder, t2)
            add_fan_out(builder, "src", ["t1", "t2"])
            add_output(builder, "t1")
            add_output(builder, "t2")
            wf = build(builder)

            @test length(wf.edge_groups) == 1
            @test wf.edge_groups[1].kind == FAN_OUT_EDGE
        end
    end

    @testset "Workflow Engine" begin
        @testset "Simple sequential workflow" begin
            upper = ExecutorSpec(id="upper", handler=(msg, ctx) -> send_message(ctx, uppercase(msg)))
            reverser = ExecutorSpec(id="reverse", handler=(msg, ctx) -> yield_output(ctx, reverse(msg)))

            builder = WorkflowBuilder(name="seq", start=upper)
            add_executor(builder, reverser)
            add_edge(builder, "upper", "reverse")
            add_output(builder, "reverse")
            wf = build(builder)

            result = run_workflow(wf, "hello")
            @test result.state == WF_IDLE
            outputs = get_outputs(result)
            @test length(outputs) == 1
            @test outputs[1] == "OLLEH"
        end

        @testset "Three-stage pipeline" begin
            stage1 = ExecutorSpec(id="s1", handler=(msg, ctx) -> send_message(ctx, msg * 2))
            stage2 = ExecutorSpec(id="s2", handler=(msg, ctx) -> send_message(ctx, msg + 10))
            stage3 = ExecutorSpec(id="s3", handler=(msg, ctx) -> yield_output(ctx, msg))

            builder = WorkflowBuilder(name="pipeline", start=stage1)
            add_executor(builder, stage2)
            add_executor(builder, stage3)
            add_edge(builder, "s1", "s2")
            add_edge(builder, "s2", "s3")
            add_output(builder, "s3")
            wf = build(builder)

            result = run_workflow(wf, 5)
            outputs = get_outputs(result)
            @test outputs == [20]  # (5*2) + 10 = 20
        end

        @testset "Fan-out workflow" begin
            splitter = ExecutorSpec(id="split", handler=(msg, ctx) -> send_message(ctx, msg))
            worker1 = ExecutorSpec(id="w1", handler=(msg, ctx) -> yield_output(ctx, "w1:$msg"))
            worker2 = ExecutorSpec(id="w2", handler=(msg, ctx) -> yield_output(ctx, "w2:$msg"))

            builder = WorkflowBuilder(name="fanout", start=splitter)
            add_executor(builder, worker1)
            add_executor(builder, worker2)
            add_fan_out(builder, "split", ["w1", "w2"])
            add_output(builder, "w1")
            add_output(builder, "w2")
            wf = build(builder)

            result = run_workflow(wf, "data")
            outputs = get_outputs(result)
            @test length(outputs) == 2
            @test "w1:data" in outputs
            @test "w2:data" in outputs
        end

        @testset "Conditional routing" begin
            router = ExecutorSpec(id="router", handler=(msg, ctx) -> send_message(ctx, msg))
            high = ExecutorSpec(id="high", handler=(msg, ctx) -> yield_output(ctx, "HIGH:$msg"))
            low = ExecutorSpec(id="low", handler=(msg, ctx) -> yield_output(ctx, "LOW:$msg"))

            builder = WorkflowBuilder(name="conditional", start=router)
            add_executor(builder, high)
            add_executor(builder, low)
            add_edge(builder, "router", "high"; condition = d -> d > 50)
            add_edge(builder, "router", "low"; condition = d -> d <= 50)
            add_output(builder, "high")
            add_output(builder, "low")
            wf = build(builder)

            result_high = run_workflow(wf, 75)
            @test get_outputs(result_high) == ["HIGH:75"]

            result_low = run_workflow(wf, 25)
            @test get_outputs(result_low) == ["LOW:25"]
        end

        @testset "Shared state" begin
            writer = ExecutorSpec(id="writer", handler=(msg, ctx) -> begin
                set_state!(ctx, "count", msg)
                send_message(ctx, msg)
            end)
            reader = ExecutorSpec(id="reader", handler=(msg, ctx) -> begin
                count = get_state(ctx, "count", 0)
                yield_output(ctx, "count=$count")
            end)

            builder = WorkflowBuilder(name="state", start=writer)
            add_executor(builder, reader)
            add_edge(builder, "writer", "reader")
            add_output(builder, "reader")
            wf = build(builder)

            result = run_workflow(wf, 42)
            @test get_outputs(result) == ["count=42"]
        end

        @testset "Error handling" begin
            failing = ExecutorSpec(id="fail", handler=(msg, ctx) -> error("boom!"))

            builder = WorkflowBuilder(name="error", start=failing)
            wf = build(builder)

            result = run_workflow(wf, "trigger")
            @test result.state == WF_FAILED
            failed_events = [e for e in result.events if e.type == EVT_EXECUTOR_FAILED]
            @test length(failed_events) >= 1
        end

        @testset "Max iterations guard" begin
            # Executor that always sends a message (infinite loop)
            looper = ExecutorSpec(id="loop", handler=(msg, ctx) -> send_message(ctx, msg))

            builder = WorkflowBuilder(name="loop", start=looper, max_iterations=5)
            add_edge(builder, "loop", "loop")
            wf = build(builder)

            result = run_workflow(wf, "ping")
            # Should terminate after max_iterations
            supersteps = [e for e in result.events if e.type == EVT_SUPERSTEP_COMPLETED]
            @test length(supersteps) <= 5
        end

        @testset "Streaming events" begin
            exec = ExecutorSpec(id="e", handler=(msg, ctx) -> yield_output(ctx, "result"))

            builder = WorkflowBuilder(name="stream", start=exec)
            add_output(builder, "e")
            wf = build(builder)

            channel = run_workflow(wf, "go"; stream=true)
            events = WorkflowEvent[]
            for evt in channel
                push!(events, evt)
            end
            @test !isempty(events)
            @test any(e -> e.type == EVT_OUTPUT, events)
        end

        @testset "Workflow display" begin
            exec = ExecutorSpec(id="e", handler=(msg, ctx) -> nothing)
            wf = build(WorkflowBuilder(name="test", start=exec))
            s = sprint(show, wf)
            @test contains(s, "Workflow")
            @test contains(s, "test")
        end

        @testset "Concurrent superstep execution" begin
            timestamps = Dict{String, Float64}()
            ts_lock = ReentrantLock()

            splitter = ExecutorSpec(id="split", handler=(msg, ctx) -> send_message(ctx, msg))
            w1 = ExecutorSpec(id="w1", handler=(msg, ctx) -> begin
                lock(ts_lock) do; timestamps["w1_start"] = time(); end
                sleep(0.2)
                lock(ts_lock) do; timestamps["w1_end"] = time(); end
                yield_output(ctx, "w1:$msg")
            end)
            w2 = ExecutorSpec(id="w2", handler=(msg, ctx) -> begin
                lock(ts_lock) do; timestamps["w2_start"] = time(); end
                sleep(0.2)
                lock(ts_lock) do; timestamps["w2_end"] = time(); end
                yield_output(ctx, "w2:$msg")
            end)

            builder = WorkflowBuilder(name="concurrent", start=splitter)
            add_executor(builder, w1)
            add_executor(builder, w2)
            add_fan_out(builder, "split", ["w1", "w2"])
            add_output(builder, "w1")
            add_output(builder, "w2")
            wf = build(builder)

            result = run_workflow(wf, "data")
            outputs = get_outputs(result)
            @test length(outputs) == 2
            @test "w1:data" in outputs
            @test "w2:data" in outputs

            # Verify concurrent execution: execution windows should overlap
            @test haskey(timestamps, "w1_start") && haskey(timestamps, "w2_start")
            overlap = min(timestamps["w1_end"], timestamps["w2_end"]) -
                      max(timestamps["w1_start"], timestamps["w2_start"])
            @test overlap > 0.05  # At least 50ms overlap proves concurrency
        end

        @testset "Fan-in across supersteps" begin
            # a (start) sends "from_a". Direct edge routes to b. Fan-in buffers a's contribution.
            # b fires next superstep, sends "from_b". Fan-in buffers b's contribution → all present → delivers to c.
            # c fires in a third superstep with both messages.
            a = ExecutorSpec(id="a", handler=(msg, ctx) -> send_message(ctx, "from_a"))
            b = ExecutorSpec(id="b", handler=(msg, ctx) -> send_message(ctx, "from_b"))
            c = ExecutorSpec(id="c", handler=(msg, ctx) -> yield_output(ctx, msg))

            builder = WorkflowBuilder(name="fanin_cross", start=a)
            add_executor(builder, b)
            add_executor(builder, c)
            add_edge(builder, "a", "b")
            add_fan_in(builder, ["a", "b"], "c")
            add_output(builder, "c")
            wf = build(builder)

            result = run_workflow(wf, "hello")
            outputs = get_outputs(result)

            @test length(outputs) == 2
            @test "from_a" in outputs
            @test "from_b" in outputs

            # Verify it took 3 supersteps (a, b, c each fire in separate steps)
            supersteps = [e for e in result.events if e.type == EVT_SUPERSTEP_COMPLETED]
            @test length(supersteps) == 3
        end

        @testset "HIL request → pause → response → resume" begin
            asker = ExecutorSpec(id="asker", handler=(msg, ctx) -> begin
                if msg isa Dict && haskey(msg, "answer")
                    yield_output(ctx, "Got: $(msg["answer"])")
                else
                    request_info(ctx, "What is your name?"; request_id="name_req")
                end
            end)

            builder = WorkflowBuilder(name="hil", start=asker)
            add_output(builder, "asker")
            wf = build(builder)

            # First run — should pause with a pending request
            result1 = run_workflow(wf, "start")
            @test get_final_state(result1) == WF_IDLE_WITH_PENDING_REQUESTS
            reqs = get_request_info_events(result1)
            @test length(reqs) == 1
            @test reqs[1].request_id == "name_req"

            # Resume with a response — executor receives Dict as new message
            result2 = run_workflow(wf; responses=Dict{String,Any}("name_req" => Dict("answer" => "Alice")))
            @test get_final_state(result2) == WF_IDLE
            outputs = get_outputs(result2)
            @test length(outputs) == 1
            @test outputs[1] == "Got: Alice"
        end

        @testset "Workflow checkpoints resume regular execution" begin
            upper = ExecutorSpec(id="upper", handler=(msg, ctx) -> send_message(ctx, uppercase(string(msg))))
            reverse_exec = ExecutorSpec(id="reverse", handler=(msg, ctx) -> yield_output(ctx, reverse(string(msg))))

            storage = InMemoryCheckpointStorage()
            builder = WorkflowBuilder(name="checkpoint_seq", start=upper, checkpoint_storage=storage)
            add_executor(builder, reverse_exec)
            add_edge(builder, "upper", "reverse")
            add_output(builder, "reverse")
            wf = build(builder)

            result1 = run_workflow(wf, "hello")
            @test get_final_state(result1) == WF_IDLE

            cps = AgentFramework.list_checkpoints(storage, "checkpoint_seq")
            @test length(cps) == 2
            cp1 = only([cp for cp in cps if cp.iteration == 1])
            @test haskey(cp1.messages, "reverse")
            @test cp1.graph_signature_hash == wf.graph_signature_hash

            wf_resume = build(builder)
            result2 = run_workflow(wf_resume; checkpoint_id=cp1.id, checkpoint_storage=storage)
            @test get_outputs(result2) == ["OLLEH"]
        end

        @testset "Workflow checkpoints resume pending requests" begin
            asker = ExecutorSpec(id="asker", handler=(msg, ctx) -> begin
                if msg isa Dict && haskey(msg, "answer")
                    yield_output(ctx, "Got: $(msg["answer"])")
                else
                    request_info(ctx, "What is your name?"; request_id="name_req")
                end
            end)

            storage = InMemoryCheckpointStorage()
            builder = WorkflowBuilder(name="checkpoint_hil", start=asker, checkpoint_storage=storage)
            add_output(builder, "asker")
            wf = build(builder)

            result1 = run_workflow(wf, "start")
            @test get_final_state(result1) == WF_IDLE_WITH_PENDING_REQUESTS

            checkpoint = something(AgentFramework.get_latest(storage, "checkpoint_hil"))
            @test length(checkpoint.pending_requests) == 1
            @test checkpoint.pending_requests[1].request_id == "name_req"

            wf_resume = build(builder)
            result2 = run_workflow(
                wf_resume;
                checkpoint_id=checkpoint.id,
                checkpoint_storage=storage,
                responses=Dict{String, Any}("name_req" => Dict("answer" => "Alice")),
            )
            @test get_final_state(result2) == WF_IDLE
            @test get_outputs(result2) == ["Got: Alice"]
        end

        @testset "Workflow checkpoint graph compatibility is enforced" begin
            storage = InMemoryCheckpointStorage()

            first_exec = ExecutorSpec(id="first", handler=(msg, ctx) -> send_message(ctx, string(msg)))
            second_exec = ExecutorSpec(id="second", handler=(msg, ctx) -> yield_output(ctx, string(msg)))
            builder = WorkflowBuilder(name="checkpoint_guard", start=first_exec, checkpoint_storage=storage)
            add_executor(builder, second_exec)
            add_edge(builder, "first", "second")
            add_output(builder, "second")
            wf = build(builder)
            run_workflow(wf, "hello")

            checkpoint = only([cp for cp in AgentFramework.list_checkpoints(storage, "checkpoint_guard") if cp.iteration == 1])

            passthrough = ExecutorSpec(id="first", handler=(msg, ctx) -> yield_output(ctx, string(msg)))
            incompatible_builder = WorkflowBuilder(name="checkpoint_guard", start=passthrough, checkpoint_storage=storage)
            add_output(incompatible_builder, "first")
            incompatible = build(incompatible_builder)

            @test_throws AgentFramework.WorkflowCheckpointError run_workflow(
                incompatible;
                checkpoint_id=checkpoint.id,
                checkpoint_storage=storage,
            )
        end

        @testset "Ownership token prevents concurrent runs" begin
            blocker = ExecutorSpec(id="block", handler=(msg, ctx) -> begin
                sleep(0.5)
                yield_output(ctx, "done")
            end)

            builder = WorkflowBuilder(name="owned", start=blocker)
            add_output(builder, "block")
            wf = build(builder)

            # Start first run in background
            t = @async run_workflow(wf, "go")
            sleep(0.1)  # Let it start and acquire ownership

            # Second run should be blocked
            @test_throws WorkflowError run_workflow(wf, "go2")

            # Wait for first to finish
            result = fetch(t)
            @test result.state == WF_IDLE

            # Now a new run should work
            result2 = run_workflow(wf, "go3")
            @test result2.state == WF_IDLE
        end
    end

    @testset "Checkpoints" begin
        @testset "InMemoryCheckpointStorage" begin
            storage = InMemoryCheckpointStorage()

            cp = WorkflowCheckpoint(
                workflow_name = "test_wf",
                iteration = 3,
                state = Dict{String, Any}("key" => "value"),
            )

            # Save
            id = AgentFramework.save!(storage, cp)
            @test !isempty(id)

            # Load
            loaded = AgentFramework.load(storage, id)
            @test loaded.workflow_name == "test_wf"
            @test loaded.iteration == 3
            @test loaded.state["key"] == "value"

            # List
            cps = AgentFramework.list_checkpoints(storage, "test_wf")
            @test length(cps) == 1

            # Get latest
            latest = AgentFramework.get_latest(storage, "test_wf")
            @test latest !== nothing
            @test latest.id == id

            # Not found
            @test AgentFramework.get_latest(storage, "nonexistent") === nothing
            @test_throws WorkflowError AgentFramework.load(storage, "bad-id")

            # Delete
            @test Base.delete!(storage, id) == true
            @test Base.delete!(storage, id) == false
            @test isempty(AgentFramework.list_checkpoints(storage, "test_wf"))
        end

        @testset "FileCheckpointStorage" begin
            dir = mktempdir()
            storage = FileCheckpointStorage(dir)

            cp = WorkflowCheckpoint(
                workflow_name = "file_test",
                iteration = 5,
                state = Dict{String, Any}("count" => 42, "name" => "test"),
                messages = Dict{String, Vector{WorkflowMessage}}(
                    "exec1" => [WorkflowMessage(data="hello", source_id="input")],
                ),
                pending_requests = [event_request_info("req-1", "exec1", Dict("question" => "name?"))],
                graph_signature_hash = "graph-hash",
            )

            # Save
            id = AgentFramework.save!(storage, cp)
            @test isfile(joinpath(dir, "$id.json"))

            # Load
            loaded = AgentFramework.load(storage, id)
            @test loaded.workflow_name == "file_test"
            @test loaded.iteration == 5
            @test loaded.state["count"] == 42
            @test loaded.state["name"] == "test"
            @test loaded.messages["exec1"][1].data == "hello"
            @test loaded.pending_requests[1].request_id == "req-1"
            @test loaded.pending_requests[1].data["question"] == "name?"
            @test loaded.graph_signature_hash == "graph-hash"

            # List
            cps = AgentFramework.list_checkpoints(storage, "file_test")
            @test length(cps) == 1

            # Get latest
            latest = AgentFramework.get_latest(storage, "file_test")
            @test latest !== nothing

            # Save another checkpoint
            cp2 = WorkflowCheckpoint(
                workflow_name = "file_test",
                iteration = 6,
                previous_id = id,
                state = Dict{String, Any}("count" => 43),
            )
            id2 = AgentFramework.save!(storage, cp2)
            @test length(AgentFramework.list_checkpoints(storage, "file_test")) == 2

            # Delete
            @test Base.delete!(storage, id) == true
            @test length(AgentFramework.list_checkpoints(storage, "file_test")) == 1

            # Cleanup
            rm(dir; recursive=true)
        end

        @testset "WorkflowCheckpoint display" begin
            cp = WorkflowCheckpoint(workflow_name="test", iteration=2)
            s = sprint(show, cp)
            @test contains(s, "test")
            @test contains(s, "iteration=2")
        end

        @testset "Checkpoint chaining" begin
            storage = InMemoryCheckpointStorage()

            cp1 = WorkflowCheckpoint(workflow_name="chain", iteration=1)
            id1 = AgentFramework.save!(storage, cp1)

            cp2 = WorkflowCheckpoint(workflow_name="chain", iteration=2, previous_id=id1)
            id2 = AgentFramework.save!(storage, cp2)

            loaded2 = AgentFramework.load(storage, id2)
            @test loaded2.previous_id == id1

            loaded1 = AgentFramework.load(storage, loaded2.previous_id)
            @test loaded1.iteration == 1
            @test loaded1.previous_id === nothing
        end
    end
end
