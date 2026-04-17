#!/usr/bin/env julia
# =============================================================================
# Workflow checkpoint & resume
# =============================================================================
#
# Long-running workflows need to survive process restarts.  AF's
# `FileCheckpointStorage` persists one checkpoint per iteration to disk;
# resuming is a one-liner: pass `checkpoint_id` + the same `builder` to
# `run_workflow` and it picks up from where it left off.
#
# This vignette models a 3-stage ingest pipeline:
#
#     ingest (tokenise) → analyse (uppercase + reverse) → summarise
#
# Run 1: go end-to-end, inspect the checkpoint history on disk.
# Run 2: simulate a crash by rebuilding the workflow in a fresh process-like
#        scope and resuming from an earlier checkpoint — the remaining
#        stages complete and produce the same final output.
#
# Fully offline — no Ollama needed.
# -----------------------------------------------------------------------------

using Test
using AgentFramework

# ── Define the three stages ──────────────────────────────────────────────────

tokenise(msg, ctx) = send_message(ctx, split(string(msg)))

function analyse(msg, ctx)
    tokens = msg isa Vector ? msg : split(string(msg))
    transformed = [reverse(uppercase(t)) for t in tokens]
    send_message(ctx, transformed)
end

summarise(msg, ctx) = yield_output(ctx, "summary:" * join(msg, "|"))

function build_pipeline(storage)
    ing = ExecutorSpec(id="ingest",    handler=tokenise)
    ana = ExecutorSpec(id="analyse",   handler=analyse)
    sum_ = ExecutorSpec(id="summarise", handler=summarise)

    b = WorkflowBuilder(
        name = "ingest_pipeline",
        start = ing,
        checkpoint_storage = storage,
    )
    add_executor(b, ana)
    add_executor(b, sum_)
    add_edge(b, "ingest",   "analyse")
    add_edge(b, "analyse",  "summarise")
    add_output(b, "summarise")
    return b
end

# ── Run 1: full run, persisting every iteration to disk ──────────────────────

checkpoint_dir = mktempdir()
println("Checkpoint dir: ", checkpoint_dir)

storage = FileCheckpointStorage(checkpoint_dir)
builder1 = build_pipeline(storage)
wf1 = build(builder1)

result1 = run_workflow(wf1, "hello brave new world")
final1 = get_outputs(result1)
println("Full-run output: ", final1)

cps = AgentFramework.list_checkpoints(storage, "ingest_pipeline")
sort!(cps, by = cp -> cp.iteration)
println("Persisted checkpoints: ", length(cps))
for cp in cps
    println("  iteration=$(cp.iteration)  id=$(cp.id)  msgs-keys=$(collect(keys(cp.messages)))")
end

# ── Run 2: simulated crash + resume from the earliest checkpoint ─────────────
#
# Pretend the process just died.  In a fresh scope we rebuild the
# workflow from scratch (same builder shape → same graph signature) and
# resume from checkpoint #1 — which sits after `ingest` has run but
# before `analyse` has done its work.

early_cp = cps[1]          # iteration == 1, after `ingest`
println("\nResuming from checkpoint at iteration $(early_cp.iteration)")

# Re-hydrate storage from disk (proves persistence is real).
storage2 = FileCheckpointStorage(checkpoint_dir)
builder2 = build_pipeline(storage2)
wf2 = build(builder2)

result2 = run_workflow(wf2; checkpoint_id = early_cp.id, checkpoint_storage = storage2)
final2 = get_outputs(result2)
println("Resumed output: ", final2)

# ── Assertions ───────────────────────────────────────────────────────────────

@testset "Workflow checkpoint + resume" begin
    @test !isempty(final1)
    @test startswith(final1[1], "summary:")
    @test occursin("OLLEH", final1[1])        # reverse(uppercase("hello"))
    @test occursin("DLROW", final1[1])

    @test length(cps) >= 2                    # at least ingest + analyse
    @test all(cp.graph_signature_hash == wf1.graph_signature_hash for cp in cps)

    @test final2 == final1                    # resumed run matches full run

    # On-disk persistence: the checkpoint files are real.
    @test length(readdir(checkpoint_dir)) >= length(cps)
end

println("\nPASS — workflow checkpoint + resume vignette")
