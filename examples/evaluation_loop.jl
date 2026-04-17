#!/usr/bin/env julia
# =============================================================================
# Evaluation loop: grading an agent across a small query set
# =============================================================================
#
# AgentFramework's evaluation harness (`evaluate_agent`) runs one or more
# queries against an agent and grades each response with a vector of
# `EvalCheck` functions.  Three shipped checks cover most of what you
# need out of the box:
#
#   * `keyword_check("foo", "bar")` — all listed keywords must appear.
#   * `tool_called_check("tool_name")` — the expected tool must be invoked.
#   * custom checks — any `(; query, response, expected_output) -> (passed, ...)`
#
# This vignette builds a tiny "faq-bot" agent backed by Ollama with one
# tool, evaluates it over four realistic queries, mixes a built-in check
# with a custom lambda check, and prints a pass/fail report.
#
# Requires OLLAMA_HOST reachable + qwen3:8b.
# -----------------------------------------------------------------------------

using Test
using AgentFramework

ollama_host = get(ENV, "OLLAMA_HOST", "http://localhost:11434")
client = OllamaChatClient(model="qwen3:8b", base_url=ollama_host)

# ── Agent under test ─────────────────────────────────────────────────────────

const CALC_CALLS = Ref(0)
function calc(expr::AbstractString)
    CALC_CALLS[] += 1
    try
        val = eval(Meta.parse(String(expr)))
        return "Result: $val"
    catch e
        return "ERROR: $(sprint(showerror, e))"
    end
end

calc_tool = FunctionTool(
    name = "calc",
    description = "Evaluate a simple Julia numeric expression and return the result.",
    func = calc,
    parameters = Dict{String,Any}(
        "type" => "object",
        "properties" => Dict{String,Any}(
            "expr" => Dict{String,Any}("type"=>"string","description"=>"A numeric expression like '2+2' or 'sqrt(144)'."),
        ),
        "required" => ["expr"],
    ),
)

agent = Agent(
    name = "faq_bot",
    client = client,
    instructions = """
        You are a friendly FAQ bot.  When the user asks something
        numeric, call the calc tool.  For text questions, answer from
        general knowledge in ONE short sentence.  Always be concise.
        """,
    tools = [calc_tool],
)

# ── Custom check: answer must be short (≤ 35 words) ──────────────────────────

function length_check(item::AgentFramework.EvalItem)::AgentFramework.CheckResult
    text = AgentFramework.eval_response(item)
    n_words = length(split(text))
    passed = n_words <= 35
    return AgentFramework.CheckResult(
        passed = passed,
        reason = "length=$(n_words) (limit=35)",
        check_name = "length_check",
    )
end

# ── Eval suite ───────────────────────────────────────────────────────────────

queries = [
    "What is 144 / 12?",
    "What is sqrt(169)?",
    "In one sentence, what is the capital of France?",
    "What planet is known as the Red Planet?",
]
expected_keywords = [
    ["12"],
    ["13"],
    ["paris"],
    ["mars"],
]

evaluators = [
    LocalEvaluator(keyword_check(expected_keywords[1]...)),
    LocalEvaluator(keyword_check(expected_keywords[2]...)),
    LocalEvaluator(keyword_check(expected_keywords[3]...)),
    LocalEvaluator(keyword_check(expected_keywords[4]...)),
]

println("\nRunning $(length(queries)) evals against the faq_bot…")

all_results = AgentFramework.EvalResults[]
for (i, q) in enumerate(queries)
    results = evaluate_agent(
        agent = agent,
        queries = [q],
        evaluators = evaluators[i],
        eval_name = "keyword-$(i)",
    )
    append!(all_results, results)
    r = results[1]
    pass = eval_passed(r)
    tot  = eval_total(r)
    println("  [$i] pass=$pass/$tot  Q: $q")
    for item in r.items
        println("       → status=$(item.status)  response=\"$(first(strip(item.output_text), 80))…\"")
    end
end

# Separate custom length-check pass
println("\nLength-check pass (custom lambda, ≤35 words each)…")
length_results = evaluate_agent(
    agent = agent,
    queries = queries,
    evaluators = LocalEvaluator(length_check),
    eval_name = "length",
)
println("  pass=$(eval_passed(length_results[1]))/$(eval_total(length_results[1]))")

# ── Assertions ───────────────────────────────────────────────────────────────

@testset "Evaluation loop" begin
    # At least 3 of 4 keyword evals should pass (qwen3 occasionally elaborates).
    keyword_pass_count = sum(eval_passed(r) == eval_total(r) for r in all_results)
    @test keyword_pass_count >= 3

    # The calc tool should have been called for the two numeric queries.
    @test CALC_CALLS[] >= 2

    # Length check passes at least 3/4.
    @test eval_passed(length_results[1]) >= 3
end

println("\nCalc tool invocations: ", CALC_CALLS[])
println("PASS — evaluation loop vignette")
