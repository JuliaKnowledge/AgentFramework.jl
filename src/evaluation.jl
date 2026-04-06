# evaluation.jl — Provider-agnostic evaluation framework
#
# Port of agent_framework._evaluation (Python) to Julia.
# Defines core evaluation types, conversation splitters, built-in checks,
# the @evaluator macro, LocalEvaluator, and orchestration functions
# (evaluate_agent, evaluate_workflow).

# ─── Exceptions ──────────────────────────────────────────────────────────────

"""Raised when evaluation results contain failures."""
struct EvalNotPassedError <: Exception
    message::String
end
Base.showerror(io::IO, e::EvalNotPassedError) = print(io, "EvalNotPassedError: ", e.message)

# ─── Conversation splitters ──────────────────────────────────────────────────

"""
Strategy for splitting a conversation into (query_messages, response_messages).

Any function `f(conversation::Vector{Message}) -> (Vector{Message}, Vector{Message})`
satisfies this interface. Built-in strategies are `SPLIT_LAST_TURN` and `SPLIT_FULL`.
"""
const ConversationSplitter = Function

"""Split at the last user message (default). Everything up to and including
the last user message is the query; everything after is the response."""
function split_last_turn(conversation::Vector{Message})
    last_user_idx = 0
    for (i, msg) in enumerate(conversation)
        if msg.role == :user
            last_user_idx = i
        end
    end
    if last_user_idx > 0
        return conversation[1:last_user_idx], conversation[last_user_idx+1:end]
    end
    return Message[], copy(conversation)
end

"""Split after the first user message. The first user message (and any preceding
system messages) is the query; the entire remainder is the response."""
function split_full(conversation::Vector{Message})
    for (i, msg) in enumerate(conversation)
        if msg.role == :user
            return conversation[1:i], conversation[i+1:end]
        end
    end
    return Message[], copy(conversation)
end

const SPLIT_LAST_TURN = split_last_turn
const SPLIT_FULL = split_full

# ─── Core data types ─────────────────────────────────────────────────────────

"""Expected tool call for correctness evaluation."""
Base.@kwdef struct ExpectedToolCall
    name::String
    arguments::Union{Nothing, Dict{String, Any}} = nothing
end

"""A single item to be evaluated.

`conversation` is the single source of truth. `query` and `response` are
derived via the split strategy."""
Base.@kwdef mutable struct EvalItem
    conversation::Vector{Message}
    tools::Union{Nothing, Vector{FunctionTool}} = nothing
    context::Union{Nothing, String} = nothing
    expected_output::Union{Nothing, String} = nothing
    expected_tool_calls::Union{Nothing, Vector{ExpectedToolCall}} = nothing
    split_strategy::Union{Nothing, Function} = nothing
end

"""Get the user query text from an EvalItem (derived via split strategy)."""
function eval_query(item::EvalItem)::String
    splitter = something(item.split_strategy, SPLIT_LAST_TURN)
    query_msgs, _ = splitter(item.conversation)
    texts = [get_text(msg) for msg in query_msgs if msg.role == :user && !isempty(get_text(msg))]
    return strip(join(texts, " "))
end

"""Get the agent response text from an EvalItem (derived via split strategy)."""
function eval_response(item::EvalItem)::String
    splitter = something(item.split_strategy, SPLIT_LAST_TURN)
    _, resp_msgs = splitter(item.conversation)
    texts = [get_text(msg) for msg in resp_msgs if msg.role == :assistant && !isempty(get_text(msg))]
    return strip(join(texts, " "))
end

"""Split the conversation into (query_messages, response_messages).
Resolution order: explicit split, then item.split_strategy, then SPLIT_LAST_TURN."""
function split_messages(item::EvalItem; split::Union{Nothing, Function}=nothing)
    effective = something(split, item.split_strategy, SPLIT_LAST_TURN)
    return effective(item.conversation)
end

"""Split a multi-turn conversation into one EvalItem per user turn.
Each item has cumulative context up to that turn."""
function per_turn_items(conversation::Vector{Message};
                        tools::Union{Nothing, Vector{FunctionTool}}=nothing,
                        context::Union{Nothing, String}=nothing)::Vector{EvalItem}
    user_indices = [i for (i, m) in enumerate(conversation) if m.role == :user]
    isempty(user_indices) && return EvalItem[]

    items = EvalItem[]
    for (turn_idx, _ui) in enumerate(user_indices)
        next_ui = turn_idx < length(user_indices) ? user_indices[turn_idx + 1] : length(conversation) + 1
        push!(items, EvalItem(
            conversation = conversation[1:next_ui-1],
            tools = tools,
            context = context,
        ))
    end
    return items
end

# ─── Score and result types ──────────────────────────────────────────────────

"""Result from a single evaluator on a single item."""
Base.@kwdef struct EvalScoreResult
    name::String
    score::Float64
    passed::Union{Nothing, Bool} = nothing
    sample::Union{Nothing, Dict{String, Any}} = nothing
end

"""Per-item result from an evaluation run."""
Base.@kwdef mutable struct EvalItemResult
    item_id::String
    status::String  # "pass", "fail", "error"
    scores::Vector{EvalScoreResult} = EvalScoreResult[]
    error_code::Union{Nothing, String} = nothing
    error_message::Union{Nothing, String} = nothing
    response_id::Union{Nothing, String} = nothing
    input_text::Union{Nothing, String} = nothing
    output_text::Union{Nothing, String} = nothing
    token_usage::Union{Nothing, Dict{String, Int}} = nothing
    metadata::Union{Nothing, Dict{String, Any}} = nothing
end

is_error(r::EvalItemResult) = r.status in ("error", "errored")
is_passed(r::EvalItemResult) = r.status == "pass"
is_failed(r::EvalItemResult) = r.status == "fail"

"""Results from an evaluation run by a single provider."""
Base.@kwdef mutable struct EvalResults
    provider::String
    eval_id::String = ""
    run_id::String = ""
    status::String = "completed"
    result_counts::Union{Nothing, Dict{String, Int}} = nothing
    report_url::Union{Nothing, String} = nothing
    error::Union{Nothing, String} = nothing
    per_evaluator::Dict{String, Dict{String, Int}} = Dict{String, Dict{String, Int}}()
    items::Vector{EvalItemResult} = EvalItemResult[]
    sub_results::Dict{String, EvalResults} = Dict{String, EvalResults}()
end

"""Number of passing results."""
eval_passed(r::EvalResults) = get(something(r.result_counts, Dict{String,Int}()), "passed", 0)

"""Number of failing results."""
eval_failed(r::EvalResults) = get(something(r.result_counts, Dict{String,Int}()), "failed", 0)

"""Total number of results (passed + failed)."""
eval_total(r::EvalResults) = eval_passed(r) + eval_failed(r)

"""Whether all results passed with no failures or errors."""
function all_passed(r::EvalResults)::Bool
    r.status != "completed" && return false
    counts = something(r.result_counts, nothing)
    errored = counts !== nothing ? get(counts, "errored", 0) : 0
    if !isempty(r.sub_results)
        own = counts !== nothing ? (eval_failed(r) == 0 && errored == 0 && eval_total(r) > 0) : true
        return own && all(all_passed(sub) for sub in values(r.sub_results))
    end
    return eval_failed(r) == 0 && errored == 0 && eval_total(r) > 0
end

"""Raise EvalNotPassedError if any results failed or errored."""
function raise_for_status(r::EvalResults; msg::Union{Nothing, String}=nothing)
    all_passed(r) && return nothing
    errored = get(something(r.result_counts, Dict{String,Int}()), "errored", 0)
    detail = something(msg, "Eval run $(r.run_id) $(r.status): $(eval_passed(r)) passed, $(eval_failed(r)) failed.")
    if errored > 0
        detail *= " $errored errored."
    end
    if r.report_url !== nothing
        detail *= " See $(r.report_url) for details."
    end
    if r.error !== nothing
        detail *= " Error: $(r.error)"
    end
    if !isempty(r.sub_results)
        failed_names = [name for (name, sub) in r.sub_results if !all_passed(sub)]
        if !isempty(failed_names)
            detail *= " Failed: $(join(failed_names, ", "))."
        end
    end
    throw(EvalNotPassedError(detail))
end

# ─── Evaluator protocol ─────────────────────────────────────────────────────

"""
Protocol for evaluation providers.

Any type implementing `evaluate(evaluator, items, eval_name)` and having a
`name` field satisfies this protocol. See `LocalEvaluator` for a reference
implementation.
"""
abstract type AbstractEvaluator end

"""Evaluate a batch of items. Must be implemented by evaluator providers."""
function evaluate end

# ─── Check types ─────────────────────────────────────────────────────────────

"""Result of a single check on a single evaluation item."""
Base.@kwdef struct CheckResult
    passed::Bool
    reason::String
    check_name::String
end

"""An EvalCheck is any callable `f(item::EvalItem) -> CheckResult`."""
const EvalCheck = Function

# ─── Built-in checks ────────────────────────────────────────────────────────

"""Check that the response contains all specified keywords.

Returns a check function for use with `LocalEvaluator`.

# Example
```julia
check = keyword_check("weather", "temperature")
```
"""
function keyword_check(keywords::String...; case_sensitive::Bool=false)
    function _check(item::EvalItem)::CheckResult
        text = case_sensitive ? eval_response(item) : lowercase(eval_response(item))
        missing_kw = [k for k in keywords if !occursin(case_sensitive ? k : lowercase(k), text)]
        if !isempty(missing_kw)
            return CheckResult(passed=false, reason="Missing keywords: $missing_kw", check_name="keyword_check")
        end
        return CheckResult(passed=true, reason="All keywords found", check_name="keyword_check")
    end
    return _check
end

"""Check that specific tools were called during the conversation.

`mode` can be `:all` (every tool must be called) or `:any` (at least one).

# Example
```julia
check = tool_called_check("get_weather", "get_flight_price")
```
"""
function tool_called_check(tool_names::String...; mode::Symbol=:all)
    function _check(item::EvalItem)::CheckResult
        expected = Set(tool_names)
        called = Set{String}()
        for msg in item.conversation
            for c in msg.contents
                if is_function_call(c) && c.name !== nothing
                    push!(called, c.name)
                end
            end
        end

        if mode == :all
            missing_tools = [t for t in tool_names if !(t in called)]
            if !isempty(missing_tools)
                return CheckResult(
                    passed=false,
                    reason="Expected tools not called: $missing_tools (called: $(sort(collect(called))))",
                    check_name="tool_called",
                )
            end
            return CheckResult(
                passed=true,
                reason="All expected tools called: $(sort(collect(called)))",
                check_name="tool_called",
            )
        else  # :any
            found = intersect(expected, called)
            if !isempty(found)
                return CheckResult(
                    passed=true,
                    reason="Expected tool found: $(sort(collect(found)))",
                    check_name="tool_called",
                )
            end
            return CheckResult(
                passed=false,
                reason="None of expected tools called: $(collect(tool_names)) (called: $(sort(collect(called))))",
                check_name="tool_called",
            )
        end
    end
    return _check
end

"""Extract (name, arguments) pairs from conversation function calls."""
function _extract_tool_calls(item::EvalItem)
    calls = Tuple{String, Union{Nothing, Dict{String, Any}}}[]
    for msg in item.conversation
        for c in msg.contents
            if is_function_call(c) && c.name !== nothing
                args = nothing
                if c.arguments isa Dict
                    args = c.arguments
                elseif c.arguments isa String
                    try
                        parsed = JSON3.read(c.arguments, Dict{String, Any})
                        if parsed isa Dict
                            args = parsed
                        end
                    catch
                    end
                end
                push!(calls, (c.name, args))
            end
        end
    end
    return calls
end

"""Check that all expected tool calls were made (unordered, extras OK).

Uses `item.expected_tool_calls` — checks names only, not arguments."""
function tool_calls_present(item::EvalItem)::CheckResult
    expected = something(item.expected_tool_calls, ExpectedToolCall[])
    isempty(expected) && return CheckResult(passed=true, reason="No expected tool calls specified.", check_name="tool_calls_present")

    actual_names = Set(name for (name, _) in _extract_tool_calls(item))
    expected_names = [e.name for e in expected]
    missing_names = [n for n in expected_names if !(n in actual_names)]

    if !isempty(missing_names)
        return CheckResult(
            passed=false,
            reason="Missing tool calls: $missing_names (called: $(sort(collect(actual_names))))",
            check_name="tool_calls_present",
        )
    end
    found = [n for n in expected_names if n in actual_names]
    return CheckResult(
        passed=true,
        reason="All expected tools called: $found (called: $(sort(collect(actual_names))))",
        check_name="tool_calls_present",
    )
end

"""Check that expected tool calls match on name and arguments.

Subset matching on arguments — extra actual arguments are OK."""
function tool_call_args_match(item::EvalItem)::CheckResult
    expected = something(item.expected_tool_calls, ExpectedToolCall[])
    isempty(expected) && return CheckResult(passed=true, reason="No expected tool calls specified.", check_name="tool_call_args_match")

    actual_calls = _extract_tool_calls(item)
    matched = 0
    details = String[]

    for exp in expected
        matching = [(n, a) for (n, a) in actual_calls if n == exp.name]
        if isempty(matching)
            push!(details, "  $(exp.name): not called")
            continue
        end

        if exp.arguments === nothing
            matched += 1
            push!(details, "  $(exp.name): called (args not checked)")
            continue
        end

        found = false
        for (_, actual_args) in matching
            actual_args === nothing && continue
            if all(get(actual_args, k, nothing) == v for (k, v) in exp.arguments)
                found = true
                break
            end
        end

        if found
            matched += 1
            push!(details, "  $(exp.name): args match")
        else
            actual_args_list = [a for (_, a) in matching]
            push!(details, "  $(exp.name): args mismatch (actual: $actual_args_list)")
        end
    end

    passed = matched == length(expected)
    score_str = "$matched/$(length(expected))"
    detail_str = join(details, "\n")
    reason = "Tool call args match: $score_str\n$detail_str"

    return CheckResult(passed=passed, reason=reason, check_name="tool_call_args_match")
end

# ─── Function evaluator wrapper ──────────────────────────────────────────────

const _KNOWN_PARAMS = Set(["query", "response", "expected_output",
                           "expected_tool_calls", "conversation", "tools", "context"])

"""Build kwargs dict for a function evaluator based on method parameter names."""
function _resolve_function_args(fn, item::EvalItem, param_names::Set{Symbol})
    field_map = Dict{Symbol, Any}(
        :query => eval_query(item),
        :response => eval_response(item),
        :expected_output => something(item.expected_output, ""),
        :expected_tool_calls => something(item.expected_tool_calls, ExpectedToolCall[]),
        :conversation => item.conversation,
        :tools => item.tools,
        :context => item.context,
    )
    return Dict(k => field_map[k] for k in param_names if haskey(field_map, k))
end

"""Convert a function evaluator return value to a CheckResult."""
function _coerce_result(value, check_name::String)::CheckResult
    value isa CheckResult && return value

    if value isa Bool
        return CheckResult(passed=value, reason=value ? "passed" : "failed", check_name=check_name)
    end

    if value isa Number
        passed = Float64(value) >= 0.5
        return CheckResult(passed=passed, reason="score=$(round(Float64(value), digits=3))", check_name=check_name)
    end

    if value isa Dict
        if haskey(value, "score")
            score = Float64(value["score"])
            passed = haskey(value, "passed") ? Bool(value["passed"]) : score >= Float64(get(value, "threshold", 0.5))
            reason = string(get(value, "reason", "score=$(round(score, digits=3))"))
            return CheckResult(passed=passed, reason=reason, check_name=check_name)
        end
        if haskey(value, "passed")
            p = Bool(value["passed"])
            return CheckResult(passed=p, reason=string(get(value, "reason", p ? "passed" : "failed")), check_name=check_name)
        end
    end

    throw(EvalTypeError("Function evaluator '$check_name' returned unsupported type $(typeof(value)). Expected Bool, Number, Dict, or CheckResult."))
end

struct EvalTypeError <: Exception
    message::String
end
Base.showerror(io::IO, e::EvalTypeError) = print(io, "TypeError: ", e.message)

"""
    make_evaluator(fn; name=nothing) -> EvalCheck

Wrap a plain function as an EvalCheck for use with LocalEvaluator.

The function's parameter names determine what data it receives from the EvalItem.
Supported parameter names: `query`, `response`, `expected_output`,
`expected_tool_calls`, `conversation`, `tools`, `context`.

Return `Bool`, `Number` (≥0.5 = pass), `Dict` with `score` or `passed` key,
or `CheckResult`.

# Examples
```julia
# Simple boolean check
length_check = make_evaluator(; name="length_check") do response
    length(response) < 2000
end

# Score-based check
overlap = make_evaluator(; name="overlap") do query, response
    words = split(lowercase(query))
    matches = count(w -> occursin(w, lowercase(response)), words)
    return matches / max(length(words), 1)
end
```
"""
function make_evaluator(fn; name::Union{Nothing, String}=nothing)
    check_name = something(name, string(nameof(fn)))
    # Introspect method signature to discover parameter names
    meths = methods(fn)
    param_names = Set{Symbol}()
    if !isempty(meths)
        m = first(meths)
        argnames = Base.method_argnames(m)
        # Skip first arg (function itself for closures)
        for an in argnames[2:end]
            sym_str = string(an)
            if sym_str in _KNOWN_PARAMS
                push!(param_names, an)
            end
        end
    end

    function _check(item::EvalItem)::CheckResult
        kwargs = _resolve_function_args(fn, item, param_names)
        # Call with positional args in the order they appear in the signature
        meths2 = methods(fn)
        m2 = first(meths2)
        argorder = Base.method_argnames(m2)[2:end]
        args = [kwargs[a] for a in argorder if haskey(kwargs, a)]
        result = fn(args...)
        return _coerce_result(result, check_name)
    end
    return _check
end

# ─── LocalEvaluator ──────────────────────────────────────────────────────────

"""Evaluation provider that runs checks locally without API calls.

Implements the evaluator protocol. Each check function is applied to every
item. An item passes only if all checks pass.

# Example
```julia
local = LocalEvaluator(
    keyword_check("weather"),
    tool_called_check("get_weather"),
)
results = evaluate(local, items)
```
"""
struct LocalEvaluator <: AbstractEvaluator
    name::String
    checks::Vector{Any}  # Vector of EvalCheck functions
end

LocalEvaluator(checks::Function...) = LocalEvaluator("Local", collect(Any, checks))
LocalEvaluator(checks::Vector) = LocalEvaluator("Local", collect(Any, checks))
LocalEvaluator() = LocalEvaluator("Local", Any[])

function evaluate(evaluator::LocalEvaluator, items::Vector{EvalItem};
                  eval_name::String="Local Eval")::EvalResults
    passed_count = 0
    failed_count = 0
    per_check = Dict{String, Dict{String, Int}}()
    failure_reasons = String[]
    result_items = EvalItemResult[]

    for (item_idx, item) in enumerate(items)
        item_passed = true
        item_scores = EvalScoreResult[]

        for check_fn in evaluator.checks
            result = check_fn(item)::CheckResult
            counts = get!(per_check, result.check_name) do
                Dict("passed" => 0, "failed" => 0, "errored" => 0)
            end
            if result.passed
                counts["passed"] += 1
            else
                counts["failed"] += 1
                item_passed = false
                push!(failure_reasons, "$(result.check_name): $(result.reason)")
            end
            push!(item_scores, EvalScoreResult(
                name=result.check_name,
                score=result.passed ? 1.0 : 0.0,
                passed=result.passed,
                sample=!isempty(result.reason) ? Dict{String,Any}("reason" => result.reason) : nothing,
            ))
        end

        if item_passed
            passed_count += 1
        else
            failed_count += 1
        end

        push!(result_items, EvalItemResult(
            item_id=string(item_idx - 1),
            status=item_passed ? "pass" : "fail",
            scores=item_scores,
            input_text=eval_query(item),
            output_text=eval_response(item),
        ))
    end

    return EvalResults(
        provider=evaluator.name,
        eval_id="local",
        run_id=eval_name,
        status="completed",
        result_counts=Dict("passed" => passed_count, "failed" => failed_count, "errored" => 0),
        per_evaluator=per_check,
        items=result_items,
        error=isempty(failure_reasons) ? nothing : join(failure_reasons, "; "),
    )
end

# ─── Evaluator resolution ───────────────────────────────────────────────────

"""Resolve evaluators: bare functions are collected and wrapped in a single LocalEvaluator."""
function _resolve_evaluators(evaluators)::Vector{AbstractEvaluator}
    if evaluators isa AbstractEvaluator
        return [evaluators]
    end

    result = AbstractEvaluator[]
    pending_checks = Any[]

    items_list = evaluators isa Vector || evaluators isa Tuple ? collect(evaluators) : [evaluators]
    for e in items_list
        if e isa AbstractEvaluator
            if !isempty(pending_checks)
                push!(result, LocalEvaluator(pending_checks...))
                pending_checks = Any[]
            end
            push!(result, e)
        else
            push!(pending_checks, e)
        end
    end
    if !isempty(pending_checks)
        push!(result, LocalEvaluator(pending_checks...))
    end
    return result
end

"""Run resolved evaluators against items, returning one EvalResults per provider."""
function _run_evaluators(evaluators, items::Vector{EvalItem}; eval_name::String)::Vector{EvalResults}
    resolved = _resolve_evaluators(evaluators)
    # Run evaluators (synchronous in Julia — could be @async'd)
    return [evaluate(ev, items; eval_name=eval_name) for ev in resolved]
end

# ─── AgentEvalConverter ──────────────────────────────────────────────────────

"""Convert agent interaction (query + response) to an EvalItem."""
function _eval_to_item(;
    query::Union{String, Vector{Message}},
    response::AgentResponse,
    agent=nothing,
    tools::Union{Nothing, Vector{FunctionTool}}=nothing,
    context::Union{Nothing, String}=nothing,
)::EvalItem
    if query isa String
        query_msgs = [Message(:user, [text_content(query)])]
    else
        query_msgs = query
    end

    resp_msgs = response.messages
    conversation = vcat(query_msgs, resp_msgs)

    actual_tools = tools
    if actual_tools === nothing && agent !== nothing
        if hasproperty(agent, :tools)
            actual_tools = agent.tools
        end
    end

    return EvalItem(
        conversation=conversation,
        tools=actual_tools,
        context=context,
    )
end

"""Extract tool definitions from an agent as dicts."""
function _eval_extract_tools(agent)::Vector{Dict{String, Any}}
    tools = Dict{String, Any}[]
    if hasproperty(agent, :tools)
        for t in agent.tools
            push!(tools, Dict{String, Any}(
                "name" => t.name,
                "description" => something(t.description, ""),
                "parameters" => something(t.parameters, Dict{String,Any}()),
            ))
        end
    end
    return tools
end

# ─── Orchestration: evaluate_agent ───────────────────────────────────────────

"""
    evaluate_agent(; agent, queries, evaluators, ...) -> Vector{EvalResults}

Run an agent against test queries and evaluate the results.

# Arguments
- `agent`: An agent instance (must support `run_agent`).
- `queries`: Test query string(s).
- `expected_output`: Ground-truth expected output(s), one per query.
- `expected_tool_calls`: Expected tool calls per query.
- `responses`: Pre-existing `AgentResponse`(s) to evaluate without running the agent.
- `evaluators`: One or more evaluator instances or check functions.
- `eval_name`: Display name for the evaluation run.
- `context`: Optional grounding context.
- `conversation_split`: Split strategy applied to all items.
- `num_repetitions`: Number of times to run each query (default 1).

# Returns
A `Vector{EvalResults}`, one per evaluator provider.
"""
function evaluate_agent(;
    agent::Union{Nothing, AbstractAgent}=nothing,
    queries::Union{Nothing, String, Vector{String}}=nothing,
    expected_output::Union{Nothing, String, Vector{String}}=nothing,
    expected_tool_calls::Union{Nothing, Vector{ExpectedToolCall}, Vector{Vector{ExpectedToolCall}}}=nothing,
    responses::Union{Nothing, AgentResponse, Vector{AgentResponse}}=nothing,
    evaluators,
    eval_name::Union{Nothing, String}=nothing,
    context::Union{Nothing, String}=nothing,
    conversation_split::Union{Nothing, Function}=nothing,
    num_repetitions::Int=1,
)::Vector{EvalResults}

    # Normalize singular values to lists
    queries_list = queries isa String ? [queries] : queries
    expected_list = expected_output isa String ? [expected_output] : expected_output
    responses_list = responses isa AgentResponse ? [responses] : responses
    tc_list = expected_tool_calls isa Vector{ExpectedToolCall} ? [expected_tool_calls] : expected_tool_calls

    num_repetitions >= 1 || throw(ArgumentError("num_repetitions must be >= 1, got $num_repetitions."))

    # Validate expected_output length
    if expected_list !== nothing && queries_list !== nothing && length(expected_list) != length(queries_list)
        throw(ArgumentError("Got $(length(queries_list)) queries but $(length(expected_list)) expected_output values."))
    end

    # Validate expected_tool_calls length
    if tc_list !== nothing && queries_list !== nothing && length(tc_list) != length(queries_list)
        throw(ArgumentError("Got $(length(queries_list)) queries but $(length(tc_list)) expected_tool_calls lists."))
    end

    items = EvalItem[]

    if responses_list !== nothing
        # Evaluate pre-existing responses
        queries_list === nothing && throw(ArgumentError("Provide 'queries' alongside 'responses'."))
        length(queries_list) != length(responses_list) && throw(ArgumentError(
            "Got $(length(queries_list)) queries but $(length(responses_list)) responses."))

        for (q, r) in zip(queries_list, responses_list)
            push!(items, _eval_to_item(
                query=q, response=r, agent=agent, context=context))
        end
    elseif queries_list !== nothing && agent !== nothing
        # Run agent against queries with repetitions
        for _rep in 1:num_repetitions
            for query in queries_list
                session = create_session(agent)
                response = run_agent(agent, query; session=session)
                push!(items, _eval_to_item(
                    query=query, response=response, agent=agent, context=context))
            end
        end
    elseif queries_list !== nothing && agent === nothing
        throw(ArgumentError("Provide 'agent' when using 'queries'."))
    else
        throw(ArgumentError("Provide either 'queries' (with 'agent') or 'responses'."))
    end

    # Stamp expected output
    if expected_list !== nothing
        qcount = length(expected_list)
        for (i, item) in enumerate(items)
            item.expected_output = expected_list[mod1(i, qcount)]
        end
    end

    # Stamp expected tool calls
    if tc_list !== nothing
        qcount = length(tc_list)
        for (i, item) in enumerate(items)
            item.expected_tool_calls = tc_list[mod1(i, qcount)]
        end
    end

    # Stamp split strategy
    if conversation_split !== nothing
        for item in items
            item.split_strategy = conversation_split
        end
    end

    name = something(eval_name, "Eval: $(agent !== nothing ? string(something(hasproperty(agent, :name) ? agent.name : nothing, "agent")) : "agent")")
    return _run_evaluators(evaluators, items; eval_name=name)
end

# ─── Orchestration: evaluate_workflow ────────────────────────────────────────

"""Extract per-agent eval data from a WorkflowRunResult."""
function _extract_agent_eval_data(result::WorkflowRunResult)
    agent_data = Dict{String, Vector{EvalItem}}()

    events = result isa Vector ? result : result.events
    i = 1
    while i <= length(events)
        evt = events[i]
        if evt.event_type == EVT_EXECUTOR_INVOKED
            executor_id = string(get(evt.data, "executor_id", "unknown"))
            # Skip internal executors
            if !startswith(executor_id, "_") && !(executor_id in ("input-conversation", "end-conversation", "end"))
                # Look for matching EXECUTOR_COMPLETED
                for j in (i+1):length(events)
                    evt_j = events[j]
                    if evt_j.event_type == EVT_EXECUTOR_COMPLETED
                        comp_exec_id = string(get(evt_j.data, "executor_id", ""))
                        if comp_exec_id == executor_id
                            # Extract conversation from the completed event data
                            conv_data = get(evt_j.data, "conversation", nothing)
                            if conv_data isa Vector
                                conversation = Message[]
                                for md in conv_data
                                    if md isa Message
                                        push!(conversation, md)
                                    elseif md isa Dict
                                        push!(conversation, message_from_dict(md))
                                    end
                                end
                                if !isempty(conversation)
                                    items = get!(agent_data, executor_id, EvalItem[])
                                    push!(items, EvalItem(conversation=conversation))
                                end
                            end
                            break
                        end
                    end
                end
            end
        end
        i += 1
    end

    return agent_data
end

"""
    evaluate_workflow(; workflow, evaluators, ...) -> Vector{EvalResults}

Evaluate a multi-agent workflow with per-agent breakdown.

# Arguments
- `workflow`: The workflow instance.
- `workflow_result`: A completed WorkflowRunResult (post-hoc mode).
- `queries`: Test queries to run through the workflow (run + evaluate mode).
- `evaluators`: One or more evaluator instances or check functions.
- `eval_name`: Display name for the evaluation.
- `include_overall`: Whether to evaluate the workflow's final output.
- `include_per_agent`: Whether to evaluate each sub-agent individually.
- `conversation_split`: Split strategy applied to all items.
- `num_repetitions`: Number of times to run each query (default 1).
"""
function evaluate_workflow(;
    workflow,
    workflow_result::Union{Nothing, WorkflowRunResult}=nothing,
    queries::Union{Nothing, String, Vector{String}}=nothing,
    evaluators,
    eval_name::Union{Nothing, String}=nothing,
    include_overall::Bool=true,
    include_per_agent::Bool=true,
    conversation_split::Union{Nothing, Function}=nothing,
    num_repetitions::Int=1,
)::Vector{EvalResults}

    queries_list = queries isa String ? [queries] : queries

    # Get or produce workflow results
    results_list = WorkflowRunResult[]
    if workflow_result !== nothing
        push!(results_list, workflow_result)
    elseif queries_list !== nothing
        for _rep in 1:num_repetitions
            for query in queries_list
                wr = run_workflow(workflow, [Message(:user, [text_content(query)])])
                push!(results_list, wr)
            end
        end
    else
        throw(ArgumentError("Provide either 'workflow_result' or 'queries'."))
    end

    name = something(eval_name, "Workflow Eval")

    # Collect overall items (final outputs)
    overall_items = EvalItem[]
    if include_overall
        for wr in results_list
            outputs = get_outputs(wr)
            for out in outputs
                if out isa String
                    conv = [
                        Message(:user, [text_content("workflow query")]),
                        Message(:assistant, [text_content(out)]),
                    ]
                    push!(overall_items, EvalItem(conversation=conv))
                end
            end
        end
    end

    # Collect per-agent items
    per_agent = Dict{String, Vector{EvalItem}}()
    if include_per_agent
        for wr in results_list
            agent_data = _extract_agent_eval_data(wr)
            for (exec_id, agent_items) in agent_data
                existing = get!(per_agent, exec_id, EvalItem[])
                append!(existing, agent_items)
            end
        end
    end

    # Stamp split strategy
    if conversation_split !== nothing
        for item in overall_items
            item.split_strategy = conversation_split
        end
        for items in values(per_agent)
            for item in items
                item.split_strategy = conversation_split
            end
        end
    end

    # Run evaluators on overall
    overall_results = !isempty(overall_items) ? _run_evaluators(evaluators, overall_items; eval_name=name) : EvalResults[]

    # Run evaluators per-agent
    if include_per_agent && !isempty(per_agent)
        if isempty(overall_results)
            # Create stub results to hold sub_results
            resolved = _resolve_evaluators(evaluators)
            overall_results = [EvalResults(
                provider=ev.name,
                eval_id="workflow",
                run_id=name,
                status="completed",
                result_counts=Dict("passed" => 0, "failed" => 0, "errored" => 0),
            ) for ev in resolved]
        end

        for (i, ov_result) in enumerate(overall_results)
            for (exec_id, agent_items) in per_agent
                resolved = _resolve_evaluators(evaluators)
                if i <= length(resolved)
                    sub = evaluate(resolved[i], agent_items; eval_name="$name: $exec_id")
                    ov_result.sub_results[exec_id] = sub
                end
            end
        end
    end

    return isempty(overall_results) ? [EvalResults(provider="Local", status="completed",
        result_counts=Dict("passed" => 0, "failed" => 0, "errored" => 0))] : overall_results
end

# ─── WorkflowAgent ───────────────────────────────────────────────────────────

"""
    WorkflowAgent <: AbstractAgent

An agent that wraps a Workflow and exposes it via the standard agent interface.
Runs the workflow with the input messages and returns the workflow's output
as an AgentResponse.

Only output events from the workflow are converted to agent response messages.
Other workflow events (logging, checkpointing) are ignored.

# Fields
- `workflow`: The workflow to wrap (any type supporting `run_workflow`).
- `name`: Agent name.
- `description`: Optional agent description.
- `id`: Unique agent identifier.
- `context_providers`: Optional context providers for the agent.
"""
Base.@kwdef mutable struct WorkflowAgent <: AbstractAgent
    workflow::Any
    name::String = "workflow_agent"
    description::Union{Nothing, String} = nothing
    id::String = string(UUIDs.uuid4())
    context_providers::Vector{Any} = Any[]
end

"""
    run_agent(agent::WorkflowAgent, input; session=nothing) -> AgentResponse

Run the wrapped workflow with the given input and return an AgentResponse.
Input is normalized to `Vector{Message}` before being passed to the workflow.
"""
function run_agent(agent::WorkflowAgent, input=nothing;
                   session::Union{Nothing, AgentSession}=nothing,
                   options=nothing)::AgentResponse
    sess = something(session, AgentSession())

    # Normalize input to messages
    msgs = if input isa String
        [Message(:user, [text_content(input)])]
    elseif input isa Vector{Message}
        input
    elseif input === nothing
        Message[]
    else
        [Message(:user, [text_content(string(input))])]
    end

    # Run context providers (before_run!)
    ctx = SessionContext(input_messages=copy(msgs))
    for provider in agent.context_providers
        if applicable(before_run!, provider, agent, sess, ctx, Dict{String, Any}())
            before_run!(provider, agent, sess, ctx, Dict{String, Any}())
        end
    end

    # Run workflow
    result = run_workflow(agent.workflow, ctx.input_messages)
    outputs = get_outputs(result)

    # Convert outputs to response messages
    response_contents = Content[]
    for out in outputs
        if out isa String
            push!(response_contents, text_content(out))
        elseif out isa Content
            push!(response_contents, out)
        elseif out isa Message
            append!(response_contents, out.contents)
        elseif out isa AgentResponse
            for m in out.messages
                append!(response_contents, m.contents)
            end
        end
    end

    response_msg = Message(:assistant, isempty(response_contents) ? [text_content("")] : response_contents)
    all_messages = vcat(msgs, [response_msg])

    response = AgentResponse(messages=all_messages)

    # Run context providers (after_run!)
    ctx.response = response
    for provider in reverse(agent.context_providers)
        if applicable(after_run!, provider, agent, sess, ctx, Dict{String, Any}())
            after_run!(provider, agent, sess, ctx, Dict{String, Any}())
        end
    end

    return response
end
