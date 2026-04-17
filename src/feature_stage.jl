# ──────────────────────────────────────────────────────────────────────────────
# feature_stage.jl — experimental / release-candidate feature markers
# ──────────────────────────────────────────────────────────────────────────────
#
# Mirrors agent-framework/python/packages/core/agent_framework/_feature_stage.py.
#
# Provides `@experimental` and `@release_candidate` macros that attach
# stage metadata to functions or types and emit a one-shot warning the
# first time the marked entity is constructed or called.
#
# Usage:
#
#     @experimental feature_id=:EVALS function run_eval(...)
#         ...
#     end
#
#     @experimental feature_id=:SKILLS struct SkillRunner ... end
#
#     @release_candidate feature_id=:PLANNED_FEATURE mutable struct Foo ... end
#
# Feature IDs are Symbols (drawn from `ExperimentalFeature` or
# `ReleaseCandidateFeature` below). Warnings are emitted via
# `@warn` once per (stage, feature_id) pair per Julia process.

const FEATURE_STAGE_EXPERIMENTAL = :experimental
const FEATURE_STAGE_RELEASE_CANDIDATE = :release_candidate

"""
    ExperimentalFeature

Enumeration-style module of current experimental feature IDs. The members
are `Symbol` constants that can be passed to `@experimental` as
`feature_id=ExperimentalFeature.EVALS`.

This is a stage-scoped inventory, not a stable introspection surface —
members may move or be removed as features advance.
"""
module ExperimentalFeature
const EVALS = :EVALS
const SKILLS = :SKILLS
end

"""
    ReleaseCandidateFeature

Enumeration-style module of current release-candidate feature IDs. Use
with `@release_candidate feature_id=ReleaseCandidateFeature.FOO ...`.
"""
module ReleaseCandidateFeature
end

# Registry of (stage, feature_id) pairs that have already emitted a warning.
const _FEATURE_STAGE_WARNED = Set{Tuple{Symbol, Symbol}}()

# Metadata lookup — object -> (stage, feature_id). Keyed by objectid for
# opaque values (e.g. function objects, types) so callers can introspect
# via `feature_stage(obj)` without polluting the object itself.
const _FEATURE_STAGE_REGISTRY = Dict{UInt, Tuple{Symbol, Symbol}}()

"""
    feature_stage(obj) -> Union{Nothing, Tuple{Symbol, Symbol}}

Return the `(stage, feature_id)` tuple for `obj` if it has been marked
with `@experimental` or `@release_candidate`, otherwise `nothing`.
"""
function feature_stage(obj)
    return get(_FEATURE_STAGE_REGISTRY, objectid(obj), nothing)
end

function _stage_warning_message(stage::Symbol, feature_id::Symbol, name::AbstractString)
    if stage == FEATURE_STAGE_EXPERIMENTAL
        return string(
            "[", feature_id, "] ", name,
            " is experimental and may change or be removed in future versions without notice.",
        )
    else
        return string(
            "[", feature_id, "] ", name,
            " is in release-candidate stage and may receive minor refinements",
            " before it is considered generally available.",
        )
    end
end

function _warn_feature_stage_once(stage::Symbol, feature_id::Symbol, name::AbstractString)
    key = (stage, feature_id)
    key in _FEATURE_STAGE_WARNED && return nothing
    push!(_FEATURE_STAGE_WARNED, key)
    # Only emit a runtime warning for experimental; release-candidate is
    # documented-only (matches the Python behaviour where the RC decorator
    # passes `warning_category=None`).
    if stage == FEATURE_STAGE_EXPERIMENTAL
        @warn _stage_warning_message(stage, feature_id, name)
    end
    return nothing
end

function _register_feature_stage(obj, stage::Symbol, feature_id::Symbol)
    _FEATURE_STAGE_REGISTRY[objectid(obj)] = (stage, feature_id)
    return obj
end

"""
    _parse_stage_args(args) -> (feature_id::Symbol, target_expr)

Helper for `@experimental` / `@release_candidate` macros. Accepts
`feature_id=<expr>` followed by a target expression (function, struct,
mutable struct, or abstract type).
"""
function _parse_stage_args(args)
    length(args) >= 2 || throw(ArgumentError(
        "expected `feature_id=<id>` followed by a declaration (function, struct, or abstract type).",
    ))
    kwargs = args[1:(end - 1)]
    target = args[end]
    feature_id_expr = nothing
    for kwarg in kwargs
        if kwarg isa Expr && kwarg.head == :(=) && kwarg.args[1] == :feature_id
            feature_id_expr = kwarg.args[2]
        else
            throw(ArgumentError("unexpected argument $(kwarg) for feature-stage macro"))
        end
    end
    feature_id_expr === nothing && throw(ArgumentError("missing feature_id=... keyword"))
    return feature_id_expr, target
end

function _wrap_target_with_stage(target, stage::Symbol, feature_id_expr, mod::Module)
    # Generate a named registration expression that, at top-level, registers
    # the defined function/type in the feature-stage registry and wraps it
    # with a one-shot warning.
    if target isa Expr && target.head in (:function, :(=)) && target.args[1] isa Expr
        # Function definition. Wrap its body to emit a one-shot warning.
        sig = target.args[1]
        body = target.args[2]
        fn_name = _extract_function_name(sig)
        name_str = string(fn_name)
        new_body = quote
            AgentFramework._warn_feature_stage_once($(QuoteNode(stage)), $(esc(feature_id_expr)), $name_str)
            $(esc(body))
        end
        new_target = Expr(:function, esc(sig), new_body)
        register = quote
            AgentFramework._register_feature_stage(
                $(esc(fn_name)),
                $(QuoteNode(stage)),
                $(esc(feature_id_expr)),
            )
        end
        return Expr(:block, new_target, register, esc(fn_name))
    elseif target isa Expr && target.head == :struct
        # struct / mutable struct. We don't wrap the constructor with a
        # warning (Julia auto-generates inner constructors and tampering
        # with them from a macro is fragile). Instead, register the type
        # in the stage metadata registry so callers can introspect it via
        # `feature_stage(MyType)`.
        struct_sig = target.args[2]
        type_name = _extract_struct_name(struct_sig)
        register = quote
            AgentFramework._register_feature_stage(
                $(esc(type_name)),
                $(QuoteNode(stage)),
                $(esc(feature_id_expr)),
            )
        end
        return Expr(:block, esc(target), register, esc(type_name))
    elseif target isa Expr && target.head == :abstract
        type_name = _extract_abstract_name(target)
        register = quote
            AgentFramework._register_feature_stage(
                $(esc(type_name)),
                $(QuoteNode(stage)),
                $(esc(feature_id_expr)),
            )
        end
        return Expr(:block, esc(target), register, esc(type_name))
    else
        throw(ArgumentError("@$(stage) can only wrap a function, struct, mutable struct, or abstract type"))
    end
end

function _extract_function_name(sig::Expr)
    if sig.head == :call
        name = sig.args[1]
        return name isa Expr && name.head == :(.) ? sig.args[1] : name
    elseif sig.head == :where
        return _extract_function_name(sig.args[1])
    else
        throw(ArgumentError("cannot extract function name from $(sig)"))
    end
end

function _extract_struct_name(sig)
    if sig isa Symbol
        return sig
    elseif sig isa Expr && sig.head == :curly
        return sig.args[1]
    elseif sig isa Expr && sig.head == :<:
        return _extract_struct_name(sig.args[1])
    else
        throw(ArgumentError("cannot extract struct name from $(sig)"))
    end
end

function _extract_abstract_name(expr::Expr)
    # abstract type T end, or abstract type T <: S end
    body = expr.args[1]
    if body isa Symbol
        return body
    elseif body isa Expr && body.head == :<:
        return _extract_struct_name(body.args[1])
    else
        throw(ArgumentError("cannot extract abstract type name from $(expr)"))
    end
end

"""
    @experimental feature_id=<id> <declaration>

Mark a function, struct, mutable struct, or abstract type as experimental.
The first time the marked entity is constructed or called, a warning is
logged via `@warn` that identifies the feature and notes the experimental
status. The stage is also registered in a global registry and can be
introspected via [`feature_stage`](@ref).

Example:

    @experimental feature_id=ExperimentalFeature.EVALS function run_eval(args...)
        ...
    end

    @experimental feature_id=ExperimentalFeature.SKILLS mutable struct SkillRunner
        name::String
    end
"""
macro experimental(args...)
    feature_id_expr, target = _parse_stage_args(args)
    return _wrap_target_with_stage(target, FEATURE_STAGE_EXPERIMENTAL, feature_id_expr, __module__)
end

"""
    @release_candidate feature_id=<id> <declaration>

Mark a function, struct, mutable struct, or abstract type as
release-candidate. Unlike `@experimental`, release-candidate entities do
not emit a runtime warning; the docstring and registry entry document
their status. Use for APIs that are stabilising but may still receive
minor refinements before being considered generally available.
"""
macro release_candidate(args...)
    feature_id_expr, target = _parse_stage_args(args)
    return _wrap_target_with_stage(target, FEATURE_STAGE_RELEASE_CANDIDATE, feature_id_expr, __module__)
end
