# Ergonomic macros for defining workflow executors, handlers, and middleware.
# Mirrors Python's @handler and @executor decorators.

# ── @executor Macro ──────────────────────────────────────────────────────────

"""
    @executor id function(msg::Type, ctx) ... end
    @executor id "Description" function(msg::Type, ctx) ... end

Create an `ExecutorSpec` from a function definition. The first argument's type
annotation determines `input_types` (defaults to `Any` if untyped).

# Examples
```julia
@executor "upper" function(msg::String, ctx)
    send_message(ctx, uppercase(msg))
end

@executor "formatter" "Format data as JSON" function(data::Dict, ctx)
    result = JSON3.write(data)
    yield_output(ctx, result)
    send_message(ctx, result)
end
```
"""
macro executor(args...)
    if length(args) == 2
        id_expr, func_expr = args
        desc_expr = ""
    elseif length(args) == 3
        id_expr, desc_expr, func_expr = args
    else
        error("@executor requires 2 or 3 arguments: @executor id [description] function(...) ... end")
    end

    # Parse the function expression
    if !(func_expr isa Expr && func_expr.head == :function)
        error("@executor: last argument must be a function definition")
    end

    sig = func_expr.args[1]
    body = func_expr.args[2]

    # Extract parameters from signature
    # sig can be :(tuple(args...)) for anonymous functions or :(call(name, args...)) for named
    if sig isa Expr && sig.head == :tuple
        params = sig.args
    elseif sig isa Expr && sig.head == :call
        params = sig.args[2:end]
    else
        params = [sig]
    end

    # First parameter: message (extract type annotation)
    first_param = length(params) >= 1 ? params[1] : :msg
    if first_param isa Expr && first_param.head == :(::)
        param_name = first_param.args[1]
        param_type = first_param.args[2]
    else
        param_name = first_param isa Symbol ? first_param : :msg
        param_type = :Any
    end

    # Second parameter: context
    ctx_param = length(params) >= 2 ? params[2] : :ctx
    if ctx_param isa Expr && ctx_param.head == :(::)
        ctx_name = ctx_param.args[1]
    else
        ctx_name = ctx_param isa Symbol ? ctx_param : :ctx
    end

    return quote
        ExecutorSpec(
            id = $(esc(id_expr)),
            description = $(esc(desc_expr)),
            input_types = DataType[$(esc(param_type))],
            output_types = DataType[Any],
            handler = ($(esc(param_name)), $(esc(ctx_name))) -> $(esc(body)),
        )
    end
end

# ── @handler Macro ───────────────────────────────────────────────────────────

"""
    @handler name function(msg, ctx) ... end

Define a named handler function following the executor handler protocol
`(message, ctx::WorkflowContext) -> nothing`.

# Examples
```julia
@handler uppercase_handler function(msg::String, ctx::WorkflowContext)
    send_message(ctx, uppercase(msg))
end

# Use it:
spec = ExecutorSpec(id="upper", handler=uppercase_handler)
```
"""
macro handler(name, func_expr)
    if !(func_expr isa Expr && func_expr.head == :function)
        error("@handler: second argument must be a function definition")
    end

    sig = func_expr.args[1]
    body = func_expr.args[2]

    # Extract parameters from signature
    if sig isa Expr && sig.head == :tuple
        params = sig.args
    elseif sig isa Expr && sig.head == :call
        params = sig.args[2:end]
    else
        params = [sig]
    end

    # First parameter
    first_param = length(params) >= 1 ? params[1] : :msg
    if first_param isa Expr && first_param.head == :(::)
        param_name = first_param.args[1]
    else
        param_name = first_param isa Symbol ? first_param : :msg
    end

    # Second parameter
    ctx_param = length(params) >= 2 ? params[2] : :ctx
    if ctx_param isa Expr && ctx_param.head == :(::)
        ctx_name = ctx_param.args[1]
    else
        ctx_name = ctx_param isa Symbol ? ctx_param : :ctx
    end

    return quote
        $(esc(name)) = ($(esc(param_name)), $(esc(ctx_name))) -> $(esc(body))
    end
end

# ── @middleware Macro ────────────────────────────────────────────────────────

const _VALID_MIDDLEWARE_KINDS = (:agent, :chat, :function)

"""
    @middleware kind name function(ctx, next) ... end

Define a named middleware function for the specified pipeline layer.

`kind` must be one of `:agent`, `:chat`, or `:function`.

# Examples
```julia
@middleware :agent logging_middleware function(ctx, next)
    @info "Agent invoked"
    result = next(ctx)
    @info "Agent completed"
    return result
end
```
"""
macro middleware(kind, name, func_expr)
    if !(kind isa QuoteNode && kind.value in _VALID_MIDDLEWARE_KINDS)
        error("@middleware: kind must be one of :agent, :chat, or :function")
    end

    if !(func_expr isa Expr && func_expr.head == :function)
        error("@middleware: third argument must be a function definition")
    end

    sig = func_expr.args[1]
    body = func_expr.args[2]

    # Extract parameters from signature
    if sig isa Expr && sig.head == :tuple
        params = sig.args
    elseif sig isa Expr && sig.head == :call
        params = sig.args[2:end]
    else
        params = [sig]
    end

    # First parameter: context
    ctx_param = length(params) >= 1 ? params[1] : :ctx
    if ctx_param isa Expr && ctx_param.head == :(::)
        ctx_name = ctx_param.args[1]
    else
        ctx_name = ctx_param isa Symbol ? ctx_param : :ctx
    end

    # Second parameter: next
    next_param = length(params) >= 2 ? params[2] : :next
    if next_param isa Expr && next_param.head == :(::)
        next_name = next_param.args[1]
    else
        next_name = next_param isa Symbol ? next_param : :next
    end

    return quote
        $(esc(name)) = ($(esc(ctx_name)), $(esc(next_name))) -> $(esc(body))
    end
end

# ── @pipeline Macro ──────────────────────────────────────────────────────────

"""
    @pipeline name executor1 => executor2 => executor3

Create a `Workflow` from a linear chain of `ExecutorSpec` values.
The first executor becomes the start; the last is marked as output.

# Examples
```julia
upper = @executor "upper" function(msg::String, ctx)
    send_message(ctx, uppercase(msg))
end

reverse_exec = @executor "reverse" function(msg::String, ctx)
    yield_output(ctx, reverse(msg))
end

workflow = @pipeline "TextPipeline" upper => reverse_exec
```
"""
macro pipeline(name, chain_expr)
    # Parse the chain A => B => C into a flat list
    executors = _parse_chain(chain_expr)

    if length(executors) < 2
        error("@pipeline requires at least 2 executors chained with =>")
    end

    return quote
        let _execs = [$(map(esc, executors)...)]
            _builder = WorkflowBuilder(name = $(esc(name)), start = _execs[1])
            for i in 2:length(_execs)
                add_executor(_builder, _execs[i])
                add_edge(_builder, _execs[i-1].id, _execs[i].id)
            end
            add_output(_builder, _execs[end].id)
            build(_builder; validate_types = false)
        end
    end
end

"""Recursively parse `A => B => C` into a flat list of expressions."""
function _parse_chain(expr)
    if expr isa Expr && expr.head == :call && expr.args[1] == :(=>)
        left = _parse_chain(expr.args[2])
        right = _parse_chain(expr.args[3])
        return vcat(left, right)
    else
        return [expr]
    end
end
