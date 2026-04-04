using AgentFramework
using Test

@testset "Middleware" begin
    @testset "Agent middleware pipeline" begin
        # Track execution order
        order = String[]

        mw1 = function(ctx::AgentContext, next)
            push!(order, "mw1_before")
            result = next(ctx)
            push!(order, "mw1_after")
            return result
        end

        mw2 = function(ctx::AgentContext, next)
            push!(order, "mw2_before")
            result = next(ctx)
            push!(order, "mw2_after")
            return result
        end

        handler = function(ctx::AgentContext)
            push!(order, "handler")
            ctx.result = "done"
            return "done"
        end

        ctx = AgentContext()
        result = apply_agent_middleware([mw1, mw2], ctx, handler)

        @test result == "done"
        @test order == ["mw1_before", "mw2_before", "handler", "mw2_after", "mw1_after"]
    end

    @testset "Empty middleware pipeline" begin
        handler = function(ctx::AgentContext)
            return "direct"
        end
        result = apply_agent_middleware([], AgentContext(), handler)
        @test result == "direct"
    end

    @testset "Chat middleware pipeline" begin
        logged = Ref(false)
        mw = function(ctx::ChatContext, next)
            logged[] = true
            return next(ctx)
        end
        handler = function(ctx::ChatContext)
            ctx.result = "chat_done"
            return "chat_done"
        end

        result = apply_chat_middleware([mw], ChatContext(), handler)
        @test result == "chat_done"
        @test logged[]
    end

    @testset "Function middleware pipeline" begin
        calls = Int[]
        mw = function(ctx::FunctionInvocationContext, next)
            push!(calls, 1)
            return next(ctx)
        end
        handler = function(ctx::FunctionInvocationContext)
            push!(calls, 2)
            ctx.result = 42
            return 42
        end

        ctx = FunctionInvocationContext(call_id="c1")
        result = apply_function_middleware([mw], ctx, handler)
        @test result == 42
        @test calls == [1, 2]
    end

    @testset "Middleware can short-circuit" begin
        mw = function(ctx::AgentContext, next)
            ctx.result = "intercepted"
            return "intercepted"  # Don't call next
        end
        handler = function(ctx::AgentContext)
            return "should not reach"
        end

        result = apply_agent_middleware([mw], AgentContext(), handler)
        @test result == "intercepted"
    end
end
