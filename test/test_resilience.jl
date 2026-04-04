using AgentFramework
using Test

# ── Test helpers ─────────────────────────────────────────────────────────────

struct MockChatClient <: AbstractChatClient end
AgentFramework.get_response(::MockChatClient, msgs, opts) =
    ChatResponse(messages=[Message(role=:assistant, contents=[text_content("mock")])])

function make_failing_next(failures::Int, error=ChatClientError("timeout"))
    count = Ref(0)
    return (ctx) -> begin
        count[] += 1
        if count[] <= failures
            throw(error)
        end
        ctx.result = ChatResponse(messages=[Message(role=:assistant, contents=[text_content("ok")])])
    end
end

# ── Tests ────────────────────────────────────────────────────────────────────

@testset "Resilience" begin

    # ── RetryConfig ──────────────────────────────────────────────────────────

    @testset "RetryConfig defaults" begin
        cfg = RetryConfig()
        @test cfg.max_retries == 3
        @test cfg.initial_delay == 1.0
        @test cfg.max_delay == 60.0
        @test cfg.multiplier == 2.0
        @test cfg.jitter == 0.1
        @test 429 in cfg.retryable_status_codes
    end

    @testset "RetryConfig custom values" begin
        cfg = RetryConfig(max_retries=5, initial_delay=0.5, multiplier=3.0, jitter=0.0)
        @test cfg.max_retries == 5
        @test cfg.initial_delay == 0.5
        @test cfg.multiplier == 3.0
        @test cfg.jitter == 0.0
    end

    # ── _compute_delay ───────────────────────────────────────────────────────

    @testset "_compute_delay exponential increase" begin
        cfg = RetryConfig(initial_delay=1.0, multiplier=2.0, jitter=0.0, max_delay=60.0)
        d0 = AgentFramework._compute_delay(0, cfg)
        d1 = AgentFramework._compute_delay(1, cfg)
        d2 = AgentFramework._compute_delay(2, cfg)
        @test d0 ≈ 1.0
        @test d1 ≈ 2.0
        @test d2 ≈ 4.0
    end

    @testset "_compute_delay respects max_delay" begin
        cfg = RetryConfig(initial_delay=1.0, multiplier=10.0, jitter=0.0, max_delay=5.0)
        d = AgentFramework._compute_delay(3, cfg)
        @test d ≈ 5.0
    end

    # ── _is_retryable ────────────────────────────────────────────────────────

    @testset "_is_retryable detects ChatClientError with timeout" begin
        e = ChatClientError("Connection timeout")
        @test AgentFramework._is_retryable(e, RetryConfig()) == true
    end

    @testset "_is_retryable returns false for unrelated errors" begin
        e = ErrorException("some bug")
        @test AgentFramework._is_retryable(e, RetryConfig()) == false
    end

    # ── retry_chat_middleware ────────────────────────────────────────────────

    @testset "retry_chat_middleware — succeeds on first try" begin
        cfg = RetryConfig(max_retries=3, initial_delay=0.001)
        mw = retry_chat_middleware(cfg)
        ctx = ChatContext()
        called = Ref(0)
        handler = (c) -> begin
            called[] += 1
            c.result = ChatResponse(messages=[Message(role=:assistant, contents=[text_content("ok")])])
        end
        mw(ctx, handler)
        @test called[] == 1
    end

    @testset "retry_chat_middleware — succeeds after transient failure" begin
        cfg = RetryConfig(max_retries=3, initial_delay=0.001, jitter=0.0)
        mw = retry_chat_middleware(cfg)
        ctx = ChatContext()
        next = make_failing_next(2, ChatClientError("timeout"))
        mw(ctx, next)
        @test ctx.result isa ChatResponse
    end

    @testset "retry_chat_middleware — exhausts retries and throws" begin
        cfg = RetryConfig(max_retries=2, initial_delay=0.001, jitter=0.0)
        mw = retry_chat_middleware(cfg)
        ctx = ChatContext()
        next = make_failing_next(10, ChatClientError("timeout"))
        @test_throws ChatClientError mw(ctx, next)
    end

    @testset "retry_chat_middleware — doesn't retry non-retryable errors" begin
        cfg = RetryConfig(max_retries=3, initial_delay=0.001)
        mw = retry_chat_middleware(cfg)
        ctx = ChatContext()
        call_count = Ref(0)
        next = (c) -> begin
            call_count[] += 1
            throw(ErrorException("non-retryable"))
        end
        @test_throws ErrorException mw(ctx, next)
        @test call_count[] == 1
    end

    # ── TokenBucketRateLimiter ───────────────────────────────────────────────

    @testset "TokenBucketRateLimiter construction" begin
        rl = TokenBucketRateLimiter(requests_per_second=5.0, burst=20)
        @test rl.capacity == 20.0
        @test rl.tokens == 20.0
        @test rl.refill_rate == 5.0
    end

    @testset "acquire! succeeds when tokens available" begin
        rl = TokenBucketRateLimiter(requests_per_second=10.0, burst=5)
        @test acquire!(rl; timeout=0.1) == true
        @test rl.tokens < 5.0  # one token consumed
    end

    @testset "acquire! blocks then succeeds after refill" begin
        rl = TokenBucketRateLimiter(requests_per_second=100.0, burst=1)
        @test acquire!(rl; timeout=1.0) == true   # consume the single token
        @test acquire!(rl; timeout=1.0) == true   # should refill quickly at 100/s
    end

    @testset "acquire! times out when rate exceeded" begin
        rl = TokenBucketRateLimiter(requests_per_second=0.01, burst=1)
        acquire!(rl; timeout=0.1)  # consume the single token
        @test acquire!(rl; timeout=0.05) == false  # very slow refill → timeout
    end

    # ── rate_limit_chat_middleware ────────────────────────────────────────────

    @testset "rate_limit_chat_middleware — allows request" begin
        rl = TokenBucketRateLimiter(requests_per_second=10.0, burst=5)
        mw = rate_limit_chat_middleware(rl; timeout=1.0)
        ctx = ChatContext()
        called = Ref(false)
        handler = (c) -> begin called[] = true; c.result = "ok" end
        mw(ctx, handler)
        @test called[] == true
    end

    @testset "rate_limit_chat_middleware — throws on timeout" begin
        rl = TokenBucketRateLimiter(requests_per_second=0.01, burst=1)
        acquire!(rl; timeout=0.1)  # drain
        mw = rate_limit_chat_middleware(rl; timeout=0.05)
        ctx = ChatContext()
        handler = (c) -> c.result = "ok"
        @test_throws ChatClientError mw(ctx, handler)
    end

    # ── Agent convenience functions ──────────────────────────────────────────

    @testset "with_retry! adds middleware" begin
        agent = Agent(client=MockChatClient())
        @test isempty(agent.chat_middlewares)
        with_retry!(agent)
        @test length(agent.chat_middlewares) == 1
    end

    @testset "with_rate_limit! adds middleware at front" begin
        agent = Agent(client=MockChatClient())
        # Add a dummy middleware first
        push!(agent.chat_middlewares, (ctx, next) -> next(ctx))
        rl = TokenBucketRateLimiter(requests_per_second=10.0, burst=5)
        with_rate_limit!(agent, rl)
        @test length(agent.chat_middlewares) == 2
    end

    # ── Combined ─────────────────────────────────────────────────────────────

    @testset "Retry + rate limit combined" begin
        rl = TokenBucketRateLimiter(requests_per_second=100.0, burst=10)
        cfg = RetryConfig(max_retries=2, initial_delay=0.001, jitter=0.0)

        agent = Agent(client=MockChatClient())
        with_rate_limit!(agent, rl)
        with_retry!(agent, cfg)
        @test length(agent.chat_middlewares) == 2

        # Run through the middleware chain manually
        ctx = ChatContext()
        call_count = Ref(0)
        handler = (c) -> begin
            call_count[] += 1
            if call_count[] == 1
                throw(ChatClientError("temporary failure"))
            end
            c.result = ChatResponse(messages=[Message(role=:assistant, contents=[text_content("ok")])])
        end
        apply_chat_middleware(agent.chat_middlewares, ctx, handler)
        @test call_count[] == 2
        @test ctx.result isa ChatResponse
    end

end  # @testset "Resilience"
