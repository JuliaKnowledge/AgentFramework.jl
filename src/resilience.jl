# Retry/resilience and rate-limiting middleware for AgentFramework.jl
# Plugs into the existing chat middleware pipeline.

# ── Retry Configuration ──────────────────────────────────────────────────────

"""
    RetryConfig

Configuration for retry behaviour with exponential backoff.

# Fields
- `max_retries::Int`: Maximum number of retry attempts (default 3).
- `initial_delay::Float64`: Base delay in seconds before first retry (default 1.0).
- `max_delay::Float64`: Cap on the computed delay (default 60.0).
- `multiplier::Float64`: Exponential backoff factor (default 2.0).
- `jitter::Float64`: Random jitter fraction 0–1 applied to delay (default 0.1).
- `retryable_errors::Vector{Type}`: Exception types that are always retryable.
- `retryable_status_codes::Vector{Int}`: HTTP status codes considered transient.
"""
Base.@kwdef struct RetryConfig
    max_retries::Int = 3
    initial_delay::Float64 = 1.0
    max_delay::Float64 = 60.0
    multiplier::Float64 = 2.0
    jitter::Float64 = 0.1
    retryable_errors::Vector{Type} = Type[HTTP.ExceptionRequest.StatusError]
    retryable_status_codes::Vector{Int} = [429, 500, 502, 503, 504]
end

# ── Retry Helpers ────────────────────────────────────────────────────────────

"""Compute the backoff delay for a given attempt number."""
function _compute_delay(attempt::Int, config::RetryConfig)::Float64
    base_delay = min(config.initial_delay * config.multiplier^attempt, config.max_delay)
    jitter_amount = base_delay * config.jitter * (2 * rand() - 1)
    return max(0.0, base_delay + jitter_amount)
end

"""Return `true` when `e` should trigger a retry."""
function _is_retryable(e::Exception, config::RetryConfig)::Bool
    # Explicit retryable types
    for T in config.retryable_errors
        e isa T && return true
    end
    # HTTP status codes
    if hasproperty(e, :status) || hasfield(typeof(e), :status)
        status = try
            getfield(e, :status)
        catch
            return false
        end
        return status in config.retryable_status_codes
    end
    # ChatClientError with transient keywords
    if e isa ChatClientError
        msg = lowercase(sprint(showerror, e))
        return any(kw -> occursin(kw, msg),
            ["timeout", "rate limit", "429", "500", "502", "503", "504",
             "connection", "temporary"])
    end
    return false
end

# ── Retry Chat Middleware ────────────────────────────────────────────────────

"""
    retry_chat_middleware(config::RetryConfig = RetryConfig())

Create a chat middleware that retries failed LLM calls with exponential backoff.
"""
function retry_chat_middleware(config::RetryConfig = RetryConfig())
    return (ctx::ChatContext, next::Function) -> begin
        last_error = nothing
        for attempt in 0:config.max_retries
            try
                return next(ctx)
            catch e
                last_error = e
                if attempt == config.max_retries || !_is_retryable(e, config)
                    rethrow()
                end
                delay = _compute_delay(attempt, config)
                @warn "Chat request failed (attempt $(attempt+1)/$(config.max_retries+1)), retrying in $(round(delay, digits=2))s" exception=e
                sleep(delay)
            end
        end
        throw(last_error)
    end
end

# ── Token Bucket Rate Limiter ────────────────────────────────────────────────

"""
    TokenBucketRateLimiter

Thread-safe token bucket rate limiter.

# Fields
- `capacity::Float64`: Maximum tokens (burst size).
- `tokens::Float64`: Current available tokens.
- `refill_rate::Float64`: Tokens added per second.
- `last_refill::Float64`: Timestamp of last refill.
- `lock::ReentrantLock`: Concurrency guard.
"""
mutable struct TokenBucketRateLimiter
    capacity::Float64
    tokens::Float64
    refill_rate::Float64
    last_refill::Float64
    lock::ReentrantLock
end

"""
    TokenBucketRateLimiter(; requests_per_second=10.0, burst=10)

Construct a rate limiter that allows `requests_per_second` sustained throughput
with a burst allowance of `burst` tokens.
"""
function TokenBucketRateLimiter(; requests_per_second::Float64 = 10.0, burst::Int = 10)
    now = time()
    TokenBucketRateLimiter(Float64(burst), Float64(burst), requests_per_second, now, ReentrantLock())
end

function _refill!(limiter::TokenBucketRateLimiter)
    now = time()
    elapsed = now - limiter.last_refill
    limiter.tokens = min(limiter.capacity, limiter.tokens + elapsed * limiter.refill_rate)
    limiter.last_refill = now
end

"""
    acquire!(limiter::TokenBucketRateLimiter; timeout=30.0) -> Bool

Block until a token is available or `timeout` seconds elapse.
Returns `true` if a token was acquired, `false` on timeout.
"""
function acquire!(limiter::TokenBucketRateLimiter; timeout::Float64 = 30.0)::Bool
    deadline = time() + timeout
    while time() < deadline
        acquired = lock(limiter.lock) do
            _refill!(limiter)
            if limiter.tokens >= 1.0
                limiter.tokens -= 1.0
                return true
            end
            return false
        end
        acquired && return true
        wait_time = min(1.0 / limiter.refill_rate, deadline - time())
        wait_time > 0 && sleep(wait_time)
    end
    return false
end

# ── Rate Limit Chat Middleware ───────────────────────────────────────────────

"""
    rate_limit_chat_middleware(limiter; timeout=30.0)

Create a chat middleware that rate-limits LLM calls using a `TokenBucketRateLimiter`.
"""
function rate_limit_chat_middleware(limiter::TokenBucketRateLimiter; timeout::Float64 = 30.0)
    return (ctx::ChatContext, next::Function) -> begin
        if !acquire!(limiter; timeout=timeout)
            throw(ChatClientError("Rate limit exceeded: could not acquire token within $(timeout)s"))
        end
        return next(ctx)
    end
end

# ── Agent Convenience Functions ──────────────────────────────────────────────

"""
    with_retry!(agent::Agent, config::RetryConfig = RetryConfig())

Add retry middleware to an agent's chat pipeline. Returns the agent.
"""
function with_retry!(agent::Agent, config::RetryConfig = RetryConfig())
    push!(agent.chat_middlewares, retry_chat_middleware(config))
    return agent
end

"""
    with_rate_limit!(agent::Agent, limiter::TokenBucketRateLimiter; timeout=30.0)

Add rate-limiting middleware to the *front* of an agent's chat pipeline. Returns the agent.
"""
function with_rate_limit!(agent::Agent, limiter::TokenBucketRateLimiter; timeout::Float64 = 30.0)
    pushfirst!(agent.chat_middlewares, rate_limit_chat_middleware(limiter; timeout=timeout))
    return agent
end
