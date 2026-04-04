// Resilience — Retry and rate limiting (C#)
//
// This sample demonstrates production-grade resilience patterns: retry with
// exponential backoff and token-bucket rate limiting. It mirrors the Julia
// vignette 11_resilience.
//
// Prerequisites:
//   - Ollama running locally with qwen3:8b pulled
//   - dotnet restore

using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var modelName = Environment.GetEnvironmentVariable("OLLAMA_MODEL_NAME")
    ?? "qwen3:8b";

// ── 1. Retry Configuration ──────────────────────────────────────────────────

Console.WriteLine("=== Retry Configuration ===");

var retryConfig = new RetryConfig(
    MaxRetries: 3,
    InitialDelay: TimeSpan.FromSeconds(1),
    MaxDelay: TimeSpan.FromSeconds(60),
    Multiplier: 2.0,
    Jitter: 0.1
);

for (int attempt = 0; attempt < 4; attempt++)
{
    var delay = retryConfig.ComputeDelay(attempt);
    Console.WriteLine($"  Attempt {attempt}: delay = {delay.TotalSeconds:F2}s");
}

// ── 2. Rate Limiter ─────────────────────────────────────────────────────────

Console.WriteLine("\n=== Rate Limiter ===");

var limiter = new TokenBucketRateLimiter(requestsPerSecond: 5.0, burst: 10);
bool acquired = await limiter.AcquireAsync(timeout: TimeSpan.FromSeconds(5));
Console.WriteLine($"  Token acquired: {acquired}");

// ── 3. Resilient Agent Call ─────────────────────────────────────────────────

Console.WriteLine("\n=== Resilient Agent Call ===");

AIAgent agent = new OllamaApiClient(new Uri(endpoint), modelName)
    .AsAIAgent(
        name: "ResilientAgent",
        instructions: "You are a helpful assistant. Keep answers brief.");

var response = await RetryHelper.RunWithRetryAsync(
    async () => await agent.RunAsync("What is the speed of light?"),
    retryConfig
);
Console.WriteLine($"  Agent: {response.Text}");

// ── Supporting Types ────────────────────────────────────────────────────────

/// <summary>Configuration for retry with exponential backoff.</summary>
record RetryConfig(
    int MaxRetries = 3,
    TimeSpan? InitialDelay = null,
    TimeSpan? MaxDelay = null,
    double Multiplier = 2.0,
    double Jitter = 0.1)
{
    private static readonly Random s_random = new();

    public TimeSpan ComputeDelay(int attempt)
    {
        var initial = InitialDelay ?? TimeSpan.FromSeconds(1);
        var max = MaxDelay ?? TimeSpan.FromSeconds(60);
        double baseDelay = Math.Min(
            initial.TotalSeconds * Math.Pow(Multiplier, attempt),
            max.TotalSeconds);
        double jitterAmount = baseDelay * Jitter * (2 * s_random.NextDouble() - 1);
        return TimeSpan.FromSeconds(Math.Max(0, baseDelay + jitterAmount));
    }

    public bool IsRetryable(Exception ex)
    {
        string msg = ex.Message.ToLowerInvariant();
        int[] retryableCodes = [429, 500, 502, 503, 504];
        string[] retryableKeywords = ["timeout", "rate limit", "connection", "temporary"];
        return retryableCodes.Any(c => msg.Contains(c.ToString()))
            || retryableKeywords.Any(kw => msg.Contains(kw));
    }
}

/// <summary>Token bucket rate limiter for controlling request throughput.</summary>
class TokenBucketRateLimiter
{
    private readonly double _capacity;
    private double _tokens;
    private readonly double _refillRate;
    private DateTime _lastRefill;

    public TokenBucketRateLimiter(double requestsPerSecond = 10.0, int burst = 10)
    {
        _capacity = burst;
        _tokens = burst;
        _refillRate = requestsPerSecond;
        _lastRefill = DateTime.UtcNow;
    }

    public async Task<bool> AcquireAsync(TimeSpan? timeout = null)
    {
        var deadline = DateTime.UtcNow + (timeout ?? TimeSpan.FromSeconds(30));
        while (DateTime.UtcNow < deadline)
        {
            Refill();
            if (_tokens >= 1.0)
            {
                _tokens -= 1.0;
                return true;
            }
            var waitTime = Math.Min(1.0 / _refillRate, (deadline - DateTime.UtcNow).TotalSeconds);
            if (waitTime > 0) await Task.Delay(TimeSpan.FromSeconds(waitTime));
        }
        return false;
    }

    private void Refill()
    {
        var now = DateTime.UtcNow;
        var elapsed = (now - _lastRefill).TotalSeconds;
        _tokens = Math.Min(_capacity, _tokens + elapsed * _refillRate);
        _lastRefill = now;
    }
}

/// <summary>Helper for running async functions with retry logic.</summary>
static class RetryHelper
{
    public static async Task<T> RunWithRetryAsync<T>(Func<Task<T>> func, RetryConfig config)
    {
        Exception? lastError = null;
        for (int attempt = 0; attempt <= config.MaxRetries; attempt++)
        {
            try
            {
                return await func();
            }
            catch (Exception ex)
            {
                lastError = ex;
                if (attempt == config.MaxRetries || !config.IsRetryable(ex))
                    throw;
                var delay = config.ComputeDelay(attempt);
                Console.WriteLine($"  Retry {attempt + 1}/{config.MaxRetries}: "
                    + $"waiting {delay.TotalSeconds:F2}s after {ex.GetType().Name}");
                await Task.Delay(delay);
            }
        }
        throw lastError!;
    }
}
