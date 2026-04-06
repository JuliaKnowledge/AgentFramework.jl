# Tokenizer protocol for AgentFramework.jl
# Provides an abstract tokenizer interface and a default character-estimation tokenizer.

# ── Abstract Interface ──────────────────────────────────────────────────────

"""
    AbstractTokenizer

Abstract base type for tokenizers. Implement `count_tokens(tokenizer, text)`.
"""
abstract type AbstractTokenizer end

"""
    count_tokens(tokenizer::AbstractTokenizer, text::AbstractString) -> Int

Count (or estimate) the number of tokens in `text` using the given tokenizer.
"""
function count_tokens end

"""
    count_message_tokens(tokenizer::AbstractTokenizer, messages::Vector{Message}) -> Int

Estimate the total token count across all messages. Default implementation
sums `count_tokens` over each message's text content, adding a small overhead
per message for role/framing (4 tokens per message, as in OpenAI's accounting).
"""
function count_message_tokens(tokenizer::AbstractTokenizer, messages::Vector{Message})::Int
    total = 0
    for msg in messages
        total += 4  # role/framing overhead
        for c in msg.contents
            if c.type == TEXT
                total += count_tokens(tokenizer, something(c.text, ""))
            elseif c.type == FUNCTION_CALL
                total += count_tokens(tokenizer, something(c.name, ""))
                total += count_tokens(tokenizer, something(c.arguments, ""))
            elseif c.type == FUNCTION_RESULT
                total += count_tokens(tokenizer, something(c.result, ""))
            end
        end
    end
    return total
end

# ── Character Estimator Tokenizer ───────────────────────────────────────────

"""
    CharacterEstimatorTokenizer <: AbstractTokenizer

A simple tokenizer that estimates token count from character length.
Uses `chars_per_token` as the average characters per token (default: 4).

This is useful when no model-specific tokenizer is available.

# Fields
- `chars_per_token::Float64`: Average characters per token (default: 4.0).

# Example
```julia
tok = CharacterEstimatorTokenizer()
count_tokens(tok, "Hello, world!")  # ≈ 4
```
"""
Base.@kwdef struct CharacterEstimatorTokenizer <: AbstractTokenizer
    chars_per_token::Float64 = 4.0
end

function count_tokens(tokenizer::CharacterEstimatorTokenizer, text::AbstractString)::Int
    return max(1, ceil(Int, length(text) / tokenizer.chars_per_token))
end

# ── Word Estimator Tokenizer ────────────────────────────────────────────────

"""
    WordEstimatorTokenizer <: AbstractTokenizer

Estimates token count by splitting on whitespace. Roughly 1 word ≈ 1.3 tokens.

# Fields
- `tokens_per_word::Float64`: Multiplier from word count to token count (default: 1.3).
"""
Base.@kwdef struct WordEstimatorTokenizer <: AbstractTokenizer
    tokens_per_word::Float64 = 1.3
end

function count_tokens(tokenizer::WordEstimatorTokenizer, text::AbstractString)::Int
    words = length(split(text))
    return max(1, ceil(Int, words * tokenizer.tokens_per_word))
end
