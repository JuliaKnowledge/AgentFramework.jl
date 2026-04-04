# Message compaction for AgentFramework.jl
# Implements message history compression to stay within token limits.
# Mirrors Python's CompactionStrategy and token management.

"""
    CompactionStrategy

Enum for message compaction strategies.
"""
@enum CompactionStrategy begin
    NO_COMPACTION           # Never compact
    SUMMARIZE_OLDEST        # Replace oldest messages with a summary
    DROP_OLDEST             # Drop oldest messages (keep system + recent)
    SLIDING_WINDOW          # Keep a fixed window of recent messages
    TRUNCATE                # Hard truncation from beginning (can split within a message)
    SELECTIVE_TOOL_CALL     # Drop tool call/result pairs selectively
    TOOL_RESULT_ONLY        # Replace tool results with "[Tool result truncated]"
end

"""
    CompactionConfig

Configuration for message compaction.

# Fields
- `strategy::CompactionStrategy`: Which strategy to use.
- `max_tokens::Int`: Approximate token budget for message history.
- `keep_system::Bool`: Always keep system/developer messages.
- `keep_recent::Int`: Number of recent messages to always keep.
- `summary_prompt::String`: Prompt template for summarization (used with SUMMARIZE_OLDEST).
- `chars_per_token::Float64`: Approximate chars-per-token ratio for heuristic counting.
"""
Base.@kwdef mutable struct CompactionConfig
    strategy::CompactionStrategy = SLIDING_WINDOW
    max_tokens::Int = 4096
    keep_system::Bool = true
    keep_recent::Int = 4
    summary_prompt::String = "Summarize the following conversation concisely, preserving key facts and decisions:\n\n"
    chars_per_token::Float64 = 4.0
end

# ── Token Counting Heuristic ────────────────────────────────────────────────

"""
    estimate_tokens(text::AbstractString; chars_per_token=4.0) -> Int

Estimate token count using a character-based heuristic.
Default ratio of 4 chars/token is reasonable for English text with GPT-style tokenizers.
"""
function estimate_tokens(text::AbstractString; chars_per_token::Float64=4.0)::Int
    return max(1, ceil(Int, length(text) / chars_per_token))
end

"""
    estimate_message_tokens(msg::Message; chars_per_token=4.0) -> Int

Estimate total tokens for a message, including role overhead.
"""
function estimate_message_tokens(msg::Message; chars_per_token::Float64=4.0)::Int
    tokens = 4  # role + formatting overhead
    for c in msg.contents
        if c.type == TEXT && c.text !== nothing
            tokens += estimate_tokens(c.text; chars_per_token)
        elseif c.type == FUNCTION_CALL
            c.name !== nothing && (tokens += estimate_tokens(c.name; chars_per_token))
            c.arguments !== nothing && (tokens += estimate_tokens(c.arguments; chars_per_token))
        elseif c.type == FUNCTION_RESULT
            r = c.result
            r !== nothing && (tokens += estimate_tokens(string(r); chars_per_token))
        end
    end
    return tokens
end

"""
    estimate_messages_tokens(messages::Vector{Message}; chars_per_token=4.0) -> Int

Estimate total tokens for a list of messages.
"""
function estimate_messages_tokens(messages::Vector{Message}; chars_per_token::Float64=4.0)::Int
    return sum(estimate_message_tokens(m; chars_per_token) for m in messages; init=0)
end

# ── Compaction Functions ─────────────────────────────────────────────────────

"""
    compact_messages(messages::Vector{Message}, config::CompactionConfig) -> Vector{Message}

Apply compaction strategy to reduce message history within token budget.
Returns a new vector — does not mutate the input.
"""
function compact_messages(messages::Vector{Message}, config::CompactionConfig)::Vector{Message}
    config.strategy == NO_COMPACTION && return copy(messages)
    
    total_tokens = estimate_messages_tokens(messages; chars_per_token=config.chars_per_token)
    total_tokens <= config.max_tokens && return copy(messages)
    
    if config.strategy == DROP_OLDEST
        return _compact_drop_oldest(messages, config)
    elseif config.strategy == SLIDING_WINDOW
        return _compact_sliding_window(messages, config)
    elseif config.strategy == SUMMARIZE_OLDEST
        return _compact_summarize_oldest(messages, config)
    elseif config.strategy == TRUNCATE
        return _compact_truncate(messages, config)
    elseif config.strategy == SELECTIVE_TOOL_CALL
        return _compact_selective_tool_call(messages, config)
    elseif config.strategy == TOOL_RESULT_ONLY
        return _compact_tool_result_only(messages, config)
    else
        return copy(messages)
    end
end

"""
    needs_compaction(messages::Vector{Message}, config::CompactionConfig) -> Bool

Check if message history exceeds the token budget.
"""
function needs_compaction(messages::Vector{Message}, config::CompactionConfig)::Bool
    config.strategy == NO_COMPACTION && return false
    return estimate_messages_tokens(messages; chars_per_token=config.chars_per_token) > config.max_tokens
end

# ── Strategy Implementations ─────────────────────────────────────────────────

function _compact_drop_oldest(messages::Vector{Message}, config::CompactionConfig)::Vector{Message}
    system_msgs, other_msgs = _split_system_messages(messages, config.keep_system)
    
    # Keep the last keep_recent messages from other_msgs
    if length(other_msgs) <= config.keep_recent
        return vcat(system_msgs, other_msgs)
    end
    
    # Drop from the front of other_msgs until within budget
    kept = other_msgs[end-config.keep_recent+1:end]
    remaining_budget = config.max_tokens - estimate_messages_tokens(vcat(system_msgs, kept); chars_per_token=config.chars_per_token)
    
    # Add back older messages that fit
    idx = length(other_msgs) - config.keep_recent
    result_middle = Message[]
    while idx >= 1 && remaining_budget > 0
        msg_tokens = estimate_message_tokens(other_msgs[idx]; chars_per_token=config.chars_per_token)
        if msg_tokens <= remaining_budget
            pushfirst!(result_middle, other_msgs[idx])
            remaining_budget -= msg_tokens
        else
            break
        end
        idx -= 1
    end
    
    return vcat(system_msgs, result_middle, kept)
end

function _compact_sliding_window(messages::Vector{Message}, config::CompactionConfig)::Vector{Message}
    system_msgs, other_msgs = _split_system_messages(messages, config.keep_system)
    
    system_tokens = estimate_messages_tokens(system_msgs; chars_per_token=config.chars_per_token)
    budget = config.max_tokens - system_tokens
    
    # Take messages from the end until budget exhausted
    result = Message[]
    tokens_used = 0
    for i in length(other_msgs):-1:1
        msg_tokens = estimate_message_tokens(other_msgs[i]; chars_per_token=config.chars_per_token)
        if tokens_used + msg_tokens <= budget
            pushfirst!(result, other_msgs[i])
            tokens_used += msg_tokens
        else
            break
        end
    end
    
    return vcat(system_msgs, result)
end

function _compact_summarize_oldest(messages::Vector{Message}, config::CompactionConfig)::Vector{Message}
    system_msgs, other_msgs = _split_system_messages(messages, config.keep_system)
    
    if length(other_msgs) <= config.keep_recent
        return vcat(system_msgs, other_msgs)
    end
    
    # Split into old (to summarize) and recent (to keep)
    n_recent = min(config.keep_recent, length(other_msgs))
    old_msgs = other_msgs[1:end-n_recent]
    recent_msgs = other_msgs[end-n_recent+1:end]
    
    # Build a summary of old messages (just concatenate texts — actual LLM summarization
    # would require a chat client which is handled at the agent level)
    summary_parts = String[]
    for msg in old_msgs
        role_str = msg.role == :user ? "User" : msg.role == :assistant ? "Assistant" : string(msg.role)
        text_parts = [something(c.text, "") for c in msg.contents if c.type == TEXT && c.text !== nothing]
        isempty(text_parts) && continue
        push!(summary_parts, "$role_str: " * join(text_parts, " "))
    end
    
    if !isempty(summary_parts)
        summary_text = "[Conversation summary: " * join(summary_parts, " | ") * "]"
        # Truncate if still too long
        max_summary_chars = ceil(Int, config.max_tokens * config.chars_per_token * 0.3)
        if length(summary_text) > max_summary_chars
            summary_text = summary_text[1:max_summary_chars] * "...]"
        end
        summary_msg = Message(role=:system, contents=[text_content(summary_text)])
        return vcat(system_msgs, [summary_msg], recent_msgs)
    else
        return vcat(system_msgs, recent_msgs)
    end
end

# ── Helpers ──────────────────────────────────────────────────────────────────

function _split_system_messages(messages::Vector{Message}, keep_system::Bool)
    if !keep_system
        return Message[], copy(messages)
    end
    system = Message[]
    other = Message[]
    for msg in messages
        if msg.role == :system || msg.role == :developer
            push!(system, msg)
        else
            push!(other, msg)
        end
    end
    return system, other
end

# ── New Strategy Implementations ─────────────────────────────────────────────

"""
    _compact_truncate(messages, config) -> Vector{Message}

Hard truncation: keep system messages + take messages from end until budget is
reached.  Unlike SLIDING_WINDOW, this strategy can split *within* the oldest
kept message by truncating its text content to fit the remaining budget.
"""
function _compact_truncate(messages::Vector{Message}, config::CompactionConfig)::Vector{Message}
    system_msgs, other_msgs = _split_system_messages(messages, config.keep_system)
    isempty(other_msgs) && return copy(system_msgs)

    system_tokens = estimate_messages_tokens(system_msgs; chars_per_token=config.chars_per_token)
    budget = config.max_tokens - system_tokens
    budget <= 0 && return copy(system_msgs)

    # Collect whole messages from the end
    result = Message[]
    tokens_used = 0
    split_idx = 0  # index in other_msgs of the message we might truncate

    for i in length(other_msgs):-1:1
        msg_tokens = estimate_message_tokens(other_msgs[i]; chars_per_token=config.chars_per_token)
        if tokens_used + msg_tokens <= budget
            pushfirst!(result, other_msgs[i])
            tokens_used += msg_tokens
        else
            split_idx = i
            break
        end
    end

    # Try to partially include the boundary message
    if split_idx > 0 && tokens_used < budget
        remaining_tokens = budget - tokens_used
        boundary_msg = other_msgs[split_idx]
        # Only truncate TEXT content
        new_contents = Content[]
        for c in boundary_msg.contents
            if c.type == TEXT && c.text !== nothing
                max_chars = floor(Int, remaining_tokens * config.chars_per_token) - 16  # leave room for overhead
                if max_chars > 0
                    truncated_text = length(c.text) > max_chars ? c.text[1:max_chars] * "…" : c.text
                    push!(new_contents, text_content(truncated_text))
                end
            end
        end
        if !isempty(new_contents)
            truncated_msg = Message(role=boundary_msg.role, contents=new_contents)
            pushfirst!(result, truncated_msg)
        end
    end

    return vcat(system_msgs, result)
end

"""
    _compact_selective_tool_call(messages, config) -> Vector{Message}

Remove complete tool-call / tool-result pairs from older messages, keeping the
most recent tool interactions.  Conversational (non-tool) messages are never
removed by this strategy.

Logic:
- Identify tool-related messages (those whose *only* content is FUNCTION_CALL
  or FUNCTION_RESULT).
- Group them into (call, result) pairs by `call_id`.
- Protect pairs that appear within the last `keep_recent` messages.
- Drop the oldest pairs first until token budget is satisfied.
"""
function _compact_selective_tool_call(messages::Vector{Message}, config::CompactionConfig)::Vector{Message}
    system_msgs, other_msgs = _split_system_messages(messages, config.keep_system)
    isempty(other_msgs) && return copy(system_msgs)

    n = length(other_msgs)
    protected_start = max(1, n - config.keep_recent + 1)

    # Build list of (call_msg_idx, result_msg_idx) pairs ordered by call position
    pairs = Tuple{Int,Int}[]
    call_index = Dict{String,Int}()  # call_id → index in other_msgs

    for (i, msg) in enumerate(other_msgs)
        for c in msg.contents
            if c.type == FUNCTION_CALL && c.call_id !== nothing
                call_index[c.call_id] = i
            elseif c.type == FUNCTION_RESULT && c.call_id !== nothing
                if haskey(call_index, c.call_id)
                    push!(pairs, (call_index[c.call_id], i))
                end
            end
        end
    end

    # Separate removable (older) pairs from protected (recent) pairs
    removable = [(ci, ri) for (ci, ri) in pairs if ci < protected_start && ri < protected_start]

    # Sort removable pairs oldest-first (by call index)
    sort!(removable; by=first)

    # Greedily remove oldest pairs until under budget
    removed_indices = Set{Int}()
    current_tokens = estimate_messages_tokens(vcat(system_msgs, other_msgs); chars_per_token=config.chars_per_token)

    for (ci, ri) in removable
        current_tokens <= config.max_tokens && break
        if !(ci in removed_indices) && !(ri in removed_indices)
            current_tokens -= estimate_message_tokens(other_msgs[ci]; chars_per_token=config.chars_per_token)
            current_tokens -= estimate_message_tokens(other_msgs[ri]; chars_per_token=config.chars_per_token)
            push!(removed_indices, ci)
            push!(removed_indices, ri)
        end
    end

    kept = [other_msgs[i] for i in 1:n if !(i in removed_indices)]
    return vcat(system_msgs, kept)
end

"""
    _compact_tool_result_only(messages, config) -> Vector{Message}

Replace FUNCTION_RESULT content with a short placeholder while keeping the
FUNCTION_CALL itself.  This preserves the conversation flow (the model can see
it called a tool) without the verbose result payload.

Only results in messages *outside* the last `keep_recent` messages are replaced.
"""
function _compact_tool_result_only(messages::Vector{Message}, config::CompactionConfig)::Vector{Message}
    system_msgs, other_msgs = _split_system_messages(messages, config.keep_system)
    isempty(other_msgs) && return copy(system_msgs)

    n = length(other_msgs)
    protected_start = max(1, n - config.keep_recent + 1)

    result = Message[]
    for (i, msg) in enumerate(other_msgs)
        if i < protected_start && any(c -> c.type == FUNCTION_RESULT, msg.contents)
            new_contents = Content[]
            for c in msg.contents
                if c.type == FUNCTION_RESULT
                    push!(new_contents, function_result_content(
                        something(c.call_id, ""),
                        "[Tool result truncated]"
                    ))
                else
                    push!(new_contents, c)
                end
            end
            push!(result, Message(role=msg.role, contents=new_contents))
        else
            push!(result, msg)
        end
    end

    return vcat(system_msgs, result)
end

# ── Strategy Composition Pipeline ────────────────────────────────────────────

"""
    CompactionPipeline

Applies multiple compaction strategies in sequence. Each strategy operates
on the output of the previous one.
"""
Base.@kwdef struct CompactionPipeline
    strategies::Vector{CompactionConfig}
end

"""
    compact_messages(messages::Vector{Message}, pipeline::CompactionPipeline) -> Vector{Message}

Apply a pipeline of compaction strategies in sequence.
"""
function compact_messages(messages::Vector{Message}, pipeline::CompactionPipeline)::Vector{Message}
    result = copy(messages)
    for config in pipeline.strategies
        result = compact_messages(result, config)
    end
    return result
end
