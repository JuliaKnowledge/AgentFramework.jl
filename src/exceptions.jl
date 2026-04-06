# Exception hierarchy for AgentFramework.jl
# Mirrors the Python agent_framework.exceptions module structure.

"""
    AgentFrameworkError <: Exception

Base exception for the Agent Framework. All framework-specific errors derive from this.

# Fields
- `message::String`: Human-readable error description.
- `inner::Union{Nothing, Exception}`: Optional wrapped cause.
"""
struct AgentFrameworkError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
AgentFrameworkError(msg::String) = AgentFrameworkError(msg, nothing)

function Base.showerror(io::IO, e::AgentFrameworkError)
    print(io, nameof(typeof(e)), ": ", e.message)
    if e.inner !== nothing
        print(io, "\n  caused by: ")
        showerror(io, e.inner)
    end
end

# ── Agent Exceptions ─────────────────────────────────────────────────────────

"""Base class for all agent exceptions."""
struct AgentError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
AgentError(msg::String) = AgentError(msg, nothing)

struct AgentInvalidAuthError <: Exception
    message::String
end

struct AgentInvalidRequestError <: Exception
    message::String
end

struct AgentInvalidResponseError <: Exception
    message::String
end

struct AgentContentFilterError <: Exception
    message::String
end

# ── Chat Client Exceptions ───────────────────────────────────────────────────

"""Base class for all chat client exceptions."""
struct ChatClientError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
ChatClientError(msg::String) = ChatClientError(msg, nothing)

struct ChatClientInvalidAuthError <: Exception
    message::String
end

struct ChatClientInvalidRequestError <: Exception
    message::String
end

struct ChatClientInvalidResponseError <: Exception
    message::String
end

struct ChatClientContentFilterError <: Exception
    message::String
end

# ── Content Exceptions ───────────────────────────────────────────────────────

struct ContentError <: Exception
    message::String
end

# ── Tool Exceptions ──────────────────────────────────────────────────────────

"""Base class for all tool-related exceptions."""
struct ToolError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
ToolError(msg::String) = ToolError(msg, nothing)

struct ToolExecutionError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
ToolExecutionError(msg::String) = ToolExecutionError(msg, nothing)

"""
    UserInputRequiredError

Raised when a tool wrapping a sub-agent requires user input to proceed.
Carries the request contents so the parent can propagate them.
"""
struct UserInputRequiredError <: Exception
    message::String
    contents::Vector{Any}
end
UserInputRequiredError(contents::Vector) = UserInputRequiredError("Tool requires user input to proceed.", contents)

# ── Middleware Exceptions ────────────────────────────────────────────────────

struct MiddlewareError <: Exception
    message::String
end

# ── Declarative Exceptions ─────────────────────────────────────────────────────

struct DeclarativeError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
DeclarativeError(msg::String) = DeclarativeError(msg, nothing)

function Base.showerror(io::IO, e::DeclarativeError)
    print(io, nameof(typeof(e)), ": ", e.message)
    if e.inner !== nothing
        print(io, "\n  caused by: ")
        showerror(io, e.inner)
    end
end

# ── Workflow Exceptions ──────────────────────────────────────────────────────

struct WorkflowError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
WorkflowError(msg::String) = WorkflowError(msg, nothing)

struct WorkflowRunnerError <: Exception
    message::String
end

struct WorkflowConvergenceError <: Exception
    message::String
end

struct WorkflowCheckpointError <: Exception
    message::String
end
