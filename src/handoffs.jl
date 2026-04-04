# Agent-to-agent handoff support for AgentFramework.jl
# Allows agents to delegate to other agents as tools.

"""
    HandoffTool <: AbstractTool

A tool that hands off conversation to another agent. When an agent calls this tool,
the target agent takes over processing with the conversation context.

# Fields
- `name::String`: Tool name the LLM will call (e.g., "transfer_to_billing").
- `description::String`: Description of when to use this handoff.
- `target::Agent`: The agent to hand off to.
- `transfer_instructions::String`: Optional instructions for the target agent.
- `include_history::Bool`: Whether to include conversation history (default: true).

# Examples
```julia
billing_agent = Agent(name="BillingAgent", client=client, instructions="Handle billing queries.")
support_agent = Agent(
    name = "SupportAgent",
    client = client,
    instructions = "Route billing questions to billing.",
    tools = [
        HandoffTool(
            name = "transfer_to_billing",
            description = "Transfer the conversation to the billing specialist.",
            target = billing_agent,
        ),
    ],
)
```
"""
Base.@kwdef struct HandoffTool <: AbstractTool
    name::String
    description::String
    target::Agent
    transfer_instructions::String = ""
    include_history::Bool = true
end

function Base.show(io::IO, h::HandoffTool)
    print(io, "HandoffTool(\"", h.name, "\" → \"", h.target.name, "\")")
end

"""
    tool_to_schema(tool::HandoffTool) -> Dict{String, Any}

Generate the function calling schema for a handoff tool.
Handoff tools take no parameters — the LLM simply calls them to transfer.
"""
function tool_to_schema(tool::HandoffTool)::Dict{String, Any}
    Dict{String, Any}(
        "type" => "function",
        "function" => Dict{String, Any}(
            "name" => tool.name,
            "description" => tool.description,
            "parameters" => Dict{String, Any}(
                "type" => "object",
                "properties" => Dict{String, Any}(
                    "message" => Dict{String, Any}(
                        "type" => "string",
                        "description" => "Optional message to include with the handoff.",
                    ),
                ),
            ),
        ),
    )
end

"""
    HandoffResult

Result of an agent handoff, containing the target agent's response
and metadata about the transfer.

# Fields
- `source_agent::String`: Name of the agent that initiated the handoff.
- `target_agent::String`: Name of the agent that handled the request.
- `response::AgentResponse`: The target agent's response.
- `handoff_message::String`: Any message passed during handoff.
"""
struct HandoffResult
    source_agent::String
    target_agent::String
    response::AgentResponse
    handoff_message::String
end

function Base.show(io::IO, h::HandoffResult)
    print(io, "HandoffResult(", h.source_agent, " → ", h.target_agent, ")")
end

"""
    execute_handoff(tool::HandoffTool, messages::Vector{Message}, handoff_message::String="") -> AgentResponse

Execute a handoff by running the target agent with the conversation context.
"""
function execute_handoff(
    tool::HandoffTool,
    messages::Vector{Message};
    handoff_message::String = "",
    session::Union{Nothing, AgentSession} = nothing,
)::AgentResponse
    # Build context for target agent
    target_messages = Message[]

    if tool.include_history
        # Include relevant conversation history (skip system messages from source agent)
        for msg in messages
            if msg.role != :system
                push!(target_messages, msg)
            end
        end
    end

    # Add transfer context
    if !isempty(tool.transfer_instructions)
        # The target agent's own instructions will be prepended by run_agent,
        # but we can add transfer-specific context
        push!(target_messages, Message(:user, tool.transfer_instructions))
    end

    if !isempty(handoff_message)
        push!(target_messages, Message(:user, handoff_message))
    end

    # If no messages were added, provide a minimal prompt
    if isempty(target_messages)
        push!(target_messages, Message(:user, "Continue the conversation."))
    end

    # Run the target agent
    return run_agent(tool.target, target_messages; session=session)
end

# ── Integration with Agent Tool System ───────────────────────────────────────

"""
    handoff_as_function_tool(tool::HandoffTool) -> FunctionTool

Convert a HandoffTool to a FunctionTool that can be used in the standard
tool execution pipeline. The function captures the handoff tool and returns
the target agent's text response.
"""
function handoff_as_function_tool(tool::HandoffTool)::FunctionTool
    FunctionTool(
        name = tool.name,
        description = tool.description,
        func = (; message="") -> begin
            # We return a marker that the agent loop can detect
            response = execute_handoff(tool, Message[]; handoff_message=string(message))
            return response.text
        end,
        parameters = Dict{String, Any}(
            "type" => "object",
            "properties" => Dict{String, Any}(
                "message" => Dict{String, Any}(
                    "type" => "string",
                    "description" => "Optional message to include with the handoff.",
                ),
            ),
        ),
    )
end

"""
    normalize_agent_tools(tools::Vector) -> Tuple{Vector{FunctionTool}, Vector{HandoffTool}}

Separate and normalize a mixed vector of FunctionTools and HandoffTools.
HandoffTools are also converted to FunctionTools for the LLM schema.
Returns (all_function_tools, handoff_tools).
"""
function normalize_agent_tools(tools::Vector)
    function_tools = FunctionTool[]
    handoff_tools = HandoffTool[]

    for t in tools
        if t isa FunctionTool
            push!(function_tools, t)
        elseif t isa HandoffTool
            push!(handoff_tools, t)
            push!(function_tools, handoff_as_function_tool(t))
        end
    end

    return (function_tools, handoff_tools)
end
