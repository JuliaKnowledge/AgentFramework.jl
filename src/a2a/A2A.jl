"""
    AgentFramework.A2A

A2A (Agent-to-Agent) remote-agent interoperability protocol for AgentFramework.jl.
Provides client and remote agent types for communicating with A2A-compatible agents.
"""
module A2A

using ..AgentFramework
using Dates
using HTTP
using JSON3
using UUIDs

import ..AgentFramework: before_run!, after_run!, create_session, run_agent, run_agent_streaming

include("exceptions.jl")
include("protocol.jl")
include("utils.jl")
include("converters.jl")
include("client.jl")
include("agent.jl")

export A2AError,
    A2AProtocolError,
    A2ATaskError,
    A2ATimeoutError,
    A2ATaskState,
    A2A_TASK_SUBMITTED,
    A2A_TASK_WORKING,
    A2A_TASK_INPUT_REQUIRED,
    A2A_TASK_AUTH_REQUIRED,
    A2A_TASK_COMPLETED,
    A2A_TASK_FAILED,
    A2A_TASK_CANCELED,
    A2A_TASK_REJECTED,
    A2A_TASK_UNKNOWN,
    A2AContinuationToken,
    A2AAgentCard,
    A2ATaskStatus,
    A2AArtifact,
    A2ATask,
    parse_task_state,
    task_state_string,
    is_terminal_task_state,
    is_in_progress_task_state,
    continuation_token_to_dict,
    continuation_token_from_dict,
    message_to_a2a_dict,
    a2a_message_to_message,
    a2a_agent_card_from_dict,
    a2a_task_from_dict,
    task_to_response,
    A2AClient,
    get_agent_card,
    send_message,
    get_task,
    wait_for_completion,
    A2ARemoteAgent,
    poll_task

end # module A2A
