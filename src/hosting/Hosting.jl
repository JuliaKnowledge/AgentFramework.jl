"""
    AgentFramework.Hosting

Local hosted runtime and durable workflow execution for AgentFramework.jl.
Provides `HostedRuntime` for running agents and workflows with persistence.
"""
module Hosting

using ..AgentFramework
using Dates
using HTTP
using JSON3
using UUIDs

include("types.jl")
include("runtime.jl")
include("server.jl")

export HostedRuntime, HostedWorkflowRun
export AbstractHostedRunStore, InMemoryHostedRunStore, FileHostedRunStore
export register_agent!, register_workflow!
export list_registered_agents, list_registered_workflows
export run_agent!, get_agent_session, list_agent_sessions, delete_agent_session!
export start_workflow_run!, resume_workflow_run!, get_workflow_run, list_workflow_runs
export hosted_workflow_run_to_dict
export handle_request, serve

end # module Hosting
