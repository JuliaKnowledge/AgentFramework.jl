"""
    AgentFramework.CodingAgents

CLI-backed GitHub Copilot and Claude Code providers for AgentFramework.jl.
Provides `GitHubCopilotChatClient` and `ClaudeCodeChatClient`.
"""
module CodingAgents

using ..AgentFramework
using JSON3
using Logging

import ..AgentFramework: get_response, get_response_streaming
import ..AgentFramework: code_interpreter_capability, file_search_capability
import ..AgentFramework: streaming_capability, structured_output_capability, web_search_capability
import ..AgentFramework: ChatClientInvalidResponseError

include("common.jl")
include("github_copilot.jl")
include("claude_code.jl")

export GitHubCopilotChatClient, ClaudeCodeChatClient

end # module CodingAgents
