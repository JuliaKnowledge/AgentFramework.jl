"""
    AgentFramework.Bedrock

Amazon Bedrock chat and embeddings support for AgentFramework.jl.
Provides `BedrockChatClient` and `BedrockEmbeddingClient` for AWS Bedrock models.
"""
module Bedrock

using ..AgentFramework
using Dates
using HTTP
using JSON3
using Logging
using SHA
using UUIDs

import ..AgentFramework: get_embeddings, get_response, get_response_streaming
import ..AgentFramework: streaming_capability, tool_calling_capability

include("auth.jl")
include("chat_client.jl")
include("embedding_client.jl")

export BedrockCredentials, BedrockChatClient, BedrockEmbeddingClient

end # module Bedrock
