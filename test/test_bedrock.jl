using AgentFramework
using AgentFramework.Bedrock
using Dates
using HTTP
using JSON3
using Test

function _restore_env!(saved)
    for (name, value) in saved
        if value === nothing
            pop!(ENV, name, nothing)
        else
            ENV[name] = value
        end
    end
    return nothing
end

include("bedrock/test_auth.jl")
include("bedrock/test_chat_client.jl")
include("bedrock/test_embedding_client.jl")
