module AgentFrameworkAzureIdentityExt

using AgentFramework
using AzureIdentity

function __init__()
    AgentFramework._HAS_AZURE_IDENTITY[] = true
end

function AgentFramework._credential_to_token_provider(
    credential::AzureIdentity.AbstractAzureCredential,
    token_scope::String,
)
    scope = String(strip(token_scope))
    isempty(scope) && throw(
        AgentFramework.ChatClientInvalidAuthError(
            "Azure OpenAI token scope not set. Provide token_scope or set AZURE_OPENAI_TOKEN_SCOPE.",
        ),
    )
    return AzureIdentity.get_bearer_token_provider(credential, scope)
end

function AgentFramework._check_azure_identity_credential(credential, label::String)
    credential isa AzureIdentity.AbstractAzureCredential || throw(
        AgentFramework.ChatClientInvalidAuthError(
            "$label credential must be an AzureIdentity credential or provide token_provider directly.",
        ),
    )
end

function AgentFramework._get_azure_bearer_token_provider(credential, scope::String)
    return AzureIdentity.get_bearer_token_provider(credential, scope)
end

end # module AgentFrameworkAzureIdentityExt
