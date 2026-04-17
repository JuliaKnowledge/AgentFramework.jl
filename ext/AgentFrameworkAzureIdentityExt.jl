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

function AgentFramework._check_azure_identity_credential(
        credential::AzureIdentity.AbstractAzureCredential, label::String)
    return nothing
end

function AgentFramework._get_azure_bearer_token_provider(
        credential::AzureIdentity.AbstractAzureCredential, scope::String)
    return AzureIdentity.get_bearer_token_provider(credential, scope)
end

end # module AgentFrameworkAzureIdentityExt
