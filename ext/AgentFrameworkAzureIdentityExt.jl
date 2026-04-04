module AgentFrameworkAzureIdentityExt

using AgentFramework
using AzureIdentity

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

end # module AgentFrameworkAzureIdentityExt
