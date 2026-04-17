using AgentFramework
using Test

include("test_content.jl")
include("test_messages.jl")
include("test_tools.jl")
include("test_sessions.jl")
include("test_chat_client.jl")
include("test_middleware.jl")
include("test_agents.jl")
include("test_structured_output.jl")
include("test_serialization.jl")
include("test_handoffs.jl")
include("test_workflows.jl")
include("test_orchestrations.jl")
include("test_compatibility_helpers.jl")
include("test_memory_providers.jl")
include("test_memory_retrieval.jl")
include("test_openai_provider.jl")
if Base.find_package("AzureIdentity") !== nothing
    include("test_foundry_provider.jl")
end
include("test_anthropic_provider.jl")
include("test_tool_macro.jl")
include("test_multimodal.jl")
include("test_compaction.jl")
include("test_compaction_v2.jl")
include("test_content_filter.jl")
include("test_scoped_state.jl")
include("test_mcp.jl")
include("test_agent_executor.jl")
include("test_protocol.jl")
include("test_telemetry.jl")
include("test_workflow_validation.jl")
include("test_macros.jl")
include("test_declarative.jl")
include("test_capabilities.jl")
include("test_skills.jl")
include("test_resilience.jl")
include("test_extras.jl")
include("test_feature_stage.jl")
include("test_session_store.jl")
include("test_settings.jl")
include("test_evaluation.jl")
include("test_approval.jl")
include("test_parity_features.jl")

# AzureIdentity extension test (conditional — only if AzureIdentity is available)
if Base.find_package("AzureIdentity") !== nothing
    include("test_azure_extension.jl")
end

# Mem0.jl (local) extension test (conditional — only if Mem0.jl is dev'd)
if Base.find_package("Mem0") !== nothing
    include("test_local_mem0_ext.jl")
end

# Submodule tests
include("test_a2a.jl")
include("test_bedrock.jl")
include("test_coding_agents.jl")
include("test_hosting.jl")
include("test_mem0_integration.jl")
include("test_neo4j.jl")

# Integration tests (require Ollama running)
if get(ENV, "AGENTFRAMEWORK_INTEGRATION", "false") == "true"
    include("test_ollama_integration.jl")
end

if get(ENV, "AGENTFRAMEWORK_RDFLIB_TEST", "false") == "true" &&
   (Base.find_package("RDFLib") !== nothing || isdefined(Main, :RDFLib))
    include("test_rdflib_memory.jl")
end
