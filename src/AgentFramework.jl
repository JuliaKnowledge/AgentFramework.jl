# AgentFramework.jl

"""
    AgentFramework

Julia port of Microsoft Agent Framework — a framework for building,
orchestrating, and deploying AI agents.

# Core Types
- [`Message`](@ref): Chat message with role and content items
- [`Content`](@ref): Polymorphic content (text, data, tool calls, etc.)
- [`AgentSession`](@ref): Conversation state container
- [`FunctionTool`](@ref): Tool wrapping a Julia function for LLM use

# Agents
- [`AbstractAgent`](@ref): Base type for all agents
- [`Agent`](@ref): Full-featured agent with middleware and telemetry

# Chat Clients
- [`AbstractChatClient`](@ref): Base type for LLM providers
- [`ChatResponse`](@ref): Response from a chat client
- [`ChatResponseUpdate`](@ref): Streaming update from a chat client
"""
module AgentFramework

using JSON3
using UUIDs
using HTTP
using Dates
using DBInterface
using Logging
using Base64
using SHA
using SQLite
using YAML

# ─── Abstract Type Hierarchy ─────────────────────────────────────────────────

"""Base type for all agents."""
abstract type AbstractAgent end

"""Base type for all chat client implementations."""
abstract type AbstractChatClient end

"""Base type for all content variants."""
abstract type AbstractContent end

"""Base type for all tool types."""
abstract type AbstractTool end

"""Base type for context providers."""
abstract type AbstractContextProvider end

"""Base type for history providers."""
abstract type AbstractHistoryProvider <: AbstractContextProvider end

"""Base type for middleware."""
abstract type AbstractMiddleware end

"""Base type for workflow executors."""
abstract type AbstractExecutor end

# ─── Source Includes ─────────────────────────────────────────────────────────

include("exceptions.jl")
include("content.jl")
include("messages.jl")
include("tokenizer.jl")
include("tools.jl")
include("sessions.jl")
include("retrieval.jl")
include("chat_client.jl")
include("capabilities.jl")
include("middleware.jl")
include("agents.jl")
include("resilience.jl")
include("telemetry.jl")
include("structured_output.jl")
include("serialization.jl")
include("session_store.jl")
include("settings.jl")
include("handoffs.jl")
include("workflows/types.jl")
include("workflows/executor.jl")
include("agent_executor.jl")
include("workflows/edges.jl")
include("workflows/checkpoints.jl")
include("workflows/engine.jl")
include("workflows/builder.jl")
include("workflows/orchestrations.jl")
include("compatibility.jl")
include("workflows/protocol.jl")
include("macros.jl")
include("declarative.jl")
include("multimodal.jl")
include("compaction.jl")
include("content_filter.jl")
include("scoped_state.jl")
include("skills.jl")
include("mcp.jl")
include("evaluation.jl")
include("providers/ollama.jl")
include("providers/memory.jl")
include("providers/openai.jl")
include("providers/foundry.jl")
include("providers/anthropic.jl")

# Stub for DevUI extension (loaded when Genie.jl is available)
"""
    serve_devui(; entities, port, host, auto_open, title, dev_mode)

Start the AgentFramework DevUI server. Requires `Genie.jl` to be loaded.

```julia
using AgentFramework, Genie
serve_devui(entities=[my_agent], port=8080)
```
"""
function serve_devui end
export serve_devui

# ─── Exports (must come before submodule includes so `using ..AgentFramework` works) ──

# Abstract types
export AbstractAgent, AbstractChatClient, AbstractContent, AbstractTool
export AbstractContextProvider, AbstractHistoryProvider
export AbstractMiddleware, AbstractExecutor

# Exceptions
export AgentFrameworkError, AgentError, ChatClientError, ToolError
export MiddlewareError, WorkflowError
export DeclarativeError
export AgentInvalidAuthError, AgentInvalidRequestError, AgentInvalidResponseError
export ChatClientInvalidAuthError, ChatClientInvalidRequestError
export ContentError, ToolExecutionError, UserInputRequiredError

# Content types
export Content, ContentType
export TEXT, TEXT_REASONING, ERROR_CONTENT, URI, DATA, HOSTED_FILE
export FUNCTION_CALL, FUNCTION_RESULT, USAGE, HOSTED_VECTOR_STORE
export text_content, reasoning_content, data_content, uri_content, error_content
export function_call_content, function_result_content
export function_approval_request_content, function_approval_response_content
export to_approval_response, is_approval_request, is_approval_response
export usage_content, hosted_file_content, hosted_vector_store_content
export is_text, is_reasoning, is_function_call, is_function_result
export get_text, parse_arguments
export content_to_dict, content_from_dict
export UsageDetails, add_usage_details, Annotation
export content_type_string, parse_content_type
export detect_media_type_from_base64

# Messages
export Message, Role, normalize_messages, prepend_instructions
export ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER, ROLE_TOOL
export AgentRunInputs
export message_to_dict, message_from_dict
export MessageGroup, group_messages, annotate_message_groups

# Tokenizer
export AbstractTokenizer, count_tokens, count_message_tokens
export CharacterEstimatorTokenizer, WordEstimatorTokenizer

# Tools
export FunctionTool, tool_to_schema, @tool
export invoke_tool, normalize_tools, find_tool
export is_declaration_only, parse_result, reset_invocation_count!

# Sessions
export AgentSession, SessionContext, InMemoryHistoryProvider
export BaseContextProvider, BaseHistoryProvider
export PerServiceCallHistoryMiddleware, with_per_service_call_history
export LOCAL_HISTORY_CONVERSATION_ID, is_local_history_conversation_id
export session_to_dict, session_from_dict
export extend_messages!, extend_instructions!, extend_tools!
export get_all_context_messages
export get_messages, save_messages!, before_run!, after_run!
export ConversationTurn, TurnTracker
export start_turn!, complete_turn!, turn_count, get_turn, last_turn, all_turn_messages
export AbstractMemoryStore, MemoryRecord, MemorySearchResult
export MemoryContextProvider
export InMemoryMemoryStore, FileMemoryStore, SQLiteMemoryStore, RDFMemoryStore
export add_memories!, search_memories, get_memories, clear_memories!, load_ontology!

# Chat client
export ChatOptions, ChatResponse, ChatResponseUpdate
export UsageDetails, FinishReason, ResponseStream
export get_response, get_response_streaming
export merge_chat_options, parse_finish_reason
export AgentResponse, AgentResponseUpdate
export get_final_response
export STOP, LENGTH, TOOL_CALLS, CONTENT_FILTER, FINISH_ERROR
export StreamingToolAccumulator, accumulate_tool_call!, get_accumulated_tool_calls
export reset_accumulator!, has_tool_calls

# Middleware
export AgentContext, ChatContext, FunctionInvocationContext
export AgentMiddlewareFunc, ChatMiddlewareFunc, FunctionMiddlewareFunc
export apply_agent_middleware, apply_chat_middleware, apply_function_middleware
export MiddlewareTermination, terminate_pipeline

# Agents
export Agent, run_agent, run_agent_streaming, create_session
export ChatCompletionAgent, AssistantAgent
export as_tool

# Resilience
export RetryConfig, retry_chat_middleware
export TokenBucketRateLimiter, acquire!, rate_limit_chat_middleware
export with_retry!, with_rate_limit!
export with_instructions, with_tools, with_name, with_options

# Structured output
export StructuredOutput, schema_from_type, response_format_for, parse_structured

# Serialization
export serialize_to_dict, deserialize_from_dict
export serialize_to_json, deserialize_from_json
export serialize_messages, deserialize_messages
export register_type!, register_state_type!

# Session store
export AbstractSessionStore, InMemorySessionStore, FileSessionStore
export load_session, save_session!, delete_session!, list_sessions, has_session

# Settings
export SecretString, Settings
export load_from_env!, load_from_dotenv!, load_from_toml!
export get_setting, get_secret, has_setting

# Handoffs
export HandoffTool, HandoffResult, execute_handoff, handoff_as_function_tool
export normalize_agent_tools

# Providers
export OllamaChatClient
export DBInterfaceHistoryProvider, RedisHistoryProvider, FileHistoryProvider
export get_create_table_sql
export OpenAIChatClient, AzureOpenAIChatClient
export FoundryChatClient, FoundryEmbeddingClient
export AnthropicChatClient

# Workflows — types
export WorkflowMessage, WorkflowMessageType, STANDARD_MESSAGE, RESPONSE_MESSAGE
export WorkflowRunState, WF_STARTED, WF_IN_PROGRESS, WF_IDLE, WF_IDLE_WITH_PENDING_REQUESTS, WF_FAILED, WF_CANCELLED
export WorkflowEvent, WorkflowEventType, WorkflowErrorDetails, WorkflowRunResult
export EVT_STARTED, EVT_STATUS, EVT_FAILED, EVT_OUTPUT, EVT_DATA
export EVT_REQUEST_INFO, EVT_WARNING, EVT_ERROR
export EVT_SUPERSTEP_STARTED, EVT_SUPERSTEP_COMPLETED
export EVT_EXECUTOR_INVOKED, EVT_EXECUTOR_COMPLETED, EVT_EXECUTOR_FAILED
export event_started, event_status, event_failed, event_output
export event_executor_invoked, event_executor_completed, event_executor_failed
export event_superstep_started, event_superstep_completed
export event_request_info, event_warning, event_error
export get_outputs, get_request_info_events, get_final_state

# Workflows — executors
export ExecutorSpec, WorkflowContext
export send_message, yield_output, get_state, set_state!, request_info
export execute_handler, agent_executor

# Agent Executor v2
export AgentExecutor, AgentExecutorRequest, AgentExecutorResponse
export to_executor_spec, handle_request!, handle_string!, handle_message!, handle_response!
export reset!, get_conversation

# Workflows — edges
export EdgeKind, DIRECT_EDGE, FAN_OUT_EDGE, FAN_IN_EDGE
export Edge, EdgeGroup
export direct_edge, fan_out_edge, fan_in_edge, switch_edge
export should_route, route_messages
export source_executor_ids, target_executor_ids

# Workflows — engine & builder
export Workflow, run_workflow
export WorkflowBuilder, add_executor, add_edge, add_fan_out, add_fan_in
export add_switch, add_output, build
export SequentialBuilder, ConcurrentBuilder, GroupChatBuilder, MagenticBuilder
export RoundRobinGroupChat, SelectorGroupChat, MagenticOneGroupChat
export ConcurrentParticipantResult, GroupChatState
export AbstractMagenticManager, StandardMagenticManager
export MagenticTaskLedger, MagenticProgressLedgerItem, MagenticProgressLedger, MagenticContext
export MagenticPlanReviewRequest, MagenticPlanReviewResponse
export with_aggregator, with_selection_func, with_termination, with_plan_review
export magentic_plan, magentic_select, magentic_finalize

# Protocol introspection
export ProtocolDescriptor, TypeCompatibilityResult, WorkflowValidationResult
export get_protocol, can_handle, can_output
export check_type_compatibility, validate_workflow_types, describe_protocol

# Comprehensive workflow validation
export ValidationCheck, CHECK_TYPE_COMPATIBILITY, CHECK_EDGE_DUPLICATION
export CHECK_EXECUTOR_DUPLICATION
export CHECK_GRAPH_CONNECTIVITY, CHECK_SELF_LOOPS, CHECK_OUTPUT_EXECUTORS, CHECK_DEAD_ENDS
export ValidationIssue, FullValidationResult, validate_workflow, ALL_CHECKS

# Macros
export @executor, @handler, @middleware, @pipeline

# Declarative workflows
export workflow_from_dict, workflow_to_dict
export workflow_from_json, workflow_to_json
export workflow_from_yaml, workflow_to_yaml
export workflow_from_file, workflow_to_file
export register_handler!, get_handler, @register_handler
export register_tool!, get_tool
export register_client!, get_client
export register_context_provider!, get_context_provider
export agent_from_dict, agent_to_dict
export agent_from_json, agent_to_json
export agent_from_yaml, agent_to_yaml
export agent_from_file, agent_to_file

# Workflows — checkpoints
export WorkflowCheckpoint, AbstractCheckpointStorage
export InMemoryCheckpointStorage, FileCheckpointStorage

# Multimodal content
export detect_mime_type, is_image_mime, is_audio_mime
export image_content, image_url_content, audio_content, file_content
export base64_to_bytes, content_to_openai_multimodal

# Message compaction
export CompactionStrategy, NO_COMPACTION, SUMMARIZE_OLDEST, DROP_OLDEST, SLIDING_WINDOW
export TRUNCATE, SELECTIVE_TOOL_CALL, TOOL_RESULT_ONLY
export CompactionConfig, CompactionPipeline, compact_messages, needs_compaction
export estimate_tokens, estimate_message_tokens, estimate_messages_tokens

# Content filtering
export ContentFilterSeverity, FILTER_SAFE, FILTER_LOW, FILTER_MEDIUM, FILTER_HIGH
export ContentFilterCategory
export FILTER_HATE, FILTER_SELF_HARM, FILTER_SEXUAL, FILTER_VIOLENCE
export FILTER_PROFANITY, FILTER_JAILBREAK, FILTER_PROTECTED_MATERIAL, FILTER_CUSTOM
export ContentFilterResult, ContentFilterResults, ContentFilteredException
export is_blocked, get_filtered_categories, max_severity
export parse_openai_content_filter

# Capabilities
export Capability, NoCapability
export HasEmbeddings, HasImageGeneration, HasCodeInterpreter
export HasFileSearch, HasWebSearch, HasStreaming, HasStructuredOutput, HasToolCalling
export embedding_capability, image_generation_capability, code_interpreter_capability
export file_search_capability, web_search_capability, streaming_capability
export structured_output_capability, tool_calling_capability
export has_capability, supports_embeddings, supports_image_generation
export supports_code_interpreter, supports_file_search, supports_web_search
export supports_streaming, supports_structured_output, supports_tool_calling
export list_capabilities, require_capability
export get_embeddings, generate_image

# Telemetry
export TelemetrySpan, finish_span!, add_event!, duration_ms
export AbstractTelemetryBackend, LoggingTelemetryBackend, InMemoryTelemetryBackend
export record_span!, get_spans, clear_spans!
export GenAIConventions
export telemetry_agent_middleware, telemetry_chat_middleware, telemetry_function_middleware
export instrument!

# Scoped state
export StateScope, SCOPE_LOCAL, SCOPE_BROADCAST, SCOPE_WORKFLOW
export ScopedValue, ScopedStateStore
export get_local, set_local!, get_broadcast, set_broadcast!
export get_workflow_state, set_workflow_state!
export list_broadcast_keys, list_local_keys, clear_executor_state!

# MCP (Model Context Protocol)
export AbstractMCPClient, StdioMCPClient, HTTPMCPClient
export MCPToolInfo, MCPResource, MCPPrompt, MCPToolResult, MCPServerCapabilities, MCPSpecificApproval
export connect!, list_tools, call_tool, list_resources, read_resource
export list_prompts, get_prompt, close_mcp!, is_connected
export mcp_tool_to_function_tool, mcp_tools_to_function_tools, load_mcp_tools
export with_mcp_client

# Skills
export SkillResource, Skill, SkillsProvider
export AbstractSkillSource, StaticSkillSource, DirectorySkillSource
export DeduplicatingSkillSource, FilteringSkillSource, AggregatingSkillSource
export SkillSourceBuilder, add_directory!, add_skills!, add_source!
export deduplicate!, filter_by!, build, load_skills!, get_skills
export get_resource_content, parse_skill_md, parse_skill_md_content
export discover_skills, add_skill!, add_skills_from_directory!
export SKILL_FILENAME, DEFAULT_SCAN_EXTENSIONS

# Evaluation
export AbstractEvaluator, EvalNotPassedError
export EvalItem, EvalResults, EvalItemResult, EvalScoreResult, ExpectedToolCall, CheckResult
export ConversationSplitter, SPLIT_LAST_TURN, SPLIT_FULL, split_last_turn, split_full
export eval_query, eval_response, split_messages, per_turn_items
export eval_passed, eval_failed, eval_total, all_passed, raise_for_status
export is_error, is_passed, is_failed
export keyword_check, tool_called_check, tool_calls_present, tool_call_args_match
export make_evaluator, LocalEvaluator
export evaluate, evaluate_agent, evaluate_workflow
export WorkflowAgent

# ─── Submodules (consolidated from separate packages) ────────────────────────

include("a2a/A2A.jl")
include("bedrock/Bedrock.jl")
include("coding_agents/CodingAgents.jl")
include("hosting/Hosting.jl")
include("mem0_integration/Mem0Integration.jl")
include("neo4j_integration/Neo4jIntegration.jl")

export A2A, Bedrock, CodingAgents, Hosting, Mem0Integration, Neo4jIntegration

end # module AgentFramework
