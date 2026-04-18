# Auto-attached docstrings for symbols whose definitions are concise enough
# that an inline docstring would clutter the source. Each `@doc` block here
# documents an exported public symbol so that `checkdocs = :exports` in the
# documentation build does not flag a missing-docstring warning.

# ── Capabilities (singleton trait dispatchers) ───────────────────────────────

@doc "Trait dispatcher returning the embedding capability of a chat client type. Returns `NoCapability()` by default; provider implementations should override to return a concrete `Capability` singleton when supported." embedding_capability
@doc "Trait dispatcher returning the image-generation capability of a chat client type. Defaults to `NoCapability()`." image_generation_capability
@doc "Trait dispatcher returning the code-interpreter capability of a chat client type. Defaults to `NoCapability()`." code_interpreter_capability
@doc "Trait dispatcher returning the file-search capability of a chat client type. Defaults to `NoCapability()`." file_search_capability
@doc "Trait dispatcher returning the web-search capability of a chat client type. Defaults to `NoCapability()`." web_search_capability
@doc "Trait dispatcher returning the streaming capability of a chat client type. Defaults to `NoCapability()`." streaming_capability
@doc "Trait dispatcher returning the structured-output capability of a chat client type. Defaults to `NoCapability()`." structured_output_capability
@doc "Trait dispatcher returning the tool-calling capability of a chat client type. Defaults to `NoCapability()`." tool_calling_capability

@doc "Return `true` if `client` supports text embeddings (i.e. `embedding_capability(typeof(client))` is not `NoCapability`)." supports_embeddings
@doc "Return `true` if `client` supports image generation." supports_image_generation
@doc "Return `true` if `client` supports code interpretation/execution." supports_code_interpreter
@doc "Return `true` if `client` supports file search." supports_file_search
@doc "Return `true` if `client` supports web search." supports_web_search
@doc "Return `true` if `client` supports streaming responses." supports_streaming
@doc "Return `true` if `client` supports structured output (JSON-schema-constrained responses)." supports_structured_output
@doc "Return `true` if `client` supports tool/function calling." supports_tool_calling

# ── Memory store API ─────────────────────────────────────────────────────────

@doc "Abstract supertype for pluggable memory stores used by `MemoryContextProvider`. Concrete subtypes implement `add_memories!`, `search_memories`, `get_memories`, and `clear_memories!`." AbstractMemoryStore
@doc "In-process implementation of `AbstractMemoryStore` that holds memories in a `Vector` keyed by session id. Suitable for tests and short-lived agents." InMemoryMemoryStore
@doc "File-backed `AbstractMemoryStore` that persists memories as JSON on disk. Each session id maps to a separate file under `path`." FileMemoryStore
@doc "SQLite-backed `AbstractMemoryStore`. Provides durable storage with simple substring search; ideal for single-process production deployments." SQLiteMemoryStore
@doc "Add one or more memory entries to the store, scoped by session id (and optional metadata). Implementations should be idempotent for identical entries." add_memories!
@doc "Search the store for memories matching `query`, optionally scoped by session id; returns a vector of memory entries ranked by relevance." search_memories
@doc "Return all memory entries for a given session id." get_memories
@doc "Remove all memory entries for a given session id." clear_memories!
@doc "`BaseContextProvider` that injects memory entries from any `AbstractMemoryStore` into agent prompts. Provides a uniform memory layer across in-memory, file, SQLite, Mem0, Graphiti, and MemPalace backends." MemoryContextProvider

# ── Workflow orchestration builders ──────────────────────────────────────────

@doc "Builder for sequential workflow orchestration: executors run one after another, each receiving the previous step's output." SequentialBuilder
@doc "Builder for concurrent workflow orchestration: multiple executors run in parallel and their results are aggregated." ConcurrentBuilder
@doc "Per-participant result emitted by a `ConcurrentBuilder` workflow, capturing executor identity, output, and any error." ConcurrentParticipantResult
@doc "Configure a custom aggregator function or executor for a `ConcurrentBuilder` workflow." with_aggregator
@doc "Builder for group-chat orchestration: multiple agents take turns based on a selection function until a termination condition is met." GroupChatBuilder
@doc "Mutable state struct tracking the current participant, message history, and termination flag in a group-chat workflow." GroupChatState
@doc "Configure the participant-selection function for a `GroupChatBuilder` workflow. The function receives the current `GroupChatState` and returns the next participant id." with_selection_func
@doc "Configure a termination predicate for a `GroupChatBuilder` workflow. The function receives the current `GroupChatState` and returns `true` to stop the chat." with_termination

@doc "Abstract supertype for Magentic-style orchestration managers. Concrete subtypes implement `magentic_plan`, `magentic_select`, and `magentic_finalize`." AbstractMagenticManager
@doc "Default `AbstractMagenticManager` implementation that delegates planning, selection, and finalization to an LLM via configurable prompt templates." StandardMagenticManager
@doc "Builder for Magentic orchestration: a planner LLM coordinates a team of executors with optional human-in-the-loop plan review." MagenticBuilder
@doc "Mutable context struct passed through Magentic orchestration, holding the task ledger, progress ledger, and conversation history." MagenticContext
@doc "Plan-review request emitted when `with_plan_review(true)` is enabled, containing the proposed plan for human approval." MagenticPlanReviewRequest
@doc "Plan-review response containing the human's approval decision and optional revisions." MagenticPlanReviewResponse
@doc "Per-step entry in a Magentic progress ledger." MagenticProgressLedgerItem
@doc "Aggregate progress ledger tracking completed and pending steps in a Magentic workflow." MagenticProgressLedger
@doc "Top-level task ledger holding the original goal, decomposition, and overall status of a Magentic workflow." MagenticTaskLedger
@doc "Produce a plan for the given task using the configured planner LLM and ledgers." magentic_plan
@doc "Select the next executor to invoke based on the current Magentic context." magentic_select
@doc "Finalize a Magentic workflow, producing the final answer from the accumulated ledger." magentic_finalize
@doc "Enable or disable human-in-the-loop plan review for a `MagenticBuilder` workflow." with_plan_review

# ── Workflow event constructors ──────────────────────────────────────────────

@doc "Construct a workflow event signaling the workflow has started." event_started
@doc "Construct a workflow event reporting an intermediate status update." event_status
@doc "Construct a workflow event signaling the workflow has failed with an error." event_failed
@doc "Construct a workflow event carrying a final output value." event_output
@doc "Construct a workflow event carrying an arbitrary error payload." event_error
@doc "Construct a workflow event indicating the workflow paused to request external info (e.g. human input)." event_request_info
@doc "Construct a workflow event reporting a non-fatal warning." event_warning
@doc "Construct a workflow event signaling a new superstep has started." event_superstep_started
@doc "Construct a workflow event signaling the current superstep has completed." event_superstep_completed
@doc "Construct a workflow event signaling that an executor has been invoked." event_executor_invoked
@doc "Construct a workflow event signaling that an executor has completed successfully." event_executor_completed
@doc "Construct a workflow event signaling that an executor has failed." event_executor_failed

# ── Content helpers ──────────────────────────────────────────────────────────

@doc "Combine two `UsageDetails` records additively (e.g. when concatenating streamed updates). Returns `nothing` if both inputs are `nothing`." add_usage_details
@doc "Return the wire-format string identifier (e.g. `\"text\"`, `\"function_call\"`) for a `ContentType` value." content_type_string
@doc "Parse a wire-format content-type string into a `ContentType` value; raises an error for unknown identifiers." parse_content_type

# ── Chat client helpers ──────────────────────────────────────────────────────

@doc "Parse a wire-format finish-reason string into a `FinishReason` value; unknown strings map to `FINISH_ERROR`." parse_finish_reason

# ── Skills constants ─────────────────────────────────────────────────────────

@doc "Default file name of a skill manifest within a skill directory." SKILL_FILENAME
@doc "Default set of file extensions scanned by skill discovery." DEFAULT_SCAN_EXTENSIONS
@doc "Load a skill ontology (a directory of related skills) from the given path." load_ontology!

# ── Evaluation predicates ────────────────────────────────────────────────────

@doc "Return `true` if the evaluation result represents a passing assertion." is_passed
@doc "Return `true` if the evaluation result represents a failing assertion." is_failed
@doc "Return `true` if the evaluation result represents an error during evaluation." is_error

# ── Exception types ──────────────────────────────────────────────────────────

@doc "Raised by an agent when authentication with the underlying provider fails." AgentInvalidAuthError
@doc "Raised by an agent when the request to the underlying provider is invalid." AgentInvalidRequestError
@doc "Raised by an agent when the response from the underlying provider cannot be parsed or is malformed." AgentInvalidResponseError
@doc "Raised by a chat client when authentication with the LLM provider fails." ChatClientInvalidAuthError
@doc "Raised by a chat client when the request to the LLM provider is invalid (bad model name, oversized input, etc.)." ChatClientInvalidRequestError
@doc "Raised when constructing or manipulating `Content` items in an inconsistent way." ContentError
@doc "Raised when a declarative agent specification fails to parse or validate." DeclarativeError
@doc "Raised when a middleware function in the pipeline throws or returns an invalid value." MiddlewareError
@doc "Raised when a tool's execution function throws an unhandled exception." ToolExecutionError
@doc "Raised when a workflow encounters a structural error (cycles, dead ends, type mismatches, etc.)." WorkflowError

# ── Misc ─────────────────────────────────────────────────────────────────────

@doc "Append additional messages to an existing message vector, normalizing their content. A no-op when `extras` is empty." extend_messages!
