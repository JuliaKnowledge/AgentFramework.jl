# Sessions & Memory

Sessions and memory stores provide conversation continuity and long-term knowledge for agents. AgentFramework.jl separates these into three concerns: session state, conversation history, and semantic memory.

## AgentSession

[`AgentSession`](@ref) is a lightweight, mutable container for conversation state:

```julia
using AgentFramework

session = AgentSession(
    id = "session-001",                      # Unique session ID (auto-generated if omitted)
    state = Dict{String, Any}(),             # User-managed mutable state
    user_id = "alice",                       # Optional user identifier
    thread_id = nothing,                     # Service-managed thread ID
    metadata = Dict{String, Any}(),          # Session metadata
)
```

You can store arbitrary data in `session.state`:

```julia
session.state["preferences"] = Dict("language" => "en", "units" => "metric")
session.state["turn_count"] = 0
```

### Creating Sessions from Agents

Use [`create_session`](@ref) as a convenience:

```julia
session = create_session(agent)
session = create_session(agent, session_id="my-session")
```

## SessionContext

[`SessionContext`](@ref) is a per-invocation context created fresh for each [`run_agent`](@ref) call. It carries the accumulated context from all providers:

```julia
ctx = SessionContext(
    session_id = "session-001",
    input_messages = [Message(:user, "Hello")],
    context_messages = Dict{String, Vector{Message}}(),  # Populated by providers
    instructions = String[],                              # Additional system instructions
    tools = FunctionTool[],                               # Additional tools
)
```

Context providers use helper functions to add context:

```julia
extend_messages!(ctx, "my_provider", [Message(:system, "Extra context")])
extend_instructions!(ctx, "Always respond in English.")
extend_tools!(ctx, [my_extra_tool])
```

## Context Providers

Context providers participate in the context engineering pipeline. They run before and after each agent invocation, adding messages, instructions, or tools to the [`SessionContext`](@ref).

### BaseContextProvider

Subtype [`BaseContextProvider`](@ref) and implement [`before_run!`](@ref) and/or [`after_run!`](@ref):

```julia
struct TimeContextProvider <: BaseContextProvider
    source_id::String
end

function AgentFramework.before_run!(provider::TimeContextProvider, agent, session, ctx, state)
    timestamp = string(Dates.now())
    extend_instructions!(ctx, "Current time: $(timestamp)")
end

agent = Agent(
    client = client,
    context_providers = [TimeContextProvider("time")],
)
```

Each provider receives its own `state::Dict{String, Any}` stored in the session, keyed by `source_id`.

### InMemoryHistoryProvider

[`InMemoryHistoryProvider`](@ref) is the simplest history provider — it stores conversation history in memory:

```julia
history = InMemoryHistoryProvider(source_id="history")

agent = Agent(
    client = client,
    instructions = "You are a helpful assistant.",
    context_providers = [history],
)

session = create_session(agent)

# Turn 1
run_agent(agent, "My name is Alice.", session=session)

# Turn 2 — the agent sees the full conversation history
run_agent(agent, "What's my name?", session=session)
```

!!! note
    `InMemoryHistoryProvider` loses data when the Julia process exits. For persistent storage, use `DBInterfaceHistoryProvider`, `FileHistoryProvider`, or `RedisHistoryProvider`.

### DBInterfaceHistoryProvider

[`DBInterfaceHistoryProvider`](@ref) stores messages in any SQL database via DBInterface.jl:

```julia
using SQLite

db = SQLite.DB(joinpath(storage_dir, "history.db"))

history = DBInterfaceHistoryProvider(
    source_id = "db_history",
    conn = db,
    table_name = "agent_messages",
    max_messages = 100,
    auto_create_table = true,
)

agent = Agent(
    client = client,
    context_providers = [history],
)
```

This works with SQLite, PostgreSQL (LibPQ.jl), MySQL (MySQL.jl), and any DBInterface-compatible driver.

### FileHistoryProvider

[`FileHistoryProvider`](@ref) stores messages as JSON files on disk:

```julia
history = FileHistoryProvider(
    source_id = "file_history",
    directory = joinpath(storage_dir, "conversation_history"),
)
```

### RedisHistoryProvider

[`RedisHistoryProvider`](@ref) stores messages in Redis for fast, distributed access:

```julia
history = RedisHistoryProvider(
    source_id = "redis_history",
    conn = (args...) -> nothing,
)
```

## Memory Stores

Memory stores provide semantic, long-term memory that persists across sessions. Unlike history providers (which replay exact messages), memory stores support similarity search for relevant context.

### Memory Records

[`MemoryRecord`](@ref) represents a stored memory:

```julia
record = MemoryRecord(
    id = "mem-001",
    scope = "session-001",
    content = "Alice prefers metric units.",
    metadata = Dict{String, Any}("source" => "conversation"),
)
```

### InMemoryMemoryStore

[`InMemoryMemoryStore`](@ref) stores memories in-process:

```julia
store = InMemoryMemoryStore()
add_memories!(store, [
    MemoryRecord(scope = "session-001", content = "User prefers dark mode."),
    MemoryRecord(scope = "session-001", content = "User's name is Alice."),
])

results = search_memories(store, "user preferences"; scope = "session-001")
```

### FileMemoryStore

[`FileMemoryStore`](@ref) persists memories to JSON files:

```julia
store = FileMemoryStore(directory = joinpath(storage_dir, "memories"))
add_memories!(store, [
    MemoryRecord(scope = "session-001", content = "User likes Julia."),
])
```

### SQLiteMemoryStore

[`SQLiteMemoryStore`](@ref) stores memories in an SQLite database:

```julia
store = SQLiteMemoryStore(joinpath(storage_dir, "memories.db"))
add_memories!(store, [
    MemoryRecord(scope = "session-001", content = "User is a data scientist."),
])
```

### RDFMemoryStore

[`RDFMemoryStore`](@ref) stores memories as RDF triples for ontology-aware retrieval:

```julia
println("RDFMemoryStore examples require RDFLib.jl in the active environment.")
```

### MemoryContextProvider

[`MemoryContextProvider`](@ref) integrates a memory store into the context provider pipeline, automatically searching for relevant memories based on the user's input:

```julia
memory_provider = MemoryContextProvider(
    store = SQLiteMemoryStore(joinpath(storage_dir, "memories.db")),
)

agent = Agent(
    client = client,
    context_providers = [history, memory_provider],
)
```

## Session Persistence

### InMemorySessionStore

[`InMemorySessionStore`](@ref) stores sessions in memory:

```julia
store = InMemorySessionStore()
save_session!(store, session)
loaded = load_session(store, session.id)
```

### FileSessionStore

[`FileSessionStore`](@ref) persists sessions to disk:

```julia
store = FileSessionStore(joinpath(storage_dir, "sessions"))
save_session!(store, session)

# Later, in a new process
loaded = load_session(store, session.id)
```

### Session Store API

All session stores implement the same interface:

```julia
session_id = session.id
save_session!(store, session)                    # Save a session
load_session(store, session_id)                  # Load by ID
delete_session!(store, session_id)               # Delete
list_sessions(store)                             # List all session IDs
has_session(store, session_id)                   # Check existence
```

## Turn Tracking

[`TurnTracker`](@ref) provides fine-grained tracking of conversation turns within a session:

```julia
tracker = TurnTracker()

# Start a turn with user messages
turn = start_turn!(tracker, [Message(:user, "Hello")])

# Complete the turn with assistant response
complete_turn!(tracker, [Message(:assistant, "Hi there!")])

# Query turn history
println(turn_count(tracker))          # 1
println(last_turn(tracker).index)     # 1
msgs = all_turn_messages(tracker)     # All messages across turns
```

Each [`ConversationTurn`](@ref) records:
- `index` — Turn number
- `user_messages` — Messages from the user
- `assistant_messages` — Messages from the assistant
- `timestamp` — When the turn occurred
- `metadata` — Custom turn metadata

## Serialization

Sessions can be serialized and deserialized for persistence:

```julia
# To/from Dict
d = session_to_dict(session)
session = session_from_dict(d)

# Messages
msg = Message(:user, "Hello")
d = message_to_dict(msg)
msg = message_from_dict(d)
```

## Next Steps

- [Workflows](workflows.md) — Multi-agent orchestration with shared state
- [Advanced Topics](@ref) — Message compaction strategies for long conversations
- [Providers](@ref) — Provider-managed sessions and thread IDs
