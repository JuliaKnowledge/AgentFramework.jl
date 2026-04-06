# Sessions

Sessions manage conversation state across agent invocations. An [`AgentSession`](@ref)
holds messages, metadata, and provider-specific state. Context providers and history
providers extend the session with additional information before each agent run.

## Core Types

```@docs
AgentFramework.AgentSession
AgentFramework.SessionContext
```

## Serialization

```@docs
AgentFramework.session_to_dict
AgentFramework.session_from_dict
```

## Context and History Providers

```@docs
AgentFramework.BaseContextProvider
AgentFramework.BaseHistoryProvider
AgentFramework.InMemoryHistoryProvider
AgentFramework.PerServiceCallHistoryMiddleware
AgentFramework.with_per_service_call_history
```

## Local History

```@docs
AgentFramework.LOCAL_HISTORY_CONVERSATION_ID
AgentFramework.is_local_history_conversation_id
```

## Context Extension

```@docs
AgentFramework.extend_messages!
AgentFramework.extend_instructions!
AgentFramework.extend_tools!
AgentFramework.get_all_context_messages
```

## Message Storage

```@docs
AgentFramework.get_messages
AgentFramework.save_messages!
AgentFramework.before_run!
AgentFramework.after_run!
```

## Conversation Turns

```@docs
AgentFramework.ConversationTurn
AgentFramework.TurnTracker
AgentFramework.start_turn!
AgentFramework.complete_turn!
AgentFramework.turn_count
AgentFramework.get_turn
AgentFramework.last_turn
AgentFramework.all_turn_messages
```

## Memory Stores

```@docs
AgentFramework.AbstractMemoryStore
AgentFramework.MemoryRecord
AgentFramework.MemorySearchResult
AgentFramework.MemoryContextProvider
AgentFramework.InMemoryMemoryStore
AgentFramework.FileMemoryStore
AgentFramework.SQLiteMemoryStore
AgentFramework.RDFMemoryStore
```

## Memory Operations

```@docs
AgentFramework.add_memories!
AgentFramework.search_memories
AgentFramework.get_memories
AgentFramework.clear_memories!
AgentFramework.load_ontology!
```

## Session Persistence

```@docs
AgentFramework.AbstractSessionStore
AgentFramework.InMemorySessionStore
AgentFramework.FileSessionStore
AgentFramework.load_session
AgentFramework.save_session!
AgentFramework.delete_session!
AgentFramework.list_sessions
AgentFramework.has_session
```
