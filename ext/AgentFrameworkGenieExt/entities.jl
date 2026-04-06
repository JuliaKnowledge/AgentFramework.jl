"""
Entity registry for managing agents and workflows in the DevUI.
"""

@enum EntityType AGENT_ENTITY WORKFLOW_ENTITY

"""
    EntityInfo

Holds metadata about a registered agent or workflow.
"""
mutable struct EntityInfo
    id::String
    name::String
    description::String
    type::EntityType
    entity::Any
    metadata::Dict{String, Any}
end

"""
    EntityRegistry

Thread-safe registry of entities (agents and workflows).
"""
mutable struct EntityRegistry
    entities::Dict{String, EntityInfo}
    lock::ReentrantLock
end

EntityRegistry() = EntityRegistry(Dict{String, EntityInfo}(), ReentrantLock())

"""
    detect_entity_type(entity) → EntityType

Auto-detect whether an entity is an Agent or Workflow.
"""
function detect_entity_type(entity)
    if entity isa AgentFramework.Agent
        return AGENT_ENTITY
    elseif entity isa AgentFramework.Workflow
        return WORKFLOW_ENTITY
    else
        error("Unknown entity type: $(typeof(entity)). Expected Agent or Workflow.")
    end
end

"""
    register_entity!(registry, entity; name, description, metadata)

Register an agent or workflow. Auto-detects type and generates ID.
"""
function register_entity!(registry::EntityRegistry, entity;
                          name::String = "",
                          description::String = "",
                          metadata::Dict{String, Any} = Dict{String, Any}())
    lock(registry.lock) do
        etype = detect_entity_type(entity)
        if isempty(name)
            name = if etype == AGENT_ENTITY
                entity.name
            else
                entity.name
            end
        end
        if isempty(description)
            description = if etype == AGENT_ENTITY
                entity.description
            else
                "Workflow: $(name)"
            end
        end

        # Ensure agents have a history provider for multi-turn conversation memory
        if etype == AGENT_ENTITY
            _ensure_history_provider!(entity)
        end

        id = string(UUIDs.uuid4())
        info = EntityInfo(id, name, description, etype, entity, metadata)
        registry.entities[id] = info
        return info
    end
end

"""
    _ensure_history_provider!(agent)

Add an InMemoryHistoryProvider if the agent doesn't already have one,
so that DevUI conversations maintain multi-turn memory by default.
"""
function _ensure_history_provider!(agent)
    has_history = any(p -> p isa AgentFramework.InMemoryHistoryProvider, agent.context_providers)
    if !has_history
        push!(agent.context_providers, AgentFramework.InMemoryHistoryProvider())
    end
end

"""
    list_entities(registry) → Vector{EntityInfo}
"""
function list_entities(registry::EntityRegistry)
    lock(registry.lock) do
        return collect(values(registry.entities))
    end
end

"""
    get_entity(registry, id) → EntityInfo or nothing
"""
function get_entity(registry::EntityRegistry, id::AbstractString)
    lock(registry.lock) do
        return get(registry.entities, id, nothing)
    end
end
