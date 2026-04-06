# Serialization

General-purpose serialization utilities for converting agent framework objects
to and from dictionaries and JSON strings. Use [`register_type!`](@ref) to
extend the serializer with custom types.

## Dict Conversion

```@docs
AgentFramework.serialize_to_dict
AgentFramework.deserialize_from_dict
```

## JSON Conversion

```@docs
AgentFramework.serialize_to_json
AgentFramework.deserialize_from_json
```

## Message Serialization

```@docs
AgentFramework.serialize_messages
AgentFramework.deserialize_messages
```

## Type Registration

```@docs
AgentFramework.register_type!
AgentFramework.register_state_type!
```
