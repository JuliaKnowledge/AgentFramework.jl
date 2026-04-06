struct A2AError <: Exception
    message::String
    inner::Union{Nothing, Exception}
end
A2AError(message::String) = A2AError(message, nothing)

function Base.showerror(io::IO, error::A2AError)
    print(io, nameof(typeof(error)), ": ", error.message)
    if error.inner !== nothing
        print(io, "\n  caused by: ")
        showerror(io, error.inner)
    end
end

struct A2AProtocolError <: Exception
    message::String
    method::Union{Nothing, String}
end
A2AProtocolError(message::String) = A2AProtocolError(message, nothing)

function Base.showerror(io::IO, error::A2AProtocolError)
    print(io, nameof(typeof(error)), ": ", error.message)
    if error.method !== nothing
        print(io, " [method=", error.method, "]")
    end
end

struct A2ATaskError <: Exception
    message::String
    task_id::Union{Nothing, String}
end
A2ATaskError(message::String) = A2ATaskError(message, nothing)

function Base.showerror(io::IO, error::A2ATaskError)
    print(io, nameof(typeof(error)), ": ", error.message)
    if error.task_id !== nothing
        print(io, " [task_id=", error.task_id, "]")
    end
end

struct A2ATimeoutError <: Exception
    message::String
end
