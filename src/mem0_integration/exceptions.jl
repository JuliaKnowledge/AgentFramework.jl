struct Mem0Error <: Exception
    message::String
    status::Union{Nothing, Int}
    body::Union{Nothing, String}
end

Mem0Error(message::String) = Mem0Error(message, nothing, nothing)
Mem0Error(message::String, status::Int) = Mem0Error(message, status, nothing)

function Base.showerror(io::IO, error::Mem0Error)
    print(io, "Mem0Error: ", error.message)
    if error.status !== nothing
        print(io, " [status=", error.status, "]")
    end
    if error.body !== nothing
        print(io, "\n  body: ", error.body)
    end
end
