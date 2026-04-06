"""
Genie.jl server setup and route configuration for the DevUI.
"""

"""
    setup_routes!(registry, conv_store, config)

Configure all API routes using Genie.Router.
"""
function setup_routes!(registry::EntityRegistry, conv_store::ConversationStore, config::DevUIConfig)
    # Resolve the public directory for static files
    pkg_dir = dirname(@__DIR__)
    public_dir = joinpath(pkg_dir, "public")

    # ─── CORS preflight handler ───────────────────────────────────────────────
    Genie.Router.route("/api/*", method=Genie.Router.OPTIONS) do
        HTTP.Response(204, [
            "Access-Control-Allow-Origin" => "*",
            "Access-Control-Allow-Methods" => "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers" => "Content-Type, Authorization",
        ])
    end

    # ─── Health & Meta ────────────────────────────────────────────────────────
    Genie.Router.route("/api/health") do
        handle_health()
    end

    Genie.Router.route("/api/meta") do
        handle_meta(config, registry)
    end

    # ─── Entities ─────────────────────────────────────────────────────────────
    Genie.Router.route("/api/entities") do
        handle_list_entities(registry)
    end

    Genie.Router.route("/api/entities/:id") do
        handle_get_entity(registry, Genie.Router.params(:id))
    end

    # ─── Chat ─────────────────────────────────────────────────────────────────
    Genie.Router.route("/api/chat", method=Genie.Router.POST) do
        body = Genie.Requests.rawpayload()
        handle_chat(registry, conv_store, body)
    end

    Genie.Router.route("/api/chat/stream", method=Genie.Router.POST) do
        body = Genie.Requests.rawpayload()

        # Buffer SSE events and return as text/event-stream response
        # (Genie does not expose raw HTTP streams for true chunked streaming;
        #  for real-time streaming, use the WebSocket endpoint at /ws/chat)
        buf = IOBuffer()
        handle_chat_stream(registry, conv_store, body, buf)
        sse_body = String(take!(buf))

        return HTTP.Response(200, [
            "Content-Type" => "text/event-stream",
            "Cache-Control" => "no-cache",
            "Connection" => "keep-alive",
            "Access-Control-Allow-Origin" => "*",
        ]; body=sse_body)
    end

    Genie.Router.route("/api/entities/:id/agui", method=Genie.Router.POST) do
        body = Genie.Requests.rawpayload()

        buf = IOBuffer()
        handle_chat_stream_agui(registry, conv_store, Genie.Router.params(:id), body, buf)
        sse_body = String(take!(buf))

        return HTTP.Response(200, [
            "Content-Type" => "text/event-stream",
            "Cache-Control" => "no-cache",
            "Connection" => "keep-alive",
            "Access-Control-Allow-Origin" => "*",
        ]; body=sse_body)
    end

    # ─── Conversations ────────────────────────────────────────────────────────
    Genie.Router.route("/api/conversations") do
        handle_list_conversations(conv_store)
    end

    Genie.Router.route("/api/conversations/:id") do
        handle_get_conversation(conv_store, Genie.Router.params(:id))
    end

    Genie.Router.route("/api/conversations/:id", method=Genie.Router.DELETE) do
        handle_delete_conversation(conv_store, Genie.Router.params(:id))
    end

    Genie.Router.route("/api/conversations/:id/rename", method=Genie.Router.POST) do
        body = Genie.Requests.rawpayload()
        handle_rename_conversation(conv_store, Genie.Router.params(:id), body)
    end

    # ─── Static Files & Index ─────────────────────────────────────────────────
    Genie.Router.route("/") do
        index_path = joinpath(public_dir, "index.html")
        if isfile(index_path)
            return HTTP.Response(200, [
                "Content-Type" => "text/html; charset=utf-8",
            ]; body=read(index_path, String))
        else
            return HTTP.Response(200, [
                "Content-Type" => "text/html; charset=utf-8",
            ]; body=DEFAULT_INDEX_HTML)
        end
    end

    # Serve static files from public/css and public/js
    Genie.Router.route("/css/:filename") do
        serve_static_file(public_dir, "css", Genie.Router.params(:filename))
    end

    Genie.Router.route("/js/:filename") do
        serve_static_file(public_dir, "js", Genie.Router.params(:filename))
    end

    @info "Routes configured" endpoints=11 static_dir=public_dir
end

"""
    serve_static_file(public_dir, subdir, filename) → HTTP.Response

Serve a static file from public/{subdir}/{filename}.
"""
function serve_static_file(public_dir::AbstractString, subdir::AbstractString, filename::AbstractString)
    filepath = joinpath(public_dir, subdir, filename)
    if !isfile(filepath)
        return HTTP.Response(404, ["Content-Type" => "text/plain"]; body="Not found")
    end

    content_type = guess_content_type(filename)
    return HTTP.Response(200, [
        "Content-Type" => content_type,
        "Cache-Control" => "public, max-age=3600",
    ]; body=read(filepath))
end

"""
    guess_content_type(filename) → String

Guess MIME type from file extension.
"""
function guess_content_type(filename::AbstractString)
    ext = lowercase(splitext(filename)[2])
    mime_types = Dict(
        ".html" => "text/html; charset=utf-8",
        ".css"  => "text/css; charset=utf-8",
        ".js"   => "application/javascript; charset=utf-8",
        ".json" => "application/json",
        ".png"  => "image/png",
        ".jpg"  => "image/jpeg",
        ".jpeg" => "image/jpeg",
        ".svg"  => "image/svg+xml",
        ".ico"  => "image/x-icon",
        ".woff" => "font/woff",
        ".woff2" => "font/woff2",
        ".ttf"  => "font/ttf",
    )
    return get(mime_types, ext, "application/octet-stream")
end

# Default index HTML when no public/index.html exists
const DEFAULT_INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentFramework DevUI</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117; color: #c9d1d9;
            display: flex; justify-content: center; align-items: center;
            min-height: 100vh;
        }
        .container {
            text-align: center; max-width: 600px; padding: 2rem;
        }
        h1 { color: #58a6ff; margin-bottom: 1rem; font-size: 2rem; }
        p { margin-bottom: 0.5rem; line-height: 1.6; }
        code {
            background: #161b22; padding: 0.2em 0.5em; border-radius: 4px;
            font-size: 0.9em;
        }
        .status { color: #3fb950; font-weight: bold; }
        .endpoints { text-align: left; margin-top: 2rem; }
        .endpoints li { margin: 0.3rem 0; font-family: monospace; font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AgentFramework DevUI</h1>
        <p class="status">✓ Server is running</p>
        <p>Place your frontend files in <code>public/</code> to serve a custom UI.</p>
        <div class="endpoints">
            <h3>API Endpoints:</h3>
            <ul>
                <li>GET  /api/health</li>
                <li>GET  /api/meta</li>
                <li>GET  /api/entities</li>
                <li>GET  /api/entities/:id</li>
                <li>POST /api/chat</li>
                <li>POST /api/chat/stream</li>
                <li>POST /api/entities/:id/agui</li>
                <li>GET  /api/conversations</li>
                <li>GET  /api/conversations/:id</li>
                <li>DELETE /api/conversations/:id</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
