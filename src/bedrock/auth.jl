const DEFAULT_BEDROCK_REGION = "us-east-1"
const DEFAULT_BEDROCK_SIGNING_SERVICE = "bedrock"

Base.@kwdef struct BedrockCredentials
    access_key_id::String
    secret_access_key::String
    session_token::Union{Nothing, String} = nothing
end

function Base.show(io::IO, credentials::BedrockCredentials)
    print(io, "BedrockCredentials(\"", credentials.access_key_id, "\")")
end

function _nonempty_string(value)::Union{Nothing, String}
    value === nothing && return nothing
    text = strip(String(value))
    isempty(text) && return nothing
    return text
end

function _first_env(names::AbstractString...)::Union{Nothing, String}
    for name in names
        value = _nonempty_string(get(ENV, name, nothing))
        value !== nothing && return value
    end
    return nothing
end

_default_credentials_file() = get(ENV, "AWS_SHARED_CREDENTIALS_FILE", joinpath(homedir(), ".aws", "credentials"))
_default_config_file() = get(ENV, "AWS_CONFIG_FILE", joinpath(homedir(), ".aws", "config"))

function _parse_ini_file(path::AbstractString)::Dict{String, Dict{String, String}}
    sections = Dict{String, Dict{String, String}}()
    isfile(path) || return sections

    current_section::Union{Nothing, String} = nothing
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        (startswith(line, "#") || startswith(line, ";")) && continue

        if startswith(line, "[") && endswith(line, "]")
            current_section = strip(line[2:end-1])
            sections[current_section] = get(sections, current_section, Dict{String, String}())
            continue
        end

        current_section === nothing && continue

        separator = findfirst(==('='), line)
        separator === nothing && (separator = findfirst(==(':'), line))
        separator === nothing && continue

        key = lowercase(strip(line[1:separator - 1]))
        value = separator == lastindex(line) ? "" : strip(line[separator + 1:end])
        sections[current_section][key] = value
    end

    return sections
end

function _resolve_profile_name(profile::AbstractString)::String
    explicit = _nonempty_string(profile)
    explicit !== nothing && return explicit
    return something(_first_env("AWS_PROFILE", "AWS_DEFAULT_PROFILE"), "default")
end

function _load_profile_settings(
    profile::AbstractString;
    credentials_file::AbstractString = _default_credentials_file(),
    config_file::AbstractString = _default_config_file(),
)::Dict{String, String}
    settings = Dict{String, String}()

    credentials_sections = _parse_ini_file(credentials_file)
    config_sections = _parse_ini_file(config_file)

    haskey(credentials_sections, profile) && merge!(settings, credentials_sections[profile])
    haskey(config_sections, profile) && merge!(settings, config_sections[profile])

    config_profile = profile == "default" ? "default" : "profile $profile"
    haskey(config_sections, config_profile) && merge!(settings, config_sections[config_profile])

    return settings
end

function _normalize_bedrock_credentials(credentials::BedrockCredentials)::BedrockCredentials
    access_key_id = _nonempty_string(credentials.access_key_id)
    secret_access_key = _nonempty_string(credentials.secret_access_key)
    session_token = _nonempty_string(credentials.session_token)

    access_key_id === nothing && throw(ChatClientInvalidAuthError("Bedrock access_key_id cannot be empty."))
    secret_access_key === nothing && throw(ChatClientInvalidAuthError("Bedrock secret_access_key cannot be empty."))

    return BedrockCredentials(
        access_key_id = access_key_id,
        secret_access_key = secret_access_key,
        session_token = session_token,
    )
end

function _resolve_bedrock_credentials(
    access_key_id::AbstractString,
    secret_access_key::AbstractString,
    session_token::Union{Nothing, AbstractString},
    profile::AbstractString,
)::BedrockCredentials
    explicit_access = _nonempty_string(access_key_id)
    explicit_secret = _nonempty_string(secret_access_key)
    explicit_token = _nonempty_string(session_token)

    if explicit_access !== nothing || explicit_secret !== nothing
        explicit_access === nothing && throw(
            ChatClientInvalidAuthError("Bedrock access_key_id and secret_access_key must be provided together."),
        )
        explicit_secret === nothing && throw(
            ChatClientInvalidAuthError("Bedrock access_key_id and secret_access_key must be provided together."),
        )
        return BedrockCredentials(
            access_key_id = explicit_access,
            secret_access_key = explicit_secret,
            session_token = explicit_token,
        )
    end

    env_access = _first_env("BEDROCK_ACCESS_KEY_ID", "BEDROCK_ACCESS_KEY", "AWS_ACCESS_KEY_ID")
    env_secret = _first_env("BEDROCK_SECRET_ACCESS_KEY", "BEDROCK_SECRET_KEY", "AWS_SECRET_ACCESS_KEY")
    env_token = _first_env("BEDROCK_SESSION_TOKEN", "AWS_SESSION_TOKEN")

    if env_access !== nothing || env_secret !== nothing
        env_access === nothing && throw(
            ChatClientInvalidAuthError("Bedrock environment credentials require both access key and secret key."),
        )
        env_secret === nothing && throw(
            ChatClientInvalidAuthError("Bedrock environment credentials require both access key and secret key."),
        )
        return BedrockCredentials(
            access_key_id = env_access,
            secret_access_key = env_secret,
            session_token = env_token,
        )
    end

    profile_name = _resolve_profile_name(profile)
    profile_settings = _load_profile_settings(profile_name)
    profile_access = _nonempty_string(get(profile_settings, "aws_access_key_id", nothing))
    profile_secret = _nonempty_string(get(profile_settings, "aws_secret_access_key", nothing))
    profile_token = _nonempty_string(get(profile_settings, "aws_session_token", nothing))

    if profile_access !== nothing || profile_secret !== nothing
        profile_access === nothing && throw(
            ChatClientInvalidAuthError("AWS profile '$profile_name' is missing aws_access_key_id."),
        )
        profile_secret === nothing && throw(
            ChatClientInvalidAuthError("AWS profile '$profile_name' is missing aws_secret_access_key."),
        )
        return BedrockCredentials(
            access_key_id = profile_access,
            secret_access_key = profile_secret,
            session_token = profile_token,
        )
    end

    throw(
        ChatClientInvalidAuthError(
            "Bedrock credentials not set. Provide access_key_id/secret_access_key, set AWS_* or BEDROCK_* environment variables, or configure ~/.aws/credentials.",
        ),
    )
end

function _resolve_bedrock_credentials(client)::BedrockCredentials
    client.credentials !== nothing && return _normalize_bedrock_credentials(client.credentials)
    return _resolve_bedrock_credentials(client.access_key_id, client.secret_access_key, client.session_token, client.profile)
end

function _resolve_bedrock_region(region::AbstractString, profile::AbstractString)::String
    explicit_region = _nonempty_string(region)
    explicit_region !== nothing && return explicit_region

    env_region = _first_env("BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
    env_region !== nothing && return env_region

    profile_settings = _load_profile_settings(_resolve_profile_name(profile))
    profile_region = _nonempty_string(get(profile_settings, "region", nothing))
    profile_region !== nothing && return profile_region

    return DEFAULT_BEDROCK_REGION
end

_resolve_bedrock_region(client)::String = _resolve_bedrock_region(client.region, client.profile)

function _aws_percent_encode(value::AbstractString; preserve_percent::Bool = false)::String
    io = IOBuffer()
    for byte in codeunits(value)
        unreserved = (
            (byte >= UInt8('A') && byte <= UInt8('Z')) ||
            (byte >= UInt8('a') && byte <= UInt8('z')) ||
            (byte >= UInt8('0') && byte <= UInt8('9')) ||
            byte == UInt8('-') ||
            byte == UInt8('.') ||
            byte == UInt8('_') ||
            byte == UInt8('~')
        )
        if unreserved || (preserve_percent && byte == UInt8('%'))
            write(io, byte)
        else
            print(io, '%', uppercase(string(Int(byte), base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

function _canonical_query_string(uri::HTTP.URI)::String
    query = getfield(uri, :query)
    query === nothing && return ""
    query_text = String(query)
    isempty(query_text) && return ""

    pairs = Tuple{String, String}[]
    for part in split(query_text, '&')
        isempty(part) && continue
        if occursin('=', part)
            key, value = split(part, "=", limit = 2)
            push!(pairs, (_aws_percent_encode(key; preserve_percent = true), _aws_percent_encode(value; preserve_percent = true)))
        else
            push!(pairs, (_aws_percent_encode(part; preserve_percent = true), ""))
        end
    end

    sort!(pairs, by = identity)
    return join(["$key=$value" for (key, value) in pairs], "&")
end

function _host_header(uri::HTTP.URI)::String
    host = String(getfield(uri, :host))
    port = getfield(uri, :port)
    scheme = lowercase(String(getfield(uri, :scheme)))

    if port === nothing || port == "" || port == 0
        return host
    end

    port_text = string(port)
    isempty(port_text) && return host

    port_number = tryparse(Int, port_text)
    if port_number !== nothing && ((scheme == "https" && port_number == 443) || (scheme == "http" && port_number == 80))
        return host
    end

    return "$host:$port_text"
end

_sha256_hex(data::AbstractString) = bytes2hex(SHA.sha256(codeunits(data)))

function _hmac_sha256(key, data)
    key_bytes = key isa AbstractString ? collect(codeunits(key)) : Vector{UInt8}(key)
    data_bytes = data isa AbstractString ? collect(codeunits(data)) : Vector{UInt8}(data)
    return SHA.hmac_sha256(key_bytes, data_bytes)
end

function _amz_timestamp_parts(timestamp::DateTime)::Tuple{String, String}
    return (
        Dates.format(timestamp, dateformat"yyyymmddTHHMMSS") * "Z",
        Dates.format(timestamp, dateformat"yyyymmdd"),
    )
end

function _canonical_headers(headers::Dict{String, String})::Tuple{String, String}
    sorted_headers = sort(collect(headers), by = first)
    canonical = IOBuffer()
    signed_headers = String[]

    for (name, value) in sorted_headers
        normalized_name = lowercase(name)
        normalized_value = join(split(strip(value)), " ")
        print(canonical, normalized_name, ":", normalized_value, "\n")
        push!(signed_headers, normalized_name)
    end

    return (String(take!(canonical)), join(signed_headers, ";"))
end

function _signing_key(secret_access_key::AbstractString, date_stamp::AbstractString, region::AbstractString, service::AbstractString)
    k_date = _hmac_sha256("AWS4" * secret_access_key, date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    return _hmac_sha256(k_service, "aws4_request")
end

function _signed_headers_for_request(
    credentials::BedrockCredentials,
    region::AbstractString,
    method::AbstractString,
    url::AbstractString,
    headers::Dict{String, String},
    payload::AbstractString;
    timestamp::DateTime = now(UTC),
    service::AbstractString = DEFAULT_BEDROCK_SIGNING_SERVICE,
)::Dict{String, String}
    uri = HTTP.URI(String(url))
    amz_date, date_stamp = _amz_timestamp_parts(timestamp)
    payload_hash = _sha256_hex(payload)

    signed_headers_dict = Dict{String, String}(lowercase(key) => value for (key, value) in headers)
    signed_headers_dict["host"] = _host_header(uri)
    signed_headers_dict["x-amz-content-sha256"] = payload_hash
    signed_headers_dict["x-amz-date"] = amz_date

    if credentials.session_token !== nothing
        signed_headers_dict["x-amz-security-token"] = credentials.session_token
    end

    canonical_headers, signed_headers = _canonical_headers(signed_headers_dict)
    canonical_uri = isempty(String(getfield(uri, :path))) ? "/" : String(getfield(uri, :path))
    canonical_request = join(
        [
            uppercase(String(method)),
            canonical_uri,
            _canonical_query_string(uri),
            canonical_headers,
            signed_headers,
            payload_hash,
        ],
        "\n",
    )

    credential_scope = "$date_stamp/$region/$service/aws4_request"
    string_to_sign = join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            _sha256_hex(canonical_request),
        ],
        "\n",
    )

    signing_key = _signing_key(credentials.secret_access_key, date_stamp, region, service)
    signature = bytes2hex(_hmac_sha256(signing_key, string_to_sign))
    signed_headers_dict["authorization"] =
        "AWS4-HMAC-SHA256 Credential=$(credentials.access_key_id)/$credential_scope, SignedHeaders=$signed_headers, Signature=$signature"

    return signed_headers_dict
end

function _build_signed_headers(
    client,
    url::AbstractString,
    payload::AbstractString;
    accept::AbstractString = "application/json",
    content_type::AbstractString = "application/json",
    timestamp::DateTime = now(UTC),
)::Vector{Pair{String, String}}
    headers = Dict{String, String}(
        "accept" => String(accept),
        "content-type" => String(content_type),
    )

    for (key, value) in client.default_headers
        name = lowercase(String(key))
        name in ("authorization", "connection", "host", "x-amz-content-sha256", "x-amz-date", "x-amz-security-token") && continue
        headers[name] = String(value)
    end

    signed = _signed_headers_for_request(
        _resolve_bedrock_credentials(client),
        _resolve_bedrock_region(client),
        "POST",
        url,
        headers,
        payload;
        timestamp = timestamp,
    )

    return Pair{String, String}[key => value for (key, value) in sort(collect(signed), by = first)]
end

function _post_json(
    client,
    url::AbstractString,
    body::Dict{String, Any};
    error_label::AbstractString = "Bedrock API",
    accept::AbstractString = "application/json",
    content_type::AbstractString = "application/json",
)::Dict{String, Any}
    payload = JSON3.write(body)
    headers = _build_signed_headers(client, url, payload; accept = accept, content_type = content_type)
    response = HTTP.post(
        String(url),
        headers,
        payload;
        status_exception = false,
        readtimeout = client.read_timeout,
        connect_timeout = 10,
        retry = false,
    )

    if response.status != 200
        throw(ChatClientError("$error_label error ($(response.status)): $(String(response.body))"))
    end

    try
        return JSON3.read(String(response.body), Dict{String, Any})
    catch
        throw(ChatClientInvalidResponseError("$error_label response was not valid JSON."))
    end
end
