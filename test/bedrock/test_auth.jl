@testset "Bedrock auth" begin
    @testset "normalizes explicit credentials" begin
        creds = Bedrock._normalize_bedrock_credentials(
            BedrockCredentials(
                access_key_id = "test-access",
                secret_access_key = "test-secret",
                session_token = "test-session",
            ),
        )
        @test creds.access_key_id == "test-access"
        @test creds.secret_access_key == "test-secret"
        @test creds.session_token == "test-session"
    end

    @testset "resolves credentials from environment" begin
        saved = [(name, get(ENV, name, nothing)) for name in (
            "BEDROCK_ACCESS_KEY_ID",
            "BEDROCK_SECRET_ACCESS_KEY",
            "BEDROCK_SESSION_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        )]

        try
            ENV["AWS_ACCESS_KEY_ID"] = "env-access"
            ENV["AWS_SECRET_ACCESS_KEY"] = "env-secret"
            ENV["AWS_SESSION_TOKEN"] = "env-session"

            creds = Bedrock._resolve_bedrock_credentials("", "", nothing, "")
            @test creds.access_key_id == "env-access"
            @test creds.secret_access_key == "env-secret"
            @test creds.session_token == "env-session"
        finally
            _restore_env!(saved)
        end
    end

    @testset "resolves credentials and region from shared profile" begin
        saved = [(name, get(ENV, name, nothing)) for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_CONFIG_FILE",
            "BEDROCK_REGION",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
        )]

        mktempdir() do dir
            credentials_path = joinpath(dir, "credentials")
            config_path = joinpath(dir, "config")

            write(
                credentials_path,
                """
                [work]
                aws_access_key_id = profile-access
                aws_secret_access_key = profile-secret
                aws_session_token = profile-session
                """,
            )
            write(
                config_path,
                """
                [profile work]
                region = eu-west-1
                """,
            )

            try
                for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION")
                    pop!(ENV, name, nothing)
                end
                ENV["AWS_PROFILE"] = "work"
                ENV["AWS_SHARED_CREDENTIALS_FILE"] = credentials_path
                ENV["AWS_CONFIG_FILE"] = config_path

                creds = Bedrock._resolve_bedrock_credentials("", "", nothing, "")
                region = Bedrock._resolve_bedrock_region("", "")

                @test creds.access_key_id == "profile-access"
                @test creds.secret_access_key == "profile-secret"
                @test creds.session_token == "profile-session"
                @test region == "eu-west-1"
            finally
                _restore_env!(saved)
            end
        end
    end

    @testset "signs headers with SigV4" begin
        creds = BedrockCredentials(
            access_key_id = "AKIDEXAMPLE",
            secret_access_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token = "session-token",
        )
        signed = Bedrock._signed_headers_for_request(
            creds,
            "us-west-2",
            "POST",
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/test-model/converse",
            Dict("accept" => "application/json", "content-type" => "application/json"),
            "{\"hello\":\"world\"}";
            timestamp = DateTime(2026, 1, 2, 3, 4, 5),
        )

        @test signed["host"] == "bedrock-runtime.us-west-2.amazonaws.com"
        @test signed["x-amz-date"] == "20260102T030405Z"
        @test signed["x-amz-security-token"] == "session-token"
        @test occursin(
            "Credential=AKIDEXAMPLE/20260102/us-west-2/bedrock/aws4_request",
            signed["authorization"],
        )
        @test occursin(
            "SignedHeaders=accept;content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token",
            signed["authorization"],
        )
    end
end
