# Tests for multimodal content helpers

using AgentFramework
using Test
using Base64

@testset "Multimodal Content" begin

    @testset "MIME Type Detection" begin
        @test detect_mime_type("photo.png") == "image/png"
        @test detect_mime_type("photo.jpg") == "image/jpeg"
        @test detect_mime_type("photo.jpeg") == "image/jpeg"
        @test detect_mime_type("image.gif") == "image/gif"
        @test detect_mime_type("image.webp") == "image/webp"
        @test detect_mime_type("icon.svg") == "image/svg+xml"
        @test detect_mime_type("song.mp3") == "audio/mpeg"
        @test detect_mime_type("sound.wav") == "audio/wav"
        @test detect_mime_type("voice.ogg") == "audio/ogg"
        @test detect_mime_type("doc.pdf") == "application/pdf"
        @test detect_mime_type("data.json") == "application/json"
        @test detect_mime_type("data.csv") == "text/csv"
        @test detect_mime_type("unknown.xyz") == "application/octet-stream"
        # Case insensitive
        @test detect_mime_type("PHOTO.PNG") == "image/png"
        @test detect_mime_type("Song.MP3") == "audio/mpeg"
    end

    @testset "MIME Type Checks" begin
        @test is_image_mime("image/png") == true
        @test is_image_mime("image/jpeg") == true
        @test is_image_mime("audio/wav") == false
        @test is_image_mime("text/plain") == false
        @test is_audio_mime("audio/wav") == true
        @test is_audio_mime("audio/mpeg") == true
        @test is_audio_mime("image/png") == false
    end

    @testset "Image Content from Bytes" begin
        data = UInt8[0x89, 0x50, 0x4E, 0x47]  # PNG header bytes
        c = image_content(data)
        @test c.type == AgentFramework.DATA
        @test c.media_type == "image/png"
        @test c.text == base64encode(data)
        @test c.additional_properties["content_category"] == "image"
    end

    @testset "Image Content from Bytes with Custom MIME" begin
        data = UInt8[0xFF, 0xD8]  # JPEG header
        c = image_content(data; media_type="image/jpeg")
        @test c.media_type == "image/jpeg"
    end

    @testset "Image Content from File" begin
        # Create a temp image file
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "test.png")
        test_data = UInt8[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        write(tmpfile, test_data)

        c = image_content(tmpfile)
        @test c.type == AgentFramework.DATA
        @test c.media_type == "image/png"
        @test base64decode(c.text) == test_data
        @test c.additional_properties["content_category"] == "image"

        rm(tmpdir; recursive=true)
    end

    @testset "Image Content from Missing File" begin
        @test_throws ContentError image_content("/nonexistent/file.png")
    end

    @testset "Image URL Content" begin
        c = image_url_content("https://example.com/photo.jpg")
        @test c.type == AgentFramework.URI
        @test c.uri == "https://example.com/photo.jpg"
        @test c.additional_properties["content_category"] == "image"
    end

    @testset "Image URL Content with Detail" begin
        c = image_url_content("https://example.com/photo.jpg"; detail="high", media_type="image/jpeg")
        @test c.additional_properties["detail"] == "high"
        @test c.media_type == "image/jpeg"
    end

    @testset "Audio Content from Bytes" begin
        data = UInt8[0x52, 0x49, 0x46, 0x46]  # RIFF header
        c = audio_content(data)
        @test c.type == AgentFramework.DATA
        @test c.media_type == "audio/wav"
        @test c.additional_properties["content_category"] == "audio"
    end

    @testset "Audio Content from File" begin
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "test.mp3")
        test_data = UInt8[0xFF, 0xFB, 0x90, 0x00]
        write(tmpfile, test_data)

        c = audio_content(tmpfile)
        @test c.type == AgentFramework.DATA
        @test c.media_type == "audio/mpeg"
        @test base64decode(c.text) == test_data

        rm(tmpdir; recursive=true)
    end

    @testset "File Content from Bytes" begin
        data = UInt8[0x25, 0x50, 0x44, 0x46]  # PDF header
        c = file_content(data; media_type="application/pdf", filename="doc.pdf")
        @test c.type == AgentFramework.DATA
        @test c.media_type == "application/pdf"
        @test c.additional_properties["filename"] == "doc.pdf"
        @test c.additional_properties["content_category"] == "file"
    end

    @testset "File Content from Path" begin
        tmpdir = mktempdir()
        tmpfile = joinpath(tmpdir, "data.json")
        write(tmpfile, """{"key": "value"}""")

        c = file_content(tmpfile)
        @test c.type == AgentFramework.DATA
        @test c.media_type == "application/json"
        @test c.additional_properties["filename"] == "data.json"

        rm(tmpdir; recursive=true)
    end

    @testset "base64_to_bytes roundtrip" begin
        original = UInt8[1, 2, 3, 4, 5, 100, 200, 255]
        encoded = base64encode(original)
        decoded = base64_to_bytes(encoded)
        @test decoded == original
    end

    @testset "OpenAI Multimodal Conversion — Text" begin
        c = text_content("Hello world")
        d = content_to_openai_multimodal(c)
        @test d["type"] == "text"
        @test d["text"] == "Hello world"
    end

    @testset "OpenAI Multimodal Conversion — Image URL" begin
        c = image_url_content("https://example.com/img.png"; detail="low")
        d = content_to_openai_multimodal(c)
        @test d["type"] == "image_url"
        @test d["image_url"]["url"] == "https://example.com/img.png"
        @test d["image_url"]["detail"] == "low"
    end

    @testset "OpenAI Multimodal Conversion — Image Data" begin
        data = UInt8[1, 2, 3]
        c = image_content(data; media_type="image/jpeg")
        d = content_to_openai_multimodal(c)
        @test d["type"] == "image_url"
        @test startswith(d["image_url"]["url"], "data:image/jpeg;base64,")
    end

    @testset "OpenAI Multimodal Conversion — Audio Data" begin
        data = UInt8[1, 2, 3]
        c = audio_content(data; media_type="audio/wav")
        d = content_to_openai_multimodal(c)
        @test d["type"] == "input_audio"
        @test d["input_audio"]["format"] == "wav"
        @test d["input_audio"]["data"] == base64encode(data)
    end
end
