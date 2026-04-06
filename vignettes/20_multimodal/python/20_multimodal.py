"""
Multimodal Input — Python version

This sample demonstrates how to send images, audio, and other binary content
to an agent alongside text prompts. It mirrors the Julia vignette
20_multimodal.

Prerequisites:
  - Ollama running locally with `llama3.2-vision:latest` pulled (for vision examples)
  - pip install agent-framework-ollama
"""

import asyncio
import base64
import struct

from agent_framework import Content, Message
from agent_framework.ollama import OllamaChatClient


def _make_tiny_png() -> bytes:
    """Create a minimal 1x1 red PNG (valid file) for demonstration."""
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        import binascii
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", binascii.crc32(c) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    raw_row = b"\x00\xff\x00\x00"  # filter=None, R=255, G=0, B=0
    idat = zlib.compress(raw_row)
    return header + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


async def main() -> None:
    # ── Image from bytes ────────────────────────────────────────────────────
    # Content.from_data() base64-encodes the bytes and creates a data URI.
    image_bytes = _make_tiny_png()
    image = Content.from_data(data=image_bytes, media_type="image/png")
    print(f"Image content type: {image.type}")
    print(f"Image media type:   {image.media_type}")

    # ── Image from URL ──────────────────────────────────────────────────────
    # Content.from_uri() references an external URL without downloading it.
    image_url = Content.from_uri(
        "https://example.com/photo.jpg",
        media_type="image/jpeg",
    )
    print(f"\nURL content type: {image_url.type}")
    print(f"URL uri:          {image_url.uri}")

    # ── Sending images to a vision agent ────────────────────────────────────
    client = OllamaChatClient(
        host="http://localhost:11434",
        model="llama3.2-vision:latest",
    )

    agent = client.as_agent(
        name="VisionAgent",
        instructions="Describe images concisely.",
    )

    message = Message(
        role="user",
        contents=[
            Content.from_text("What do you see in this image?"),
            Content.from_data(data=image_bytes, media_type="image/png"),
        ],
    )

    print("\nSending image to vision model (llama3.2-vision)...")
    try:
        result = await agent.run(message)
        print(f"Vision agent: {result}")
    except Exception as exc:
        print(f"  ⚠ Vision call skipped ({type(exc).__name__}): {exc}")

    # ── Audio content (demonstrate Content creation) ────────────────────────
    audio_bytes = b"\x00" * 64  # dummy audio payload for demonstration
    audio = Content.from_data(data=audio_bytes, media_type="audio/wav")
    print(f"\nAudio content type: {audio.type}")
    print(f"Audio media type:   {audio.media_type}")

    # ── File attachments (demonstrate Content creation) ─────────────────────
    pdf_bytes = b"%PDF-1.0 dummy"  # dummy PDF payload for demonstration
    pdf = Content.from_data(
        data=pdf_bytes,
        media_type="application/pdf",
        additional_properties={"filename": "report.pdf"},
    )
    print(f"\nPDF content type: {pdf.type}")
    print(f"PDF media type:   {pdf.media_type}")

    # ── Multiple content items ──────────────────────────────────────────────
    photo1_bytes = _make_tiny_png()
    photo2_bytes = _make_tiny_png()

    multi_message = Message(
        role="user",
        contents=[
            Content.from_text("Compare these two images:"),
            Content.from_data(data=photo1_bytes, media_type="image/jpeg"),
            Content.from_data(data=photo2_bytes, media_type="image/jpeg"),
        ],
    )
    print("\nSending two images for comparison...")
    try:
        result = await agent.run(multi_message)
        print(f"Comparison: {result}")
    except Exception as exc:
        print(f"  ⚠ Comparison call skipped ({type(exc).__name__}): {exc}")

    # ── Base64 utilities ────────────────────────────────────────────────────
    raw = _make_tiny_png()
    encoded = base64.b64encode(raw).decode("utf-8")
    decoded = base64.b64decode(encoded)
    print(f"\nBase64 round-trip: {raw == decoded}")  # True


if __name__ == "__main__":
    asyncio.run(main())
