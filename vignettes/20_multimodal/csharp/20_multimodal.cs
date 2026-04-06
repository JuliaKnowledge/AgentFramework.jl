// Multimodal Input — C# version
//
// This sample demonstrates how to send images, audio, and other binary content
// to an agent alongside text prompts. It mirrors the Julia vignette
// 20_multimodal.
//
// Prerequisites:
//   - Ollama running locally with llava:7b pulled (for vision examples)
//   - dotnet restore

using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OllamaSharp;

var endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT")
    ?? "http://localhost:11434";
var visionModel = Environment.GetEnvironmentVariable("OLLAMA_VISION_MODEL")
    ?? "llava:7b";

// ── Image from file ─────────────────────────────────────────────────────────
// DataContent wraps binary data with a MIME type.
var imageData = await File.ReadAllBytesAsync("photo.png");
var imageContent = new DataContent(imageData, "image/png");
Console.WriteLine($"Image media type: {imageContent.MediaType}");

// ── Image from URL ──────────────────────────────────────────────────────────
// UriContent references an external resource without downloading it.
var imageUrl = new UriContent("https://example.com/photo.jpg", "image/jpeg");
Console.WriteLine($"URL media type: {imageUrl.MediaType}");

// ── Sending images to a vision agent ────────────────────────────────────────
AIAgent agent = new OllamaApiClient(new Uri(endpoint), visionModel)
    .AsAIAgent(
        instructions: "Describe images concisely.",
        name: "VisionAgent");

// Compose a message with text and image content.
ChatMessage message = new(ChatRole.User, [
    new TextContent("What do you see in this image?"),
    new DataContent(imageData, "image/png"),
]);

Console.WriteLine(await agent.RunAsync(message));

// ── Audio content ───────────────────────────────────────────────────────────
var audioData = await File.ReadAllBytesAsync("recording.wav");
var audioContent = new DataContent(audioData, "audio/wav");
Console.WriteLine($"\nAudio media type: {audioContent.MediaType}");

// ── File attachments ────────────────────────────────────────────────────────
var pdfData = await File.ReadAllBytesAsync("report.pdf");
var pdfContent = new DataContent(pdfData, "application/pdf");
Console.WriteLine($"PDF media type: {pdfContent.MediaType}");

// ── Multiple content items ──────────────────────────────────────────────────
var photo1Data = await File.ReadAllBytesAsync("photo1.jpg");
var photo2Data = await File.ReadAllBytesAsync("photo2.jpg");

ChatMessage multiMessage = new(ChatRole.User, [
    new TextContent("Compare these two images:"),
    new DataContent(photo1Data, "image/jpeg"),
    new DataContent(photo2Data, "image/jpeg"),
]);

Console.WriteLine(await agent.RunAsync(multiMessage));

// ── Base64 utilities ────────────────────────────────────────────────────────
var rawBytes = await File.ReadAllBytesAsync("photo.png");
var encoded = Convert.ToBase64String(rawBytes);
var decoded = Convert.FromBase64String(encoded);
Console.WriteLine($"\nBase64 round-trip: {rawBytes.SequenceEqual(decoded)}");  // True
