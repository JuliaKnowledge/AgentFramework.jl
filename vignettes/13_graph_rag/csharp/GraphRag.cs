/*
 * Graph RAG with dotNetRDF — C# version
 *
 * Demonstrates GraphRAG: extracting entities/relationships from text into
 * an RDF knowledge graph, then using programmatic retrieval (not tool-calling)
 * to ground LLM answers in knowledge graph facts.
 *
 * Requirements:
 *   dotnet add package Microsoft.Extensions.AI
 *   dotnet add package dotNetRdf
 *   ollama pull gemma3
 */

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using OllamaSharp;
using VDS.RDF;
using VDS.RDF.Parsing;
using VDS.RDF.Query;
using VDS.RDF.Query.Datasets;
using VDS.RDF.Writing;

// ── Setup ────────────────────────────────────────────────────────────────────

IChatClient client = new OllamaApiClient(new Uri("http://localhost:11434"), "gemma3:latest");
const string AstroNs = "http://example.org/astro/";

var kg = new VDS.RDF.Graph();
kg.NamespaceMap.AddNamespace("astro", new Uri(AstroNs));
kg.NamespaceMap.AddNamespace("rdf", new Uri("http://www.w3.org/1999/02/22-rdf-syntax-ns#"));

static string ToUri(string name) =>
    Regex.Replace(name.ToLower().Trim(), @"[^a-z0-9]+", "_");

// ── Corpus ───────────────────────────────────────────────────────────────────

string[] corpus = [
    """
    The Sun is the star at the center of the Solar System. It is a G-type
    main-sequence star (G2V). The Sun's diameter is about 1,392,700 km, roughly
    109 times that of Earth. It contains 99.86% of the total mass of the Solar System.
    """,

    """
    Mercury is the smallest planet in the Solar System and the closest to
    the Sun. It has no atmosphere and its surface temperature ranges from
    -180°C to 430°C. Mercury's orbital period is about 88 Earth days.
    """,

    """
    Venus is the second planet from the Sun. It is similar in size to Earth
    and is sometimes called Earth's sister planet. Venus has a thick atmosphere
    of carbon dioxide with sulfuric acid clouds. Its surface temperature is
    about 465°C, making it the hottest planet.
    """,

    """
    Earth is the third planet from the Sun and the only known planet to
    harbor life. It has one natural satellite, the Moon. Earth's diameter is
    12,742 km and its orbital period is 365.25 days.
    """,

    """
    Mars is the fourth planet from the Sun, often called the Red Planet due
    to iron oxide on its surface. Mars has two small moons: Phobos and Deimos.
    It has a thin atmosphere of mostly carbon dioxide. Mars's diameter is about
    6,779 km.
    """,

    """
    Jupiter is the fifth planet from the Sun and the largest in the Solar
    System. It is a gas giant with a mass more than twice that of all other
    planets combined. Jupiter has at least 95 known moons, including the four
    large Galilean moons: Io, Europa, Ganymede, and Callisto.
    """,

    """
    Saturn is the sixth planet from the Sun, famous for its ring system. It
    is the second-largest planet in the Solar System. Saturn has at least 146
    known moons, with Titan being the largest. Titan has a thick nitrogen
    atmosphere.
    """,
];

// ── Step 1: Entity Extraction ────────────────────────────────────────────────

const string ExtractionPrompt =
    """
    You are a knowledge graph extraction system.
    Given a text passage, extract entities and relationships as JSON triples.

    Return ONLY a JSON array of objects, each with "subject", "predicate", and "object" fields.
    Use simple, clear names (e.g., "Earth", "is_planet_of", "Solar_System").
    Extract factual relationships like: type_of, part_of, has_moon, has_diameter,
    has_temperature, has_atmosphere, orbits, distance_from, discovered_by, etc.
    For numeric values, include the units (e.g., "12742 km", "365.25 days").

    Do NOT include any text outside the JSON array. Do NOT use markdown code blocks.
    """;

var allExtracted = new List<Dictionary<string, string>>();

for (int i = 0; i < corpus.Length; i++)
{
    Console.Write($"Extracting from passage {i + 1}/{corpus.Length}...");

    var messages = new List<ChatMessage>
    {
        new(ChatRole.System, ExtractionPrompt),
        new(ChatRole.User, $"Extract knowledge triples from:\n\n{corpus[i]}"),
    };

    var response = await client.GetResponseAsync(messages, new() { Temperature = 0.1f });
    string rawText = response.Text ?? "";

    // Clean and parse JSON
    string cleaned = Regex.Replace(rawText, @"```json\s*", "");
    cleaned = Regex.Replace(cleaned, @"```\s*", "").Trim();

    int startIdx = cleaned.IndexOf('[');
    int endIdx = cleaned.LastIndexOf(']');
    if (startIdx >= 0 && endIdx >= 0)
    {
        try
        {
            string jsonStr = cleaned[startIdx..(endIdx + 1)];
            var extracted = JsonSerializer.Deserialize<List<Dictionary<string, string>>>(jsonStr);
            if (extracted != null)
            {
                allExtracted.AddRange(extracted);
                Console.WriteLine($"  → Extracted {extracted.Count} triples");
            }
        }
        catch (JsonException e)
        {
            Console.WriteLine($"  ⚠ Parse error: {e.Message}");
        }
    }
    else
    {
        Console.WriteLine("  ⚠ No JSON array found");
    }
}

Console.WriteLine($"\nTotal extracted triples: {allExtracted.Count}");

// ── Step 2: Populate RDF Graph ───────────────────────────────────────────────

string[] measurementIndicators = ["km", "°C", "days", "%", "kg"];

foreach (var tripleDict in allExtracted)
{
    if (!tripleDict.ContainsKey("subject") || !tripleDict.ContainsKey("predicate") || !tripleDict.ContainsKey("object"))
        continue;

    string subjStr = tripleDict["subject"];
    string predStr = tripleDict["predicate"];
    string objStr = tripleDict["object"];
    if (string.IsNullOrEmpty(subjStr) || string.IsNullOrEmpty(predStr) || string.IsNullOrEmpty(objStr))
        continue;

    var s = kg.CreateUriNode(new Uri(AstroNs + ToUri(subjStr)));
    var p = kg.CreateUriNode(new Uri(AstroNs + ToUri(predStr)));

    INode o;
    if (objStr.Any(char.IsDigit) && measurementIndicators.Any(u => objStr.Contains(u)))
        o = kg.CreateLiteralNode(objStr);
    else
        o = kg.CreateUriNode(new Uri(AstroNs + ToUri(objStr)));

    kg.Assert(new VDS.RDF.Triple(s, p, o));

    // Add RDF type triples
    if (predStr is "type_of" or "is_a")
    {
        var rdfType = kg.CreateUriNode(new Uri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));
        var typeObj = kg.CreateUriNode(new Uri(AstroNs + ToUri(objStr)));
        kg.Assert(new VDS.RDF.Triple(s, rdfType, typeObj));
    }
}

Console.WriteLine($"Knowledge graph: {kg.Triples.Count} triples");

// Print Turtle
var writer = new CompressingTurtleWriter();
using (var sw = new System.IO.StringWriter())
{
    writer.Save(kg, sw);
    Console.WriteLine(sw.ToString()[..Math.Min(2000, sw.ToString().Length)]);
}

// ── Step 3: Query Functions ──────────────────────────────────────────────────

string QueryEntity(string entityName)
{
    var uri = new Uri(AstroNs + ToUri(entityName));
    var results = new List<string>();

    foreach (var t in kg.GetTriplesWithSubject(kg.CreateUriNode(uri)))
    {
        string pred = t.Predicate.ToString().Split('/').Last();
        string obj = t.Object is ILiteralNode lit ? lit.Value : t.Object.ToString().Split('/').Last();
        results.Add($"{pred}: {obj}");
    }
    foreach (var t in kg.GetTriplesWithObject(kg.CreateUriNode(uri)))
    {
        string subj = t.Subject.ToString().Split('/').Last();
        string pred = t.Predicate.ToString().Split('/').Last();
        results.Add($"{subj} {pred} this entity");
    }

    return results.Count > 0
        ? $"Facts about {entityName}:\n{string.Join("\n", results)}"
        : $"No facts found for '{entityName}'.";
}

string RunSparql(string query)
{
    try
    {
        var parser = new SparqlQueryParser();
        var sparqlQuery = parser.ParseFromString(query);
        var processor = new LeviathanQueryProcessor(new InMemoryDataset(kg));
        var resultSet = processor.ProcessQuery(sparqlQuery) as SparqlResultSet;

        if (resultSet == null || resultSet.Count == 0)
            return "Query returned no results.";

        var lines = new List<string>();
        foreach (var result in resultSet)
        {
            var parts = result.Variables.Select(v =>
            {
                var node = result[v];
                string val = node is ILiteralNode lit ? lit.Value : node?.ToString()?.Split('/').Last() ?? "";
                return $"{v}={val}";
            });
            lines.Add(string.Join(", ", parts));
        }
        return $"Results ({resultSet.Count} rows):\n{string.Join("\n", lines)}";
    }
    catch (Exception e)
    {
        return $"SPARQL error: {e.Message}";
    }
}

string ListEntitiesOfType(string typeName)
{
    var typeUri = new Uri(AstroNs + ToUri(typeName));
    var rdfType = new Uri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
    var entities = new HashSet<string>();

    foreach (var t in kg.GetTriplesWithPredicateObject(kg.CreateUriNode(rdfType), kg.CreateUriNode(typeUri)))
        entities.Add(t.Subject.ToString().Split('/').Last());

    var typeOfUri = new Uri(AstroNs + "type_of");
    foreach (var t in kg.GetTriplesWithPredicateObject(kg.CreateUriNode(typeOfUri), kg.CreateUriNode(typeUri)))
        entities.Add(t.Subject.ToString().Split('/').Last());

    return entities.Count > 0
        ? $"Entities of type {typeName}: {string.Join(", ", entities.Order())}"
        : $"No entities of type '{typeName}'.";
}

// ── Step 4: Programmatic GraphRAG ────────────────────────────────────────────

/// Ask a question using GraphRAG: retrieve facts from the KG, then generate with the LLM.
async Task<string> AskGraphRag(string question, string context)
{
    string systemPrompt =
        $"""
        You are an astronomy expert. Answer the question using ONLY the
        knowledge graph facts provided below. Do not use your own knowledge.
        If the facts don't contain enough information, say so.

        Knowledge Graph Facts:
        {context}
        """;

    var messages = new List<ChatMessage>
    {
        new(ChatRole.System, systemPrompt),
        new(ChatRole.User, question),
    };

    var response = await client.GetResponseAsync(messages, new() { Temperature = 0.1f });
    return response.Text ?? "";
}

// ── Step 5: Ask Questions ────────────────────────────────────────────────────

// Q1: Simple entity lookup
var q1 = "What do we know about Mars?";
var ctx1 = QueryEntity("Mars");

// Q2: Comparative question
var q2 = "Which planet is larger, Earth or Mars? Give their diameters.";
var ctx2 = QueryEntity("Earth") + "\n\n" + QueryEntity("Mars");

// Q3: Multi-entity query
var q3 = "What moons does Jupiter have? Which are the Galilean moons?";
var ctx3 = QueryEntity("Jupiter");

// Q4: Type listing
var q4 = "List all planets in the Solar System.";
var ctx4 = ListEntitiesOfType("Planet");

// Q5: Comparative temperatures
var q5 = "What is the surface temperature of Venus and how does it compare to Mercury?";
var ctx5 = QueryEntity("Venus") + "\n\n" + QueryEntity("Mercury");

var questionsWithContext = new (string Question, string Context)[]
{
    (q1, ctx1), (q2, ctx2), (q3, ctx3), (q4, ctx4), (q5, ctx5),
};

Console.WriteLine("\n========== GraphRAG (grounded in knowledge graph) ==========\n");

foreach (var (q, ctx) in questionsWithContext)
{
    Console.WriteLine(new string('=', 60));
    Console.WriteLine($"Q: {q}");
    Console.WriteLine(new string('-', 60));
    string answer = await AskGraphRag(q, ctx);
    Console.WriteLine($"A: {answer}");
    Console.WriteLine();
}

// ── Step 6: Comparison — Plain LLM (no knowledge graph) ─────────────────────

Console.WriteLine("\n========== Plain LLM (no knowledge graph) ==========\n");

foreach (var (q, _) in questionsWithContext)
{
    Console.WriteLine(new string('=', 60));
    Console.WriteLine($"Q: {q}");
    Console.WriteLine(new string('-', 60));

    var messages = new List<ChatMessage>
    {
        new(ChatRole.System, "You are a helpful assistant. Answer questions concisely."),
        new(ChatRole.User, q),
    };
    var response = await client.GetResponseAsync(messages, new() { Temperature = 0.1f });
    Console.WriteLine($"A: {response.Text}");
    Console.WriteLine();
}
