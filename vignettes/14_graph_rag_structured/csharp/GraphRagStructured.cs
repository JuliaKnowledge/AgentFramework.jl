// Structured GraphRAG — C# version
//
// Maps tabular CSV data to an RDF knowledge graph using an OWL ontology,
// then uses SPARQL retrieval + LLM generation for grounded answers.
//
// Requirements:
//   dotnet add package dotNetRDF
//   dotnet add package Microsoft.Extensions.AI
//   dotnet add package Microsoft.Extensions.AI.Ollama
//   ollama pull gemma3

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using VDS.RDF;
using VDS.RDF.Parsing;
using VDS.RDF.Query;
using VDS.RDF.Writing;
using Microsoft.Extensions.AI;
using OllamaSharp;

// ── Setup ────────────────────────────────────────────────────────────────────

IChatClient chatClient = new OllamaApiClient(new Uri("http://localhost:11434"), "gemma3:latest");

var store = new VDS.RDF.TripleStore();
var graph = new VDS.RDF.Graph();
store.Add(graph);

string ontNs = "http://example.org/ontology#";
string resNs = "http://example.org/resource#";
graph.NamespaceMap.AddNamespace("ont", new Uri(ontNs));
graph.NamespaceMap.AddNamespace("res", new Uri(resNs));

string ToIri(string name) => Regex.Replace(name.Trim(), @"[^a-zA-Z0-9]+", "");

// ── Step 1: Load Ontology ────────────────────────────────────────────────────

string ontologyPath = Path.Combine("..", "data", "jaguar_ontology.ttl");
var ttlParser = new TurtleParser();
ttlParser.Load(graph, ontologyPath);
Console.WriteLine($"Ontology loaded: {graph.Triples.Count} triples");

// ── Step 2: Load and Map CSV ─────────────────────────────────────────────────

string csvPath = Path.Combine("..", "data", "jaguars.csv");
var lines = File.ReadAllLines(csvPath);
var header = lines[0].Split(',');
var records = new List<Dictionary<string, string>>();

foreach (var line in lines.Skip(1))
{
    var fields = line.Split(',');
    var record = new Dictionary<string, string>();
    for (int i = 0; i < header.Length && i < fields.Length; i++)
        record[header[i].Trim()] = fields[i].Trim();
    records.Add(record);
}

Console.WriteLine($"Loaded {records.Count} jaguar records");

var rdf = graph.CreateUriNode(new Uri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"));
var rdfsLabel = graph.CreateUriNode(new Uri("http://www.w3.org/2000/01/rdf-schema#label"));
var rdfsComment = graph.CreateUriNode(new Uri("http://www.w3.org/2000/01/rdf-schema#comment"));

foreach (var record in records)
{
    string id = record.GetValueOrDefault("jaguar_id", "");
    if (string.IsNullOrEmpty(id)) continue;

    var jaguar = graph.CreateUriNode(new Uri(resNs + id));
    var jaguarType = graph.CreateUriNode(new Uri(ontNs + "Jaguar"));

    graph.Assert(new Triple(jaguar, rdf, jaguarType));
    graph.Assert(new Triple(jaguar, rdfsLabel,
        graph.CreateLiteralNode(record.GetValueOrDefault("name", ""))));
    graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "scientificName")),
        graph.CreateLiteralNode("Panthera onca")));

    // Gender
    string gender = record.GetValueOrDefault("gender", "");
    if (!string.IsNullOrEmpty(gender))
        graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "hasGender")),
            graph.CreateLiteralNode(gender)));

    // Locations (multi-valued)
    string location = record.GetValueOrDefault("location", "");
    if (!string.IsNullOrEmpty(location))
    {
        foreach (var loc in location.Split(';'))
        {
            string locTrimmed = loc.Trim();
            var locIri = graph.CreateUriNode(new Uri(resNs + ToIri(locTrimmed)));
            graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "occursIn")), locIri));
            graph.Assert(new Triple(locIri, rdf, graph.CreateUriNode(new Uri(ontNs + "Location"))));
            graph.Assert(new Triple(locIri, rdfsLabel, graph.CreateLiteralNode(locTrimmed)));
        }
    }

    // Monitoring organizations (multi-valued)
    string monitoringOrg = record.GetValueOrDefault("monitoring_org", "");
    if (!string.IsNullOrEmpty(monitoringOrg))
    {
        foreach (var org in monitoringOrg.Split(';'))
        {
            string orgTrimmed = org.Trim();
            var orgIri = graph.CreateUriNode(new Uri(resNs + ToIri(orgTrimmed)));
            graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "monitoredByOrg")), orgIri));
            graph.Assert(new Triple(orgIri, rdf, graph.CreateUriNode(new Uri(ontNs + "ConservationOrganization"))));
            graph.Assert(new Triple(orgIri, rdfsLabel, graph.CreateLiteralNode(orgTrimmed)));
        }
    }

    // First sighted date
    string firstSighted = record.GetValueOrDefault("first_sighted", "");
    if (!string.IsNullOrEmpty(firstSighted))
        graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "hasMonitoringStartDate")),
            graph.CreateLiteralNode(firstSighted)));

    // Killed status
    string isKilled = record.GetValueOrDefault("is_killed", "");
    if (!string.IsNullOrEmpty(isKilled))
        graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "wasKilled")),
            graph.CreateLiteralNode(isKilled.ToLower() == "true" ? "true" : "false")));

    // Cause of death
    string causeOfDeath = record.GetValueOrDefault("cause_of_death", "");
    if (!string.IsNullOrEmpty(causeOfDeath))
        graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "causeOfDeath")),
            graph.CreateLiteralNode(causeOfDeath)));

    // Identification mark
    string idMark = record.GetValueOrDefault("identification_mark", "");
    if (!string.IsNullOrEmpty(idMark))
        graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "hasIdentificationMark")),
            graph.CreateLiteralNode(idMark)));

    // Threats (multi-valued)
    string threats = record.GetValueOrDefault("threats", "");
    if (!string.IsNullOrEmpty(threats))
    {
        foreach (var threat in threats.Split(';'))
        {
            string t = threat.Trim();
            var threatIri = graph.CreateUriNode(new Uri(resNs + ToIri(t)));
            graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "facesThreat")), threatIri));
            graph.Assert(new Triple(threatIri, rdf, graph.CreateUriNode(new Uri(ontNs + "Threat"))));
            graph.Assert(new Triple(threatIri, rdfsLabel, graph.CreateLiteralNode(t)));
        }
    }

    // Monitoring techniques (multi-valued)
    string monTech = record.GetValueOrDefault("monitoring_technique", "");
    if (!string.IsNullOrEmpty(monTech))
    {
        foreach (var tech in monTech.Split(';'))
        {
            string techTrimmed = tech.Trim();
            var techIri = graph.CreateUriNode(new Uri(resNs + ToIri(techTrimmed)));
            graph.Assert(new Triple(jaguar, graph.CreateUriNode(new Uri(ontNs + "monitoredByTechnique")), techIri));
            graph.Assert(new Triple(techIri, rdf, graph.CreateUriNode(new Uri(ontNs + "MonitoringTechnique"))));
            graph.Assert(new Triple(techIri, rdfsLabel, graph.CreateLiteralNode(techTrimmed)));
        }
    }

    // Status notes
    string statusNotes = record.GetValueOrDefault("status_notes", "");
    if (!string.IsNullOrEmpty(statusNotes))
        graph.Assert(new Triple(jaguar, rdfsComment,
            graph.CreateLiteralNode(statusNotes)));
}

Console.WriteLine($"Knowledge graph: {graph.Triples.Count} triples");

// ── Step 3: SPARQL Queries ───────────────────────────────────────────────────

string RunSparql(string sparql)
{
    var processor = new VDS.RDF.Query.LeviathanQueryProcessor(store);
    var parser = new SparqlQueryParser();
    var query = parser.ParseFromString(sparql);
    var resultSet = processor.ProcessQuery(query) as SparqlResultSet;
    if (resultSet == null || resultSet.Count == 0)
        return "No results.";

    var sb = new StringBuilder();
    foreach (var result in resultSet)
    {
        var parts = result.Variables
            .Where(v => result.HasValue(v) && result[v] != null)
            .Select(v =>
            {
                var node = result[v];
                string val = node is ILiteralNode lit ? lit.Value
                    : node?.ToString()?.Split('/').Last() ?? "";
                return $"{v}={val}";
            });
        sb.AppendLine(string.Join(", ", parts));
    }
    return sb.ToString().TrimEnd();
}

Console.WriteLine("\n=== Jaguars and their locations ===");
Console.WriteLine(RunSparql(@"
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name ?location WHERE {
    ?j a ont:Jaguar ; rdfs:label ?name ; ont:occursIn ?loc .
    ?loc rdfs:label ?location .
} ORDER BY ?name"));

// ── Step 4: Structured GraphRAG ──────────────────────────────────────────────

async Task<string> AskStructuredRag(string question, string context)
{
    var messages = new List<ChatMessage>
    {
        new(ChatRole.System, $@"You are a wildlife conservation expert specializing in jaguars.
Answer the question using ONLY the knowledge graph facts provided below.
Do not use your own knowledge. If the facts don't contain enough information, say so.

Knowledge Graph Facts:
{context}"),
        new(ChatRole.User, question),
    };

    var response = await chatClient.GetResponseAsync(messages,
        new ChatOptions { Temperature = 0.1f });
    return response.Text;
}

// ── Step 5: Ask Questions ────────────────────────────────────────────────────

string elJefeFacts = RunSparql(@"
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?pred ?obj WHERE {
    ?j a ont:Jaguar ; rdfs:label ""El Jefe"" ; ?pred ?obj .
}");

Console.WriteLine("\n=== Tell me about El Jefe ===");
Console.WriteLine(await AskStructuredRag("Tell me about El Jefe.", elJefeFacts));

string killedFacts = RunSparql(@"
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name ?cause WHERE {
    ?j a ont:Jaguar ; rdfs:label ?name ; ont:wasKilled ""true"" .
    OPTIONAL { ?j ont:causeOfDeath ?cause }
}");

Console.WriteLine("\n=== Which jaguars were killed? ===");
Console.WriteLine(await AskStructuredRag("Which jaguars have been killed and what were the causes?", killedFacts));
