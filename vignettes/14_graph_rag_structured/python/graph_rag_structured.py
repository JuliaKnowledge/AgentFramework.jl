"""
Structured GraphRAG — Python version

Maps tabular CSV data to an RDF knowledge graph using an OWL ontology,
then uses SPARQL retrieval + LLM generation for grounded answers.

Requirements:
    pip install rdflib openai
    ollama pull gemma3
"""

import csv
import re
from rdflib import Graph, Namespace, Literal, RDF, RDFS, URIRef, OWL
from openai import OpenAI

# ── Setup ─────────────────────────────────────────────────────────────────────

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gemma3:latest"

ONT = Namespace("http://example.org/ontology#")
RES = Namespace("http://example.org/resource#")

kg = Graph()
kg.bind("ont", ONT)
kg.bind("res", RES)


def to_iri(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", name.strip())


# ── Step 1: Load Ontology ────────────────────────────────────────────────────

kg.parse("../data/jaguar_ontology.ttl", format="turtle")
print(f"Ontology loaded: {len(kg)} triples")

# ── Step 2: Load and Map CSV ─────────────────────────────────────────────────

with open("../data/jaguars.csv") as f:
    reader = csv.DictReader(f)
    records = list(reader)

print(f"Loaded {len(records)} jaguar records")

for record in records:
    jaguar_id = record["jaguar_id"]
    jaguar = RES[jaguar_id]

    # Core identity
    kg.add((jaguar, RDF.type, ONT.Jaguar))
    kg.add((jaguar, RDFS.label, Literal(record["name"])))
    kg.add((jaguar, ONT.scientificName, Literal("Panthera onca")))

    # Gender
    if record.get("gender"):
        kg.add((jaguar, ONT.hasGender, Literal(record["gender"])))

    # Locations (multi-valued)
    if record.get("location"):
        for loc in record["location"].split(";"):
            loc = loc.strip()
            loc_iri = RES[to_iri(loc)]
            kg.add((jaguar, ONT.occursIn, loc_iri))
            kg.add((loc_iri, RDF.type, ONT.Location))
            kg.add((loc_iri, RDFS.label, Literal(loc)))

    # Monitoring organizations (multi-valued)
    if record.get("monitoring_org"):
        for org in record["monitoring_org"].split(";"):
            org = org.strip()
            org_iri = RES[to_iri(org)]
            kg.add((jaguar, ONT.monitoredByOrg, org_iri))
            kg.add((org_iri, RDF.type, ONT.ConservationOrganization))
            kg.add((org_iri, RDFS.label, Literal(org)))

    # First sighted
    if record.get("first_sighted"):
        kg.add((jaguar, ONT.hasMonitoringStartDate, Literal(record["first_sighted"])))

    # Killed status
    if record.get("is_killed"):
        kg.add((jaguar, ONT.wasKilled, Literal(record["is_killed"].lower())))

    # Cause of death
    if record.get("cause_of_death"):
        kg.add((jaguar, ONT.causeOfDeath, Literal(record["cause_of_death"])))

    # Identification mark
    if record.get("identification_mark"):
        kg.add((jaguar, ONT.hasIdentificationMark, Literal(record["identification_mark"])))

    # Threats (multi-valued)
    if record.get("threats"):
        for threat in record["threats"].split(";"):
            threat = threat.strip()
            threat_iri = RES[to_iri(threat)]
            kg.add((jaguar, ONT.facesThreat, threat_iri))
            kg.add((threat_iri, RDF.type, ONT.Threat))
            kg.add((threat_iri, RDFS.label, Literal(threat)))

    # Monitoring techniques (multi-valued)
    if record.get("monitoring_technique"):
        for tech in record["monitoring_technique"].split(";"):
            tech = tech.strip()
            tech_iri = RES[to_iri(tech)]
            kg.add((jaguar, ONT.monitoredByTechnique, tech_iri))
            kg.add((tech_iri, RDF.type, ONT.MonitoringTechnique))
            kg.add((tech_iri, RDFS.label, Literal(tech)))

    # Status notes
    if record.get("status_notes"):
        kg.add((jaguar, RDFS.comment, Literal(record["status_notes"])))

print(f"Knowledge graph: {len(kg)} triples")
print()
print("Turtle serialization (first 2000 chars):")
print(kg.serialize(format="turtle")[:2000])

# ── Step 3: SPARQL Queries ───────────────────────────────────────────────────

print("\n=== Jaguars and their locations ===")
results = kg.query("""
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name ?location WHERE {
    ?j a ont:Jaguar ; rdfs:label ?name ; ont:occursIn ?loc .
    ?loc rdfs:label ?location .
} ORDER BY ?name
""")
for row in results:
    print(f"  {row.name} → {row.location}")

print("\n=== Jaguars that were killed ===")
results = kg.query("""
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name ?cause WHERE {
    ?j a ont:Jaguar ; rdfs:label ?name ; ont:wasKilled "true" .
    OPTIONAL { ?j ont:causeOfDeath ?cause }
}
""")
for row in results:
    cause = f" — {row.cause}" if row.cause else ""
    print(f"  {row.name}{cause}")


# ── Step 4: Retrieval Functions ──────────────────────────────────────────────

def query_jaguar(name: str) -> str:
    results = list(kg.query(f"""
    PREFIX ont: <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?pred ?obj WHERE {{
        ?j a ont:Jaguar ; rdfs:label "{name}" ; ?pred ?obj .
    }}
    """))
    if not results:
        return f"No jaguar found with name '{name}'."
    lines = []
    for row in results:
        pred = str(row.pred).split("#")[-1].split("/")[-1]
        if pred == "type":
            continue
        lines.append(f"{pred}: {row.obj}")
    return f"Facts about {name}:\n" + "\n".join(lines)


def run_sparql_query(query: str) -> str:
    try:
        results = list(kg.query(query))
        if not results:
            return "Query returned no results."
        lines = []
        for row in results:
            parts = [f"{str(val)}" for val in row]
            lines.append(", ".join(parts))
        return f"Results ({len(results)} rows):\n" + "\n".join(lines)
    except Exception as e:
        return f"SPARQL error: {e}"


# ── Step 5: Structured GraphRAG Pipeline ─────────────────────────────────────

def ask_structured_rag(question: str, jaguars=None, sparql_query_str="") -> str:
    context_parts = []
    for name in (jaguars or []):
        context_parts.append(query_jaguar(name))
    if sparql_query_str:
        context_parts.append(run_sparql_query(sparql_query_str))
    context = "\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""You are a wildlife conservation expert specializing in jaguars.
Answer the question using ONLY the knowledge graph facts provided below.
Do not use your own knowledge. If the facts don't contain enough information, say so.

Knowledge Graph Facts:
{context}""",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


# ── Step 6: Ask Questions ────────────────────────────────────────────────────

questions = [
    ("Tell me about El Jefe.", ["El Jefe"], ""),
    (
        "Which jaguars have been killed and what were the causes?",
        ["Macho B", "Yo'oko"],
        """PREFIX ont: <http://example.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?name ?cause WHERE {
            ?j a ont:Jaguar ; rdfs:label ?name ; ont:wasKilled "true" .
            OPTIONAL { ?j ont:causeOfDeath ?cause }
        }""",
    ),
    (
        "What monitoring techniques are used to track jaguars?",
        [],
        """PREFIX ont: <http://example.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?name ?technique WHERE {
            ?j a ont:Jaguar ; rdfs:label ?name ;
               ont:monitoredByTechnique ?t .
            ?t rdfs:label ?technique .
        }""",
    ),
]

for q, jags, sparql in questions:
    print("=" * 60)
    print(f"Q: {q}")
    print("-" * 60)
    answer = ask_structured_rag(q, jaguars=jags, sparql_query_str=sparql)
    print(f"A: {answer}")
    print()
