"""
Graph RAG with rdflib — Python version

Demonstrates GraphRAG: extracting entities/relationships from text into
an RDF knowledge graph, then using programmatic retrieval (not tool-calling)
to ground LLM answers in knowledge graph facts.

Requirements:
    pip install rdflib openai
    ollama pull gemma3
"""

import json
import re
from rdflib import Graph, Namespace, Literal, RDF, URIRef
from openai import OpenAI

# ── Setup ─────────────────────────────────────────────────────────────────────

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "gemma3:latest"

ASTRO = Namespace("http://example.org/astro/")
SCHEMA = Namespace("http://schema.org/")

kg = Graph()
kg.bind("astro", ASTRO)
kg.bind("schema", SCHEMA)


def to_uri(name: str) -> str:
    """Convert a name to a URI-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().strip())


# ── Corpus ────────────────────────────────────────────────────────────────────

corpus = [
    """The Sun is the star at the center of the Solar System. It is a G-type
    main-sequence star (G2V). The Sun's diameter is about 1,392,700 km, roughly
    109 times that of Earth. It contains 99.86% of the total mass of the Solar System.""",

    """Mercury is the smallest planet in the Solar System and the closest to
    the Sun. It has no atmosphere and its surface temperature ranges from
    -180°C to 430°C. Mercury's orbital period is about 88 Earth days.""",

    """Venus is the second planet from the Sun. It is similar in size to Earth
    and is sometimes called Earth's sister planet. Venus has a thick atmosphere
    of carbon dioxide with sulfuric acid clouds. Its surface temperature is
    about 465°C, making it the hottest planet.""",

    """Earth is the third planet from the Sun and the only known planet to
    harbor life. It has one natural satellite, the Moon. Earth's diameter is
    12,742 km and its orbital period is 365.25 days.""",

    """Mars is the fourth planet from the Sun, often called the Red Planet due
    to iron oxide on its surface. Mars has two small moons: Phobos and Deimos.
    It has a thin atmosphere of mostly carbon dioxide. Mars's diameter is about
    6,779 km.""",

    """Jupiter is the fifth planet from the Sun and the largest in the Solar
    System. It is a gas giant with a mass more than twice that of all other
    planets combined. Jupiter has at least 95 known moons, including the four
    large Galilean moons: Io, Europa, Ganymede, and Callisto.""",

    """Saturn is the sixth planet from the Sun, famous for its ring system. It
    is the second-largest planet in the Solar System. Saturn has at least 146
    known moons, with Titan being the largest. Titan has a thick nitrogen
    atmosphere.""",
]

# ── Step 1: Entity Extraction ─────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a knowledge graph extraction system.
Given a text passage, extract entities and relationships as JSON triples.

Return ONLY a JSON array of objects, each with "subject", "predicate", and "object" fields.
Use simple, clear names (e.g., "Earth", "is_planet_of", "Solar_System").
Extract factual relationships like: type_of, part_of, has_moon, has_diameter,
has_temperature, has_atmosphere, orbits, distance_from, discovered_by, etc.
For numeric values, include the units (e.g., "12742 km", "365.25 days").

Do NOT include any text outside the JSON array. Do NOT use markdown code blocks.

Example output:
[{"subject": "Earth", "predicate": "type_of", "object": "Planet"},
 {"subject": "Earth", "predicate": "has_moon", "object": "Moon"},
 {"subject": "Earth", "predicate": "orbits", "object": "Sun"}]
"""

all_extracted = []

for i, passage in enumerate(corpus, 1):
    print(f"Extracting from passage {i}/{len(corpus)}...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"Extract knowledge triples from:\n\n{passage}"},
        ],
        temperature=0.1,
    )
    raw_text = response.choices[0].message.content

    # Clean and parse JSON
    cleaned = re.sub(r"```json\s*", "", raw_text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start >= 0 and end >= 0:
        try:
            extracted = json.loads(cleaned[start : end + 1])
            all_extracted.extend(extracted)
            print(f"  → Extracted {len(extracted)} triples")
        except json.JSONDecodeError as e:
            print(f"  ⚠ Parse error: {e}")
    else:
        print("  ⚠ No JSON array found")

print(f"\nTotal extracted triples: {len(all_extracted)}")

# ── Step 2: Populate RDF Graph ────────────────────────────────────────────────

MEASUREMENT_INDICATORS = ["km", "°C", "days", "%", "kg"]

for triple_dict in all_extracted:
    s = ASTRO[to_uri(triple_dict["subject"])]
    p = ASTRO[to_uri(triple_dict["predicate"])]

    obj_str = triple_dict["object"]
    if any(c.isdigit() for c in obj_str) and any(u in obj_str for u in MEASUREMENT_INDICATORS):
        o = Literal(obj_str)
    else:
        o = ASTRO[to_uri(obj_str)]

    kg.add((s, p, o))

    # Add RDF type triples
    if triple_dict["predicate"] in ("type_of", "is_a"):
        kg.add((s, RDF.type, ASTRO[to_uri(obj_str)]))

print(f"Knowledge graph: {len(kg)} triples")
print()
print("Turtle serialization:")
print(kg.serialize(format="turtle"))

# ── Step 3: Query Functions ───────────────────────────────────────────────────


def query_entity(entity_name: str) -> str:
    """Find all known facts about a specific entity."""
    uri = ASTRO[to_uri(entity_name)]
    results = []
    for s, p, o in kg.triples((uri, None, None)):
        pred = str(p).split("/")[-1]
        obj = str(o) if isinstance(o, Literal) else str(o).split("/")[-1]
        results.append(f"{pred}: {obj}")
    for s, p, o in kg.triples((None, None, uri)):
        subj = str(s).split("/")[-1]
        pred = str(p).split("/")[-1]
        results.append(f"{subj} {pred} this entity")
    return f"Facts about {entity_name}:\n" + "\n".join(results) if results else f"No facts found for '{entity_name}'."


def run_sparql(query: str) -> str:
    """Execute a SPARQL query against the knowledge graph."""
    try:
        results = list(kg.query(query))
        if not results:
            return "Query returned no results."
        lines = []
        for row in results:
            parts = []
            for val in row:
                v = str(val) if isinstance(val, Literal) else str(val).split("/")[-1]
                parts.append(v)
            lines.append(", ".join(parts))
        return f"Results ({len(results)} rows):\n" + "\n".join(lines)
    except Exception as e:
        return f"SPARQL error: {e}"


def list_entities_of_type(type_name: str) -> str:
    """List all entities of a given type in the knowledge graph."""
    type_uri = ASTRO[to_uri(type_name)]
    entities = set()
    for s, p, o in kg.triples((None, RDF.type, type_uri)):
        entities.add(str(s).split("/")[-1])
    for s, p, o in kg.triples((None, ASTRO["type_of"], type_uri)):
        entities.add(str(s).split("/")[-1])
    return f"Entities of type {type_name}: {', '.join(sorted(entities))}" if entities else f"No entities of type '{type_name}'."


# ── Step 4: Programmatic GraphRAG ─────────────────────────────────────────────


def ask_graph_rag(question: str, entities: list[str] | None = None) -> str:
    """Ask a question using GraphRAG: retrieve from KG, then generate with LLM."""
    # Step 1: Retrieve — gather facts from the knowledge graph
    context_parts = []
    for entity in entities or []:
        facts = query_entity(entity)
        context_parts.append(facts)
    context = "\n\n".join(context_parts)

    # Step 2: Generate — ask the LLM to answer using only the retrieved context
    system_prompt = f"""You are an astronomy expert. Answer the question using ONLY the
knowledge graph facts provided below. Do not use your own knowledge.
If the facts don't contain enough information, say so.

Knowledge Graph Facts:
{context}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


# ── Step 5: Ask Questions ─────────────────────────────────────────────────────

questions_with_entities = [
    ("What do we know about Mars?", ["Mars"]),
    ("Which planet is larger, Earth or Mars? Give their diameters.", ["Earth", "Mars"]),
    ("What moons does Jupiter have? Which are the Galilean moons?", ["Jupiter"]),
    ("List all planets in the Solar System.", None),  # uses list_entities_of_type
    ("What is the surface temperature of Venus and how does it compare to Mercury?", ["Venus", "Mercury"]),
]

print("\n" + "=" * 20 + " GraphRAG (grounded in knowledge graph) " + "=" * 20 + "\n")

for question, entities in questions_with_entities:
    print("=" * 60)
    print(f"Q: {question}")
    print("-" * 60)

    if entities is None:
        # Q4: use list_entities_of_type as context instead of entity lookup
        context = list_entities_of_type("Planet")
        system_prompt = f"""You are an astronomy expert. Answer the question using ONLY the
knowledge graph facts provided below. Do not use your own knowledge.
If the facts don't contain enough information, say so.

Knowledge Graph Facts:
{context}"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
        )
        answer = response.choices[0].message.content or ""
    else:
        answer = ask_graph_rag(question, entities=entities)

    print(f"A: {answer}")
    print()

# ── Step 6: Comparison — Plain LLM (no knowledge graph) ──────────────────────

print("\n" + "=" * 20 + " Plain LLM (no knowledge graph) " + "=" * 20 + "\n")

for question, _ in questions_with_entities:
    print("=" * 60)
    print(f"Q: {question}")
    print("-" * 60)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer questions concisely."},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )
    print(f"A: {response.choices[0].message.content}")
    print()
