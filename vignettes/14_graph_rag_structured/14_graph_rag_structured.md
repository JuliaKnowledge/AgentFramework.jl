# Structured GraphRAG: From Tabular Data to Knowledge Graphs
AgentFramework.jl

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [The Dataset: Jaguars of the
  Americas](#the-dataset-jaguars-of-the-americas)
- [Step 1: Loading and Preprocessing the
  CSV](#step-1-loading-and-preprocessing-the-csv)
- [Step 2: Loading the Ontology](#step-2-loading-the-ontology)
- [Step 3: Mapping Tabular Data to
  RDF](#step-3-mapping-tabular-data-to-rdf)
- [Step 4: Querying the Knowledge
  Graph](#step-4-querying-the-knowledge-graph)
- [Step 5: Retrieval Functions](#step-5-retrieval-functions)
- [Step 6: Structured GraphRAG
  Pipeline](#step-6-structured-graphrag-pipeline)
- [Step 7: Asking Questions](#step-7-asking-questions)
  - [Individual jaguar lookup](#individual-jaguar-lookup)
  - [Location-based query](#location-based-query)
  - [Conservation status query](#conservation-status-query)
- [Step 8: Structured GraphRAG vs Plain
  LLM](#step-8-structured-graphrag-vs-plain-llm)
- [Step 9: Graph Statistics](#step-9-graph-statistics)
- [Summary](#summary)
  - [Key Takeaways](#key-takeaways)

## Overview

Standard RAG takes structured data (like a CSV), turns it into text
chunks, embeds it, and searches for “similar” chunks — destroying the
exact relationships in your data. **Structured GraphRAG** takes a
different approach: map tabular data directly onto an RDF knowledge
graph using an ontology, preserving every relationship with 100%
fidelity.

In this vignette, you will learn how to:

- Load structured tabular data (CSV) and map it to an RDF knowledge
  graph
- Use an OWL ontology to give semantic meaning to columns and
  relationships
- Query the knowledge graph with SPARQL
- Build a programmatic GraphRAG pipeline that retrieves graph facts and
  generates LLM answers
- Compare structured GraphRAG with plain LLM responses

This follows the same approach as the [Agent Framework
`graph_RAG_structured`
sample](https://github.com/microsoft/agent-framework), ported to Julia
using RDFLib.jl.

## Prerequisites

- **Ollama** running locally:

  ``` bash
  ollama pull gemma3
  ```

## Setup

``` julia
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using AgentFramework
using RDFLib
using DelimitedFiles
```

## The Dataset: Jaguars of the Americas

Our dataset tracks individual jaguars across the Americas — their
locations, monitoring organizations, threats, and status. This is
real-world structured data with multi-valued fields
(semicolon-separated), optional columns, and mixed data types.

``` julia
csv_path = joinpath(@__DIR__, "data", "jaguars.csv")
println(read(csv_path, String))
```

    jaguar_id,name,gender,location,monitoring_org,first_sighted,is_killed,cause_of_death,identification_mark,threats,monitoring_technique,status_notes
    ElJefe,El Jefe,Male,WhetstoneMountains;SantaRitaMountains;Sonora,UnivOfArizona;ConservationCATalyst;NorthernJaguarProject,2011-11-19,false,,Unique spot patterns,,CameraTrap;ScatDetection,
    MachoB,Macho B,Male,BaboquivariMountains,,1996-01-01,true,Euthanasia following kidney failure caused by capture,Mark resembling Betty Boop,SnareTrap,GPSTracking,
    Sombra,Sombra,Male,DosCabezasMountains;ChiricahuaMountains,,2016-11-01,false,,,,CameraTrap;GPSTracking,
    Yooko,Yo'oko,Male,HuachucaMountains;Sonora,,,true,Killed by mountain lion hunter,,Poaching,,
    Cochise,Cochise,Male,HuachucaMountains,,2023-12-20,false,,,,CameraTrap,
    Asa,Asa,Female,Pantanal,PantanalJaguarProject,2008-03-15,false,,Light colored coat with widely spaced rosettes,,,
    Mick,Mick,Male,Pantanal,PantanalJaguarProject,2015-06-22,false,,Large male with prominent rosettes,,,
    Ferinha,Ferinha,Female,Pantanal,PantanalJaguarProject,2012-09-10,false,,Small female with dense rosette pattern,,,
    OshadNukudam,O:ṣhad Ñu:kudam,,Arizona,,,false,,,,,
    Mariposa,Mariposa,Female,HatoLaAurora,PantheraColombia,2009-01-01,false,,,,,
    Cayenita,Cayenita,Female,HatoLaAurora,,,false,,,,,
    Xama,Xamã,Male,AmazonRainforest,,,false,,,WildfireThreat,GPSTracking,Orphaned;Rehabilitated;Released

## Step 1: Loading and Preprocessing the CSV

We parse the CSV into a structured format, handling multi-valued fields
by splitting on semicolons.

``` julia
# Parse CSV manually to handle the complex structure
lines = split(strip(read(csv_path, String)), '\n')
header = [strip(h) for h in split(lines[1], ',')]
println("Columns: ", join(header, ", "))

# Parse each row into a Dict
records = Dict{String,String}[]
for line in lines[2:end]
    # Simple CSV parsing (handles our data which has no quoted commas)
    fields = split(line, ',')
    row = Dict{String,String}()
    for (i, col) in enumerate(header)
        val = i <= length(fields) ? strip(fields[i]) : ""
        row[col] = val
    end
    push!(records, row)
end

println("\nLoaded $(length(records)) jaguar records")
for r in records[1:3]
    println("  $(r["name"]) ($(r["gender"])) — $(r["location"])")
end
```

    Columns: jaguar_id, name, gender, location, monitoring_org, first_sighted, is_killed, cause_of_death, identification_mark, threats, monitoring_technique, status_notes

    Loaded 12 jaguar records
      El Jefe (Male) — WhetstoneMountains;SantaRitaMountains;Sonora
      Macho B (Male) — BaboquivariMountains
      Sombra (Male) — DosCabezasMountains;ChiricahuaMountains

## Step 2: Loading the Ontology

The ontology defines the semantic schema — what classes exist (Jaguar,
Location, ConservationOrganization, Threat, etc.) and how they relate.
This is what makes our graph meaningful rather than just a bag of
triples.

``` julia
ontology_path = joinpath(@__DIR__, "data", "jaguar_ontology.ttl")
kg = RDFGraph()

# Load the OWL ontology
ontology_ttl = read(ontology_path, String)
parse_rdf!(kg, ontology_ttl, TurtleFormat())

println("Ontology loaded: $(length(kg)) triples")

# Show the class hierarchy
classes = sparql_query(kg, """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?class ?parent WHERE {
    ?class a owl:Class .
    OPTIONAL { ?class rdfs:subClassOf ?parent }
}
""")

println("\nOntology classes:")
for r in classes
    cls = split(string(r["class"]), "#")[end]
    parent = haskey(r, "parent") ? " → " * split(string(r["parent"]), "#")[end] : ""
    println("  $cls$parent")
end
```

    Ontology loaded: 226 triples

    Ontology classes:
      GovernmentAgency → ConservationOrganization
      Threat
      EconomicBenefit
      AcademicInstitution → ConservationOrganization
      Person
      Observation
      Reptile → Prey
      Grassland → Habitat
      Rainforest → Forest
      Region → Location
      Forest → Habitat
      Location
      Event
      Mammal → Animal
      Jaguar → BigCat
      IndigenousPerson → Person
      BigCat → Mammal
      JaguarPopulation
      CulturalSignificance
      Herbivore → Prey
      LawEnforcement → Person
      Mesopredator → Prey
      LegalFramework
      NGO → ConservationOrganization
      Prey → Animal
      Fish → Prey
      Habitat
      MountainRange → Location
      MonitoringTechnique
      Researcher → Person
      DietType
      Tourist → Person
      HabitatArea → Location
      ConservationOrganization
      Conservationist → Person
      Shrubland → Habitat
      ConservationEffort
      Livestock → Prey
      Animal
      Wetland → Habitat
      Rancher → Person
      WaterBody → Habitat
      Country → Location
      State → Location

## Step 3: Mapping Tabular Data to RDF

This is the core of structured GraphRAG: deterministically mapping each
CSV row to RDF triples following the ontology. Unlike LLM-based
extraction, this is **100% precise** — no hallucination, no missed
relationships.

``` julia
ont = Namespace("http://example.org/ontology#")
res = Namespace("http://example.org/resource#")

function to_iri(name::AbstractString)
    replace(strip(name), r"[^a-zA-Z0-9]+" => "")
end

# Map each record to RDF triples
for record in records
    id = record["jaguar_id"]
    jaguar = res(id)

    # Core type and identity
    add!(kg, Triple(jaguar, RDF.type, ont("Jaguar")))
    add!(kg, Triple(jaguar, URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                    Literal(record["name"])))
    add!(kg, Triple(jaguar, ont("scientificName"), Literal("Panthera onca")))

    # Gender
    gender = get(record, "gender", "")
    if !isempty(gender)
        add!(kg, Triple(jaguar, ont("hasGender"), Literal(gender)))
    end

    # Locations (multi-valued, semicolon-separated)
    location = get(record, "location", "")
    if !isempty(location)
        for loc in split(location, ";")
            loc = strip(loc)
            loc_iri = res(to_iri(loc))
            add!(kg, Triple(jaguar, ont("occursIn"), loc_iri))
            add!(kg, Triple(loc_iri, RDF.type, ont("Location")))
            add!(kg, Triple(loc_iri, URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                            Literal(loc)))
        end
    end

    # Monitoring organizations (multi-valued)
    monitoring_org = get(record, "monitoring_org", "")
    if !isempty(monitoring_org)
        for org in split(monitoring_org, ";")
            org = strip(org)
            org_iri = res(to_iri(org))
            add!(kg, Triple(jaguar, ont("monitoredByOrg"), org_iri))
            add!(kg, Triple(org_iri, RDF.type, ont("ConservationOrganization")))
            add!(kg, Triple(org_iri, URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                            Literal(org)))
        end
    end

    # First sighted date
    first_sighted = get(record, "first_sighted", "")
    if !isempty(first_sighted)
        add!(kg, Triple(jaguar, ont("hasMonitoringStartDate"),
                        Literal(first_sighted)))
    end

    # Killed status
    is_killed = get(record, "is_killed", "")
    if !isempty(is_killed)
        add!(kg, Triple(jaguar, ont("wasKilled"),
                        Literal(lowercase(is_killed) == "true" ? "true" : "false")))
    end

    # Cause of death
    cause_of_death = get(record, "cause_of_death", "")
    if !isempty(cause_of_death)
        add!(kg, Triple(jaguar, ont("causeOfDeath"), Literal(cause_of_death)))
    end

    # Identification mark
    identification_mark = get(record, "identification_mark", "")
    if !isempty(identification_mark)
        add!(kg, Triple(jaguar, ont("hasIdentificationMark"),
                        Literal(identification_mark)))
    end

    # Threats (multi-valued)
    threats = get(record, "threats", "")
    if !isempty(threats)
        for threat in split(threats, ";")
            threat = strip(threat)
            threat_iri = res(to_iri(threat))
            add!(kg, Triple(jaguar, ont("facesThreat"), threat_iri))
            add!(kg, Triple(threat_iri, RDF.type, ont("Threat")))
            add!(kg, Triple(threat_iri, URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                            Literal(threat)))
        end
    end

    # Monitoring techniques (multi-valued)
    monitoring_technique = get(record, "monitoring_technique", "")
    if !isempty(monitoring_technique)
        for tech in split(monitoring_technique, ";")
            tech = strip(tech)
            tech_iri = res(to_iri(tech))
            add!(kg, Triple(jaguar, ont("monitoredByTechnique"), tech_iri))
            add!(kg, Triple(tech_iri, RDF.type, ont("MonitoringTechnique")))
            add!(kg, Triple(tech_iri, URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
                            Literal(tech)))
        end
    end

    # Status notes
    status_notes = get(record, "status_notes", "")
    if !isempty(status_notes)
        add!(kg, Triple(jaguar, URIRef("http://www.w3.org/2000/01/rdf-schema#comment"),
                        Literal(status_notes)))
    end
end

println("Knowledge graph after mapping: $(length(kg)) triples")
```

    Knowledge graph after mapping: 374 triples

Let’s inspect the graph:

``` julia
println(serialize(kg, TurtleFormat()))
```

    @prefix : <http://example.org/resource#> .
    @prefix ont: <http://example.org/ontology#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix skos: <http://www.w3.org/2004/02/skos/core#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    ont:AcademicInstitution a owl:Class ;
        rdfs:subClassOf ont:ConservationOrganization .

    ont:Animal a owl:Class .

    ont:BigCat a owl:Class ;
        rdfs:subClassOf ont:Mammal .

    ont:CarnivoreDiet a ont:DietType .

    ont:ConservationEffort a owl:Class .

    ont:ConservationOrganization a owl:Class ;
        rdfs:comment "An organization involved in monitoring and protecting wildlife." ;
        rdfs:label "Conservation Organization" .

    ont:Conservationist a owl:Class ;
        rdfs:subClassOf ont:Person .

    ont:Country a owl:Class ;
        rdfs:subClassOf ont:Location .

    ont:CulturalSignificance a owl:Class .

    ont:DietType a owl:Class .

    ont:EconomicBenefit a owl:Class .

    ont:Event a owl:Class .

    ont:Fish a owl:Class ;
        rdfs:subClassOf ont:Prey .

    ont:Forest a owl:Class ;
        rdfs:subClassOf ont:Habitat .

    ont:GovernmentAgency a owl:Class ;
        rdfs:subClassOf ont:ConservationOrganization .

    ont:Grassland a owl:Class ;
        rdfs:subClassOf ont:Habitat .

    ont:Habitat a owl:Class .

    ont:HabitatArea a owl:Class ;
        rdfs:subClassOf ont:Location .

    ont:Herbivore a owl:Class ;
        rdfs:subClassOf ont:Prey .

    ont:IndigenousPerson a owl:Class ;
        rdfs:subClassOf ont:Person .

    ont:Jaguar a owl:Class ;
        rdfs:comment "The Panthera onca species." ;
        rdfs:subClassOf ont:BigCat .

    ont:JaguarPopulation a owl:Class ;
        rdfs:comment "A group or population of jaguars." .

    ont:LawEnforcement a owl:Class ;
        rdfs:subClassOf ont:Person .

    ont:LegalFramework a owl:Class .

    ont:Livestock a owl:Class ;
        rdfs:subClassOf ont:Prey .

    ont:Location a owl:Class .

    ont:Mammal a owl:Class ;
        rdfs:subClassOf ont:Animal .

    ont:Mesopredator a owl:Class ;
        rdfs:subClassOf ont:Prey .

    ont:MonitoringTechnique a owl:Class .

    ont:MountainRange a owl:Class ;
        rdfs:subClassOf ont:Location .

    ont:NGO a owl:Class ;
        rdfs:subClassOf ont:ConservationOrganization .

    ont:Observation a owl:Class ;
        rdfs:comment "An event recording the sighting of an animal." ;
        rdfs:label "Observation" .

    ont:Person a owl:Class ;
        rdfs:comment "A human observer or researcher involved in recording animal sightings." ;
        rdfs:label "Person" .

    ont:Prey a owl:Class ;
        rdfs:subClassOf ont:Animal .

    ont:Rainforest a owl:Class ;
        rdfs:subClassOf ont:Forest .

    ont:Rancher a owl:Class ;
        rdfs:subClassOf ont:Person .

    ont:Region a owl:Class ;
        rdfs:subClassOf ont:Location .

    ont:Reptile a owl:Class ;
        rdfs:subClassOf ont:Prey .

    ont:Researcher a owl:Class ;
        rdfs:subClassOf ont:Person .

    ont:Shrubland a owl:Class ;
        rdfs:subClassOf ont:Habitat .

    ont:State a owl:Class ;
        rdfs:subClassOf ont:Location .

    ont:Threat a owl:Class .

    ont:Tourist a owl:Class ;
        rdfs:subClassOf ont:Person .

    ont:WaterBody a owl:Class ;
        rdfs:subClassOf ont:Habitat .

    ont:Wetland a owl:Class ;
        rdfs:subClassOf ont:Habitat .

    ont:causeOfDeath a owl:DatatypeProperty ;
        rdfs:comment "The cause of death for the jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:string .

    ont:connectsHabitat a owl:ObjectProperty ;
        rdfs:comment "Indicates which habitat areas a wildlife corridor connects." ;
        rdfs:domain ont:WildlifeCorridor ;
        rdfs:range ont:HabitatArea .

    ont:facesThreat a owl:ObjectProperty ;
        rdfs:comment "Indicates a threat faced by the jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range ont:Threat .

    ont:habitat a owl:ObjectProperty ;
        rdfs:domain ont:Animal ;
        rdfs:range ont:Habitat .

    ont:hasAcreage a owl:DatatypeProperty ;
        rdfs:comment "The size of the habitat area in acres." ;
        rdfs:domain ont:HabitatArea ;
        rdfs:range xsd:integer .

    ont:hasDietType a owl:ObjectProperty ;
        rdfs:domain ont:Animal ;
        rdfs:range ont:DietType .

    ont:hasGender a owl:DatatypeProperty ;
        rdfs:comment "Gender of the jaguar (e.g., Male, Female)." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:string .

    ont:hasIdentificationMark a owl:DatatypeProperty ;
        rdfs:comment "Unique spot pattern or other distinguishing mark." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:string .

    ont:hasLastSightingDate a owl:DatatypeProperty ;
        rdfs:comment "Date of the last confirmed sighting of the individual jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:date .

    ont:hasLifespan a owl:DatatypeProperty ;
        rdfs:comment "Lifespan in years." ;
        rdfs:domain ont:Animal ;
        rdfs:range xsd:integer .

    ont:hasMonitoringStartDate a owl:DatatypeProperty ;
        rdfs:comment "Date when monitoring of the individual jaguar began." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:date .

    ont:hasObservation a owl:ObjectProperty ;
        rdfs:comment "Links an animal to one of its observation events." ;
        rdfs:domain ont:Animal ;
        rdfs:range ont:Observation .

    ont:hasOffspring a owl:ObjectProperty ;
        rdfs:comment "Links a jaguar to its offspring." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range ont:Jaguar .

    ont:hasPopulationEstimate a owl:DatatypeProperty ;
        rdfs:comment "Estimated number of jaguars in a population." ;
        rdfs:domain ont:JaguarPopulation ;
        rdfs:range xsd:integer .

    ont:hasReleaseDate a owl:DatatypeProperty ;
        rdfs:comment "Date of the jaguar's release." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:date .

    ont:hasRescueDate a owl:DatatypeProperty ;
        rdfs:comment "Date of the jaguar's rescue." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:date .

    ont:implementsEffort a owl:ObjectProperty ;
        rdfs:comment "Indicates a conservation effort implemented by an organization." ;
        rdfs:domain ont:ConservationOrganization ;
        rdfs:range ont:ConservationEffort .

    ont:isDependentOn a owl:ObjectProperty ;
        rdfs:comment "Indicates if one jaguar population is dependent on another (e.g., for dispersal)." ;
        rdfs:domain ont:JaguarPopulation ;
        rdfs:range ont:JaguarPopulation .

    ont:isOrphaned a owl:DatatypeProperty ;
        rdfs:comment "Indicates if the jaguar was orphaned." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:boolean .

    ont:isRehabilitated a owl:DatatypeProperty ;
        rdfs:comment "Indicates if the jaguar underwent rehabilitation." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:boolean .

    ont:isReleased a owl:DatatypeProperty ;
        rdfs:comment "Indicates if the jaguar was released into the wild." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:boolean .

    ont:locatedIn a owl:ObjectProperty ;
        rdfs:comment "Specifies the state or administrative region in which a habitat is located." ;
        rdfs:domain ont:Habitat ;
        rdfs:range ont:Location .

    ont:locatedInCountry a owl:ObjectProperty ;
        rdfs:comment "Specifies the country in which a state is located." ;
        rdfs:domain ont:State ;
        rdfs:range ont:Country .

    ont:monitoredByOrg a owl:ObjectProperty ;
        rdfs:comment "Links an animal to the conservation organization that monitors it." ;
        rdfs:domain ont:Animal ;
        rdfs:range ont:ConservationOrganization .

    ont:monitoredByTechnique a owl:ObjectProperty ;
        rdfs:comment "Indicates the technique used to monitor the jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range ont:MonitoringTechnique .

    ont:name a owl:DatatypeProperty ;
        rdfs:domain ont:Animal ;
        rdfs:range xsd:string .

    ont:namedBy a owl:ObjectProperty ;
        rdfs:comment "The person or group who named the jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range ont:Person .

    ont:observedBy a owl:ObjectProperty ;
        rdfs:comment "The person who recorded the observation." ;
        rdfs:domain ont:Observation ;
        rdfs:range ont:Person .

    ont:observedDate a owl:DatatypeProperty ;
        rdfs:comment "The date on which the observation took place." ;
        rdfs:domain ont:Observation ;
        rdfs:range xsd:date .

    ont:occursIn a owl:ObjectProperty ;
        rdfs:comment "Indicates a state where an animal has been observed or is known to occur." ;
        rdfs:domain ont:Animal ;
        rdfs:range ont:Location .

    ont:originatesFrom a owl:ObjectProperty ;
        rdfs:comment "Indicates the origin location of a dispersing jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range ont:Location .

    ont:reintroducedBy a owl:ObjectProperty ;
        rdfs:comment "The organization that reintroduced the jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range ont:ConservationOrganization .

    ont:rescuedBy a owl:ObjectProperty ;
        rdfs:comment "The organization that rescued the jaguar." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range ont:ConservationOrganization .

    ont:scientificName a owl:DatatypeProperty ;
        rdfs:domain ont:Animal ;
        rdfs:range xsd:string .

    ont:wasKilled a owl:DatatypeProperty ;
        rdfs:comment "Indicates if the jaguar was killed." ;
        rdfs:domain ont:Jaguar ;
        rdfs:range xsd:boolean .

    :AmazonRainforest a ont:Location ;
        rdfs:label "AmazonRainforest" .

    :Arizona a ont:Location ;
        rdfs:label "Arizona" .

    :Asa a ont:Jaguar ;
        ont:hasGender "Female" ;
        ont:hasIdentificationMark "Light colored coat with widely spaced rosettes" ;
        ont:hasMonitoringStartDate "2008-03-15" ;
        ont:monitoredByOrg :PantanalJaguarProject ;
        ont:occursIn :Pantanal ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "Asa" .

    :BaboquivariMountains a ont:Location ;
        rdfs:label "BaboquivariMountains" .

    :CameraTrap a ont:MonitoringTechnique ;
        rdfs:label "CameraTrap" .

    :Cayenita a ont:Jaguar ;
        ont:hasGender "Female" ;
        ont:occursIn :HatoLaAurora ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "Cayenita" .

    :ChiricahuaMountains a ont:Location ;
        rdfs:label "ChiricahuaMountains" .

    :Cochise a ont:Jaguar ;
        ont:hasGender "Male" ;
        ont:hasMonitoringStartDate "2023-12-20" ;
        ont:monitoredByTechnique :CameraTrap ;
        ont:occursIn :HuachucaMountains ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "Cochise" .

    :ConservationCATalyst a ont:ConservationOrganization ;
        rdfs:label "ConservationCATalyst" .

    :DosCabezasMountains a ont:Location ;
        rdfs:label "DosCabezasMountains" .

    :ElJefe a ont:Jaguar ;
        ont:hasDietType :CarnivoreDiet ;
        ont:hasGender "Male" ;
        ont:hasIdentificationMark "Unique spot patterns" ;
        ont:hasMonitoringStartDate "2011-11-19" ;
        ont:monitoredByOrg :UnivOfArizona,
            :ConservationCATalyst,
            :NorthernJaguarProject ;
        ont:monitoredByTechnique :CameraTrap,
            :ScatDetection ;
        ont:occursIn :WhetstoneMountains,
            :SantaRitaMountains,
            :Sonora ;
        ont:originatesFrom :Sonora ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "El Jefe" .

    :Ferinha a ont:Jaguar ;
        ont:hasGender "Female" ;
        ont:hasIdentificationMark "Small female with dense rosette pattern" ;
        ont:hasMonitoringStartDate "2012-09-10" ;
        ont:monitoredByOrg :PantanalJaguarProject ;
        ont:occursIn :Pantanal ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "Ferinha" .

    :GPSTracking a ont:MonitoringTechnique ;
        rdfs:label "GPSTracking" .

    :HatoLaAurora a ont:Location ;
        rdfs:label "HatoLaAurora" .

    :HuachucaMountains a ont:Location ;
        rdfs:label "HuachucaMountains" .

    :MachoB a ont:Jaguar ;
        ont:causeOfDeath "Euthanasia following kidney failure caused by capture" ;
        ont:facesThreat :SnareTrap ;
        ont:hasGender "Male" ;
        ont:hasIdentificationMark "Mark resembling Betty Boop" ;
        ont:hasMonitoringStartDate "1996-01-01" ;
        ont:monitoredByTechnique :GPSTracking ;
        ont:occursIn :BaboquivariMountains ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "true" ;
        rdfs:label "Macho B" .

    :Mariposa a ont:Jaguar ;
        ont:hasGender "Female" ;
        ont:hasMonitoringStartDate "2009-01-01" ;
        ont:monitoredByOrg :PantheraColombia ;
        ont:occursIn :HatoLaAurora ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "Mariposa" .

    :Mexico a ont:Country ;
        rdfs:label "Mexico" .

    :Mick a ont:Jaguar ;
        ont:hasGender "Male" ;
        ont:hasIdentificationMark "Large male with prominent rosettes" ;
        ont:hasMonitoringStartDate "2015-06-22" ;
        ont:monitoredByOrg :PantanalJaguarProject ;
        ont:occursIn :Pantanal ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "Mick" .

    :NorthernJaguarProject a ont:ConservationOrganization ;
        rdfs:label "NorthernJaguarProject" .

    :OshadNukudam a ont:Jaguar ;
        ont:occursIn :Arizona ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "O:ṣhad Ñu:kudam" .

    :Pantanal a ont:Location ;
        rdfs:label "Pantanal" .

    :PantanalJaguarProject a ont:ConservationOrganization ;
        rdfs:label "PantanalJaguarProject" .

    :PantheraColombia a ont:ConservationOrganization ;
        rdfs:label "PantheraColombia" .

    :Poaching a ont:Threat ;
        rdfs:label "Poaching" .

    :SantaRitaMountains a ont:Location ;
        rdfs:label "SantaRitaMountains" .

    :ScatDetection a ont:MonitoringTechnique ;
        rdfs:label "ScatDetection" .

    :SnareTrap a ont:Threat ;
        rdfs:label "SnareTrap" .

    :Sombra a ont:Jaguar ;
        ont:hasGender "Male" ;
        ont:hasMonitoringStartDate "2016-11-01" ;
        ont:monitoredByTechnique :CameraTrap,
            :GPSTracking ;
        ont:occursIn :DosCabezasMountains,
            :ChiricahuaMountains ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:label "Sombra" .

    :Sonora a ont:Location ;
        rdfs:label "Sonora" .

    :UnivOfArizona a ont:ConservationOrganization ;
        rdfs:label "UnivOfArizona" .

    :WhetstoneMountains a ont:Location ;
        rdfs:label "WhetstoneMountains" .

    :WildfireThreat a ont:Threat ;
        rdfs:label "WildfireThreat" .

    :Xama a ont:Jaguar ;
        ont:facesThreat :WildfireThreat ;
        ont:hasGender "Male" ;
        ont:monitoredByTechnique :GPSTracking ;
        ont:occursIn :AmazonRainforest ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "false" ;
        rdfs:comment "Orphaned;Rehabilitated;Released" ;
        rdfs:label "Xamã" .

    :Yooko a ont:Jaguar ;
        ont:causeOfDeath "Killed by mountain lion hunter" ;
        ont:facesThreat :Poaching ;
        ont:hasGender "Male" ;
        ont:occursIn :HuachucaMountains,
            :Sonora ;
        ont:scientificName "Panthera onca" ;
        ont:wasKilled "true" ;
        rdfs:label "Yo'oko" .

## Step 4: Querying the Knowledge Graph

With the data mapped to RDF, we can now run precise SPARQL queries.
Unlike vector similarity search, these return **exact** results.

``` julia
# Query 1: All jaguars and their locations
results = sparql_query(kg, """
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name ?location WHERE {
    ?j a ont:Jaguar ;
       rdfs:label ?name ;
       ont:occursIn ?loc .
    ?loc rdfs:label ?location .
}
ORDER BY ?name
""")

println("Jaguars and their locations:")
for r in results
    println("  $(string(r["name"])) → $(string(r["location"]))")
end
```

    Jaguars and their locations:
      Asa → Pantanal
      Cayenita → HatoLaAurora
      Cochise → HuachucaMountains
      El Jefe → Sonora
      El Jefe → WhetstoneMountains
      El Jefe → SantaRitaMountains
      Ferinha → Pantanal
      Macho B → BaboquivariMountains
      Mariposa → HatoLaAurora
      Mick → Pantanal
      O:ṣhad Ñu:kudam → Arizona
      Sombra → ChiricahuaMountains
      Sombra → DosCabezasMountains
      Xamã → AmazonRainforest
      Yo'oko → Sonora
      Yo'oko → HuachucaMountains

``` julia
# Query 2: Which jaguars were killed?
results = sparql_query(kg, """
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?name ?cause WHERE {
    ?j a ont:Jaguar ;
       rdfs:label ?name ;
       ont:wasKilled "true" .
    OPTIONAL { ?j ont:causeOfDeath ?cause }
}
""")

println("Jaguars that were killed:")
for r in results
    cause = haskey(r, "cause") ? " — $(string(r["cause"]))" : ""
    println("  $(string(r["name"]))$cause")
end
```

    Jaguars that were killed:
      Yo'oko — Killed by mountain lion hunter
      Macho B — Euthanasia following kidney failure caused by capture

``` julia
# Query 3: Which organizations monitor which jaguars?
results = sparql_query(kg, """
PREFIX ont: <http://example.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?org ?name WHERE {
    ?j a ont:Jaguar ;
       rdfs:label ?name ;
       ont:monitoredByOrg ?o .
    ?o rdfs:label ?org .
}
ORDER BY ?org
""")

println("Monitoring organizations and their jaguars:")
current_org = ""
for r in results
    org = string(r["org"])
    if org != current_org
        println("  $org:")
        current_org = org
    end
    println("    - $(string(r["name"]))")
end
```

    Monitoring organizations and their jaguars:
      ConservationCATalyst:
        - El Jefe
      NorthernJaguarProject:
        - El Jefe
      PantanalJaguarProject:
        - Asa
        - Ferinha
        - Mick
      PantheraColombia:
        - Mariposa
      UnivOfArizona:
        - El Jefe

## Step 5: Retrieval Functions

We define retrieval functions that extract structured context from the
knowledge graph for the LLM.

``` julia
"""Query all facts about a specific jaguar by name."""
function query_jaguar(name::String)
    results = sparql_query(kg, """
    PREFIX ont: <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?pred ?obj WHERE {
        ?j a ont:Jaguar ;
           rdfs:label "$name" ;
           ?pred ?obj .
    }
    """)

    isempty(results) && return "No jaguar found with name '$name'."
    lines = String[]
    for r in results
        pred = split(string(r["pred"]), r"[#/]")[end]
        obj = string(r["obj"])
        # Skip the type triple
        pred == "type" && continue
        push!(lines, "$pred: $obj")
    end
    return "Facts about $name:\n" * join(lines, "\n")
end

"""List all jaguars in a given location."""
function query_location(location::String)
    results = sparql_query(kg, """
    PREFIX ont: <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?name WHERE {
        ?j a ont:Jaguar ;
           rdfs:label ?name ;
           ont:occursIn ?loc .
        ?loc rdfs:label ?location .
        FILTER(CONTAINS(LCASE(STR(?location)), LCASE("$location")))
    }
    """)

    isempty(results) && return "No jaguars found in '$location'."
    names = [string(r["name"]) for r in results]
    return "Jaguars in $location: " * join(names, ", ")
end

"""Run a custom SPARQL query."""
function run_sparql(query::String)
    try
        results = sparql_query(kg, query)
        isempty(results) && return "Query returned no results."
        lines = String[]
        for row in results
            parts = ["$k=$(v isa Literal ? string(v) : split(string(v), r"[#/]")[end])" for (k, v) in row]
            push!(lines, join(parts, ", "))
        end
        return "Results ($(length(results)) rows):\n" * join(lines, "\n")
    catch e
        return "SPARQL error: $(sprint(showerror, e))"
    end
end
nothing
```

Let’s verify the retrieval functions:

``` julia
println(query_jaguar("El Jefe"))
println()
println(query_location("Pantanal"))
```

    Facts about El Jefe:
    originatesFrom: http://example.org/resource#Sonora
    wasKilled: false
    hasMonitoringStartDate: 2011-11-19
    monitoredByOrg: http://example.org/resource#NorthernJaguarProject
    monitoredByOrg: http://example.org/resource#ConservationCATalyst
    monitoredByOrg: http://example.org/resource#UnivOfArizona
    monitoredByTechnique: http://example.org/resource#ScatDetection
    monitoredByTechnique: http://example.org/resource#CameraTrap
    hasDietType: http://example.org/resource#CarnivoreDiet
    label: El Jefe
    hasIdentificationMark: Unique spot patterns
    scientificName: Panthera onca
    hasGender: Male
    occursIn: http://example.org/resource#Sonora
    occursIn: http://example.org/resource#WhetstoneMountains
    occursIn: http://example.org/resource#SantaRitaMountains

    Jaguars in Pantanal: Asa, Ferinha, Mick

## Step 6: Structured GraphRAG Pipeline

Now we combine graph retrieval with LLM generation. The pipeline:

1.  **Retrieve**: Query the knowledge graph for relevant facts
2.  **Generate**: Pass retrieved facts as context to the LLM

``` julia
client = OllamaChatClient(model="gemma3:latest")

"""Ask a question using structured GraphRAG."""
function ask_structured_rag(question::String; jaguars::Vector{String}=String[],
                            locations::Vector{String}=String[],
                            sparql::String="")
    # Retrieve context from knowledge graph
    context_parts = String[]
    for name in jaguars
        push!(context_parts, query_jaguar(name))
    end
    for loc in locations
        push!(context_parts, query_location(loc))
    end
    if !isempty(sparql)
        push!(context_parts, run_sparql(sparql))
    end
    context = join(context_parts, "\n\n")

    rag_agent = Agent(
        name="JaguarExpert",
        instructions="""You are a wildlife conservation expert specializing in jaguars.
Answer the question using ONLY the knowledge graph facts provided below.
Do not use your own knowledge. If the facts don't contain enough information, say so.

Knowledge Graph Facts:
$context""",
        client=client,
        options=ChatOptions(temperature=0.1),
    )
    response = run_agent(rag_agent, question)
    return get_text(response)
end
nothing
```

## Step 7: Asking Questions

### Individual jaguar lookup

``` julia
answer = ask_structured_rag(
    "Tell me about El Jefe. Where has he been seen and who monitors him?",
    jaguars=["El Jefe"],
)
println(answer)
```

    El Jefe is a male jaguar (*Panthera onca*) with unique spot patterns. He originates from Sonora and has been seen in Sonora, the Whetstone Mountains, and the Santa Rita Mountains. He is monitored by the Northern Jaguar Project, ConservationCATalyst, the University of Arizona, using scat detection and camera trap techniques.

### Location-based query

``` julia
answer = ask_structured_rag(
    "Which jaguars have been documented in the Pantanal region?",
    locations=["Pantanal"],
    jaguars=["Asa", "Mick", "Ferinha"],
)
println(answer)
```

    Asa, Ferinha, and Mick have been documented in the Pantanal region.

### Conservation status query

``` julia
answer = ask_structured_rag(
    "Which jaguars have been killed and what were the causes?",
    jaguars=["Macho B", "Yo'oko"],
    sparql="""
    PREFIX ont: <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?name ?cause WHERE {
        ?j a ont:Jaguar ; rdfs:label ?name ; ont:wasKilled "true" .
        OPTIONAL { ?j ont:causeOfDeath ?cause }
    }""",
)
println(answer)
```

    name=Yo'oko, cause=Killed by mountain lion hunter
    name=Macho B, cause=Euthanasia following kidney failure caused by capture

## Step 8: Structured GraphRAG vs Plain LLM

Let’s compare the structured GraphRAG pipeline against a plain LLM:

``` julia
plain_agent = Agent(
    name="PlainAssistant",
    instructions="You are a helpful assistant. Answer questions concisely.",
    client=client,
    options=ChatOptions(temperature=0.1),
)

question = "What monitoring techniques are used to track jaguars in the US?"

println("=" ^ 60)
println("Question: $question")
println("=" ^ 60)

println("\n--- Structured GraphRAG (grounded in knowledge graph) ---")
answer_rag = ask_structured_rag(question,
    sparql="""
    PREFIX ont: <http://example.org/ontology#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?name ?technique WHERE {
        ?j a ont:Jaguar ; rdfs:label ?name ;
           ont:monitoredByTechnique ?t .
        ?t rdfs:label ?technique .
    }""",
)
println(answer_rag)

println("\n--- Plain LLM (no knowledge graph) ---")
response_plain = run_agent(plain_agent, question)
println(get_text(response_plain))
```

    ============================================================
    Question: What monitoring techniques are used to track jaguars in the US?
    ============================================================

    --- Structured GraphRAG (grounded in knowledge graph) ---
    Based on the provided knowledge graph, the following monitoring techniques are used to track jaguars in the US:

    *   CameraTrap
    *   GPSTracking
    *   ScatDetection

    --- Plain LLM (no knowledge graph) ---
    *   **GPS Collars:** Most common for tracking movements and home ranges.
    *   **Camera Traps:** Used to monitor jaguar presence and behavior.
    *   **Telemetry:** Utilizing satellite tracking for longer-term monitoring.
    *   **Sightings Reports:** Citizen science data contributes to tracking.

The structured GraphRAG answer is grounded in the exact data from the
CSV, while the plain LLM provides generic knowledge that may not match
the specific dataset.

## Step 9: Graph Statistics

``` julia
subjects_set = Set(string(t.subject) for t in triples(kg))
predicates_set = Set(string(t.predicate) for t in triples(kg))

# Count instances by type
jaguar_count = length(sparql_query(kg, """
    PREFIX ont: <http://example.org/ontology#>
    SELECT ?j WHERE { ?j a ont:Jaguar }
"""))
location_count = length(sparql_query(kg, """
    PREFIX ont: <http://example.org/ontology#>
    SELECT ?l WHERE { ?l a ont:Location }
"""))
org_count = length(sparql_query(kg, """
    PREFIX ont: <http://example.org/ontology#>
    SELECT ?o WHERE { ?o a ont:ConservationOrganization }
"""))

println("Knowledge Graph Statistics:")
println("  Total triples: ", length(kg))
println("  Unique subjects: ", length(subjects_set))
println("  Unique predicates: ", length(predicates_set))
println()
println("Instance counts:")
println("  Jaguars: $jaguar_count")
println("  Locations: $location_count")
println("  Conservation Orgs: $org_count")
```

    Knowledge Graph Statistics:
      Total triples: 374
      Unique subjects: 115
      Unique predicates: 18

    Instance counts:
      Jaguars: 12
      Locations: 11
      Conservation Orgs: 5

## Summary

| Aspect | Structured GraphRAG | Text-based GraphRAG (Vignette 13) |
|----|----|----|
| **Input** | Tabular CSV data | Unstructured text passages |
| **Extraction** | Deterministic column mapping | LLM-based entity extraction |
| **Accuracy** | 100% precise | Probabilistic (can miss/hallucinate) |
| **Ontology** | OWL schema defines classes & properties | Implicit from extraction prompt |
| **Multi-valued fields** | Split on delimiter, map each | LLM must recognize lists |
| **Speed** | Instant (no LLM needed for ingestion) | Slow (LLM call per passage) |

### Key Takeaways

1.  **Don’t embed, map**: Structured data should be mapped directly to a
    knowledge graph, not embedded into vectors.
2.  **Ontology-driven**: An OWL ontology gives semantic meaning to
    columns and ensures consistent relationships.
3.  **Zero hallucination ingestion**: Unlike LLM-based extraction,
    tabular mapping is deterministic and complete.
4.  **SPARQL precision**: Exact queries replace fuzzy similarity search.
5.  **RDFLib.jl**: Provides the full RDF stack for loading ontologies,
    mapping data, and querying with SPARQL.
