using AgentFramework
using RDFLib
using Test

@testset "RDFLib Memory Store" begin
    @testset "ontology labels enrich retrieval" begin
        store = RDFMemoryStore(rdflib=RDFLib)
        load_ontology!(store, """
            @prefix ex: <http://example.org/ontology#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

            ex:AuthenticationIssue rdfs:label "authentication" .
            ex:TokenRefresh rdfs:label "token refresh" .
        """)

        add_memories!(store, [
            MemoryRecord(
                scope = "user-1",
                kind = :procedural,
                role = :assistant,
                content = "Refresh the access token before retrying the protected request.",
                metadata = Dict{String, Any}(
                    "concepts" => [
                        "http://example.org/ontology#AuthenticationIssue",
                        "http://example.org/ontology#TokenRefresh",
                    ],
                    "task" => "recover from expired credentials",
                ),
            ),
        ])

        results = search_memories(store, "authentication token refresh"; scope="user-1", limit=3)
        @test length(results) == 1
        @test results[1].record.kind == :procedural
        @test occursin("Refresh the access token", results[1].record.content)

        memories = get_memories(store; scope="user-1")
        @test length(memories) == 1
        @test memories[1].metadata["task"] == "recover from expired credentials"

        clear_memories!(store; scope="user-1")
        @test isempty(get_memories(store; scope="user-1"))
    end
end
