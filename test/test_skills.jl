using Test
using AgentFramework

@testset "Skills" begin
    # ── SkillResource ────────────────────────────────────────────────────────

    @testset "SkillResource construction with static content" begin
        r = SkillResource(name="readme", description="A readme", content="Hello world")
        @test r.name == "readme"
        @test r.description == "A readme"
        @test r.content == "Hello world"
        @test r.mime_type == "text/plain"
        @test r.fn === nothing
    end

    @testset "SkillResource with callable function" begin
        r = SkillResource(name="dynamic", fn=() -> "computed")
        @test r.fn !== nothing
        @test r.content === nothing
    end

    @testset "get_resource_content for static" begin
        r = SkillResource(name="s", content="static text")
        @test get_resource_content(r) == "static text"
    end

    @testset "get_resource_content for callable" begin
        counter = Ref(0)
        r = SkillResource(name="c", fn=() -> begin counter[] += 1; "call $(counter[])" end)
        @test get_resource_content(r) == "call 1"
        @test get_resource_content(r) == "call 2"
    end

    @testset "get_resource_content with neither content nor fn" begin
        r = SkillResource(name="empty")
        @test get_resource_content(r) == ""
    end

    # ── Skill ────────────────────────────────────────────────────────────────

    @testset "Skill construction with defaults" begin
        s = Skill(name="test")
        @test s.name == "test"
        @test s.description == ""
        @test s.version == "1.0.0"
        @test s.instructions == ""
        @test isempty(s.resources)
        @test isempty(s.tags)
        @test s.source_path === nothing
    end

    # ── SKILL.md Parsing ─────────────────────────────────────────────────────

    @testset "parse_skill_md_content with frontmatter" begin
        content = """
        ---
        name: TestSkill
        description: A test skill
        version: 2.0.0
        tags: [test, demo]
        ---

        # How to use

        This skill does testing things.
        """
        skill = parse_skill_md_content(content)
        @test skill.name == "TestSkill"
        @test skill.description == "A test skill"
        @test skill.version == "2.0.0"
        @test "test" in skill.tags
        @test "demo" in skill.tags
        @test contains(skill.instructions, "How to use")
    end

    @testset "parse_skill_md_content without frontmatter" begin
        content = "Just some instructions here."
        skill = parse_skill_md_content(content)
        @test skill.name == "unnamed"
        @test skill.instructions == "Just some instructions here."
    end

    @testset "parse_skill_md_content without frontmatter with source_path" begin
        skill = parse_skill_md_content("Instructions"; source_path="/a/my_skill/SKILL.md")
        @test skill.name == "my_skill"
    end

    @testset "parse_skill_md from file" begin
        mktempdir() do tmpdir
            skill_dir = joinpath(tmpdir, "my_skill")
            mkpath(skill_dir)
            write(joinpath(skill_dir, "SKILL.md"), """
            ---
            name: FileSkill
            description: From file
            version: 3.0.0
            ---

            File-based instructions.
            """)
            skill = parse_skill_md(joinpath(skill_dir, "SKILL.md"))
            @test skill.name == "FileSkill"
            @test skill.version == "3.0.0"
            @test skill.source_path == joinpath(skill_dir, "SKILL.md")
        end
    end

    # ── Tag Parsing ──────────────────────────────────────────────────────────

    @testset "_parse_tags with brackets" begin
        tags = AgentFramework._parse_tags("[alpha, beta, gamma]")
        @test tags == ["alpha", "beta", "gamma"]
    end

    @testset "_parse_tags without brackets" begin
        tags = AgentFramework._parse_tags("one, two")
        @test tags == ["one", "two"]
    end

    @testset "_parse_tags empty string" begin
        @test isempty(AgentFramework._parse_tags(""))
    end

    # ── Directory Scanner ────────────────────────────────────────────────────

    @testset "discover_skills finds SKILL.md files" begin
        mktempdir() do tmpdir
            skill_dir = joinpath(tmpdir, "my_skill")
            mkpath(skill_dir)
            write(joinpath(skill_dir, "SKILL.md"), """
            ---
            name: DiscoverMe
            description: Found by scanner
            ---

            Scanner instructions.
            """)
            skills = discover_skills(tmpdir)
            @test length(skills) == 1
            @test skills[1].name == "DiscoverMe"
        end
    end

    @testset "discover_skills respects max_depth" begin
        mktempdir() do tmpdir
            # Create a deeply nested skill
            deep = joinpath(tmpdir, "a", "b", "c", "d")
            mkpath(deep)
            write(joinpath(deep, "SKILL.md"), """
            ---
            name: DeepSkill
            ---

            Deep.
            """)
            # max_depth=1 should not find it (depth 0=a, 1=b, 2=c, 3=d)
            skills = discover_skills(tmpdir; max_depth=1)
            @test isempty(skills)

            # max_depth=4 should find it
            skills = discover_skills(tmpdir; max_depth=4)
            @test length(skills) == 1
            @test skills[1].name == "DeepSkill"
        end
    end

    @testset "discover_skills skips symlinks by default" begin
        mktempdir() do tmpdir
            real_dir = joinpath(tmpdir, "real_skill")
            mkpath(real_dir)
            write(joinpath(real_dir, "SKILL.md"), """
            ---
            name: RealSkill
            ---

            Real.
            """)
            link_dir = joinpath(tmpdir, "link_skill")
            try
                symlink(real_dir, link_dir)
            catch
                @info "Symlink creation not supported; skipping symlink test"
                return
            end

            # Without following symlinks, only the real one is found
            skills = discover_skills(tmpdir; follow_symlinks=false)
            names = [s.name for s in skills]
            @test "RealSkill" in names
            # The link_skill directory is a symlink and should be skipped
            link_found = count(n -> n == "RealSkill", names)
            @test link_found == 1
        end
    end

    @testset "Empty skills directory returns empty list" begin
        mktempdir() do tmpdir
            skills = discover_skills(tmpdir)
            @test isempty(skills)
        end
    end

    @testset "discover_skills throws on non-directory" begin
        mktempdir() do tmpdir
            @test_throws ArgumentError discover_skills(joinpath(tmpdir, "nonexistent"))
        end
    end

    # ── Security ─────────────────────────────────────────────────────────────

    @testset "_is_safe_path allows valid paths" begin
        mktempdir() do tmpdir
            child = joinpath(tmpdir, "child")
            mkpath(child)
            @test AgentFramework._is_safe_path(child, tmpdir) == true
        end
    end

    @testset "_is_safe_path blocks path traversal" begin
        mktempdir() do tmpdir
            # A path outside the root
            @test AgentFramework._is_safe_path("/tmp", tmpdir) == false
        end
    end

    @testset "_is_safe_path returns false for nonexistent paths" begin
        mktempdir() do tmpdir
            @test AgentFramework._is_safe_path(joinpath(tmpdir, "nope", "nada"), tmpdir) == false
        end
    end

    # ── File Resources ───────────────────────────────────────────────────────

    @testset "File resources discovered automatically" begin
        mktempdir() do tmpdir
            skill_dir = joinpath(tmpdir, "res_skill")
            mkpath(skill_dir)
            write(joinpath(skill_dir, "SKILL.md"), """
            ---
            name: ResourceSkill
            ---

            Has resources.
            """)
            write(joinpath(skill_dir, "helper.jl"), "println(\"hello\")")
            write(joinpath(skill_dir, "data.json"), "{\"key\": \"value\"}")
            write(joinpath(skill_dir, "image.png"), "not scanned")

            skills = discover_skills(tmpdir)
            @test length(skills) == 1
            skill = skills[1]
            @test haskey(skill.resources, "helper.jl")
            @test haskey(skill.resources, "data.json")
            @test !haskey(skill.resources, "image.png")

            # Check MIME types
            @test skill.resources["helper.jl"].mime_type == "text/x-julia"
            @test skill.resources["data.json"].mime_type == "application/json"

            # Lazy loading works
            @test get_resource_content(skill.resources["helper.jl"]) == "println(\"hello\")"
            @test get_resource_content(skill.resources["data.json"]) == "{\"key\": \"value\"}"
        end
    end

    # ── SkillsProvider ───────────────────────────────────────────────────────

    @testset "SkillsProvider construction" begin
        sp = SkillsProvider()
        @test isempty(sp.skills)
        @test contains(sp.system_prompt_template, "{skills}")
    end

    @testset "add_skill! adds to provider" begin
        sp = SkillsProvider()
        s = Skill(name="Added")
        add_skill!(sp, s)
        @test length(sp.skills) == 1
        @test sp.skills[1].name == "Added"
    end

    @testset "add_skills_from_directory! loads from filesystem" begin
        mktempdir() do tmpdir
            skill_dir = joinpath(tmpdir, "dir_skill")
            mkpath(skill_dir)
            write(joinpath(skill_dir, "SKILL.md"), """
            ---
            name: DirSkill
            ---

            From directory.
            """)
            sp = SkillsProvider()
            add_skills_from_directory!(sp, tmpdir)
            @test length(sp.skills) == 1
            @test sp.skills[1].name == "DirSkill"
        end
    end

    # ── Skill Lookup ─────────────────────────────────────────────────────────

    @testset "_find_skill case-insensitive lookup" begin
        sp = SkillsProvider()
        add_skill!(sp, Skill(name="MySkill"))
        @test AgentFramework._find_skill(sp, "myskill") !== nothing
        @test AgentFramework._find_skill(sp, "MYSKILL") !== nothing
        @test AgentFramework._find_skill(sp, "MySkill") !== nothing
        @test AgentFramework._find_skill(sp, "other") === nothing
    end

    # ── Formatting ───────────────────────────────────────────────────────────

    @testset "_format_skill_list produces readable output" begin
        skills = [
            Skill(name="Alpha", description="First", version="1.0.0", tags=["a"]),
            Skill(name="Beta", description="Second", version="2.0.0"),
        ]
        output = AgentFramework._format_skill_list(skills)
        @test contains(output, "**Alpha**")
        @test contains(output, "v1.0.0")
        @test contains(output, "[a]")
        @test contains(output, "**Beta**")
    end

    @testset "_format_skill_instructions includes all sections" begin
        s = Skill(
            name="Full",
            description="Described",
            version="1.2.3",
            instructions="Do this",
            resources=Dict("r1" => SkillResource(name="r1", description="Resource one")),
        )
        output = AgentFramework._format_skill_instructions(s)
        @test contains(output, "# Full (v1.2.3)")
        @test contains(output, "Described")
        @test contains(output, "## Instructions")
        @test contains(output, "Do this")
        @test contains(output, "## Available Resources")
        @test contains(output, "**r1**")
    end

    # ── Skill Tools ──────────────────────────────────────────────────────────

    @testset "load_skill tool returns instructions" begin
        sp = SkillsProvider()
        add_skill!(sp, Skill(name="ToolTest", instructions="Use me carefully"))
        tool = AgentFramework._make_load_skill_tool(sp)
        result = tool.func(Dict{String, Any}("skill_name" => "ToolTest"))
        @test contains(result, "ToolTest")
        @test contains(result, "Use me carefully")
    end

    @testset "load_skill tool returns error for unknown skill" begin
        sp = SkillsProvider()
        add_skill!(sp, Skill(name="Known"))
        tool = AgentFramework._make_load_skill_tool(sp)
        result = tool.func(Dict{String, Any}("skill_name" => "Unknown"))
        @test contains(result, "Error")
        @test contains(result, "not found")
        @test contains(result, "Known")
    end

    @testset "read_skill_resource tool returns content" begin
        sp = SkillsProvider()
        s = Skill(
            name="ResSkill",
            resources=Dict("data" => SkillResource(name="data", content="payload")),
        )
        add_skill!(sp, s)
        tool = AgentFramework._make_read_resource_tool(sp)
        result = tool.func(Dict{String, Any}("skill_name" => "ResSkill", "resource_name" => "data"))
        @test result == "payload"
    end

    @testset "read_skill_resource tool returns error for unknown skill" begin
        sp = SkillsProvider()
        tool = AgentFramework._make_read_resource_tool(sp)
        result = tool.func(Dict{String, Any}("skill_name" => "Nope", "resource_name" => "x"))
        @test contains(result, "Error")
        @test contains(result, "not found")
    end

    @testset "read_skill_resource tool returns error for unknown resource" begin
        sp = SkillsProvider()
        add_skill!(sp, Skill(
            name="HasRes",
            resources=Dict("a" => SkillResource(name="a", content="aa")),
        ))
        tool = AgentFramework._make_read_resource_tool(sp)
        result = tool.func(Dict{String, Any}("skill_name" => "HasRes", "resource_name" => "missing"))
        @test contains(result, "Error")
        @test contains(result, "not found in skill")
    end

    # ── before_run! Integration ──────────────────────────────────────────────

    @testset "before_run! injects instructions and tools" begin
        sp = SkillsProvider()
        add_skill!(sp, Skill(name="Injected", description="A skill"))
        session = AgentSession()
        ctx = SessionContext()
        state = Dict{String, Any}()
        before_run!(sp, nothing, session, ctx, state)
        @test length(ctx.instructions) == 1
        @test contains(ctx.instructions[1], "Injected")
        @test length(ctx.tools) == 2
        tool_names = [t.name for t in ctx.tools]
        @test "load_skill" in tool_names
        @test "read_skill_resource" in tool_names
    end

    @testset "before_run! does nothing with empty skills" begin
        sp = SkillsProvider()
        ctx = SessionContext()
        before_run!(sp, nothing, AgentSession(), ctx, Dict{String, Any}())
        @test isempty(ctx.instructions)
        @test isempty(ctx.tools)
    end
end
