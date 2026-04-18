# Skills

Skills are reusable bundles of instructions, tools, and resources that can be
dynamically loaded into an agent's context. The [`SkillsProvider`](@ref)
context provider injects matching skills before each agent run. Skill sources
can be static, directory-based, or composed with deduplication and filtering.

## Core Types

```@docs
AgentFramework.SkillResource
AgentFramework.Skill
AgentFramework.SkillsProvider
```

## Skill Sources

```@docs
AgentFramework.AbstractSkillSource
AgentFramework.StaticSkillSource
AgentFramework.DirectorySkillSource
AgentFramework.DeduplicatingSkillSource
AgentFramework.FilteringSkillSource
AgentFramework.AggregatingSkillSource
```

## Source Builder

```@docs
AgentFramework.SkillSourceBuilder
AgentFramework.add_directory!
AgentFramework.add_skills!
AgentFramework.add_source!
AgentFramework.deduplicate!
AgentFramework.filter_by!
AgentFramework.build(::AgentFramework.SkillSourceBuilder)
AgentFramework.load_skills!
AgentFramework.get_skills
```

## Parsing and Discovery

```@docs
AgentFramework.get_resource_content
AgentFramework.parse_skill_md
AgentFramework.parse_skill_md_content
AgentFramework.discover_skills
AgentFramework.add_skill!
AgentFramework.add_skills_from_directory!
```

## Constants

```@docs
AgentFramework.SKILL_FILENAME
AgentFramework.DEFAULT_SCAN_EXTENSIONS
```
