# Evaluation

The evaluation framework provides tools for testing agent and workflow quality.
Define [`EvalItem`](@ref) test cases with expected outputs, run them through
an agent or workflow, and inspect [`EvalResults`](@ref) for pass/fail
summaries and individual scores.

## Core Types

```@docs
AgentFramework.AbstractEvaluator
AgentFramework.LocalEvaluator
AgentFramework.EvalNotPassedError
```

## Data Types

```@docs
AgentFramework.EvalItem
AgentFramework.EvalResults
AgentFramework.EvalItemResult
AgentFramework.EvalScoreResult
AgentFramework.ExpectedToolCall
AgentFramework.CheckResult
```

## Conversation Splitting

```@docs
AgentFramework.ConversationSplitter
AgentFramework.SPLIT_LAST_TURN
AgentFramework.SPLIT_FULL
AgentFramework.split_last_turn
AgentFramework.split_full
AgentFramework.split_messages
AgentFramework.per_turn_items
```

## Query and Response Extraction

```@docs
AgentFramework.eval_query
AgentFramework.eval_response
```

## Result Inspection

```@docs
AgentFramework.eval_passed
AgentFramework.eval_failed
AgentFramework.eval_total
AgentFramework.all_passed
AgentFramework.raise_for_status
AgentFramework.is_error
AgentFramework.is_passed
AgentFramework.is_failed
```

## Built-in Checks

```@docs
AgentFramework.keyword_check
AgentFramework.tool_called_check
AgentFramework.tool_calls_present
AgentFramework.tool_call_args_match
```

## Running Evaluations

```@docs
AgentFramework.make_evaluator
AgentFramework.evaluate
AgentFramework.evaluate_agent
AgentFramework.evaluate_workflow
AgentFramework.WorkflowAgent
```
