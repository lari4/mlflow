# MLflow Agent Pipelines Documentation

This document describes all agent pipelines and workflows in MLflow, showing how prompts, data, and components interact to enable AI evaluation, optimization, and observability.

## Table of Contents

1. [Standard Evaluation Pipeline](#standard-evaluation-pipeline)
2. [Trace-Based Judge Pipeline](#trace-based-judge-pipeline)
3. [RAG Evaluation Pipelines](#rag-evaluation-pipelines)
4. [Prompt Optimization Pipeline (GEPA)](#prompt-optimization-pipeline-gepa)
5. [Built-in Judge Pipeline](#built-in-judge-pipeline)
6. [Custom Prompt Judge Pipeline](#custom-prompt-judge-pipeline)

---

## Standard Evaluation Pipeline

**Purpose**: Evaluate model outputs using multiple scorers (judges/metrics) in parallel, optionally with prediction function.

**Main Entry Point**: `mlflow.genai.evaluation.harness.run()`
**File**: `/home/user/mlflow/mlflow/genai/evaluation/harness.py`

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STANDARD EVALUATION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   Dataset    │
                                    │  (List[Dict])│
                                    └──────┬───────┘
                                           │
                                           ▼
                               ┌───────────────────────┐
                               │   Convert to EvalItems │
                               │  - inputs              │
                               │  - outputs             │
                               │  - expectations        │
                               │  - trace               │
                               │  - tags                │
                               └───────────┬───────────┘
                                           │
                                           ▼
                        ┌──────────────────────────────────────┐
                        │   Parallel Execution (ThreadPool)    │
                        │   For each EvalItem:                 │
                        └──────────────────┬───────────────────┘
                                           │
                        ┌──────────────────┴────────────────────┐
                        │                                       │
                        ▼                                       ▼
           ┌─────────────────────────┐            ┌─────────────────────────┐
           │   If predict_fn exists  │            │   Use existing outputs  │
           │   ┌──────────────────┐  │            │   from dataset          │
           │   │  Run predict_fn  │  │            └─────────────────────────┘
           │   │  with inputs     │  │
           │   └────────┬─────────┘  │
           │            │             │
           │            ▼             │
           │   ┌──────────────────┐  │
           │   │  Capture Trace   │  │
           │   └────────┬─────────┘  │
           └────────────┼─────────────┘
                        │
                        ▼
           ┌────────────────────────────────────┐
           │   Run Scorers (Parallel)           │
           │   ┌───────────────────────────┐    │
           │   │  For each scorer:         │    │
           │   │  - inputs                 │    │
           │   │  - outputs                │    │
           │   │  - expectations           │    │
           │   │  - trace (optional)       │    │
           │   └──────────┬────────────────┘    │
           │              │                      │
           │              ▼                      │
           │   ┌─────────────────────────┐      │
           │   │  Optionally wrap in     │      │
           │   │  trace span             │      │
           │   └──────────┬──────────────┘      │
           │              │                      │
           │              ▼                      │
           │   ┌─────────────────────────┐      │
           │   │  Execute scorer         │      │
           │   │  Returns: Feedback      │      │
           │   │  - name                 │      │
           │   │  - value                │      │
           │   │  - rationale            │      │
           │   └──────────┬──────────────┘      │
           └──────────────┼─────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────┐
           │   Collect all Feedback objects   │
           └──────────────┬───────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────┐
           │   Log Assessments to Trace       │
           └──────────────┬───────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────┐
           │   Return EvalResult              │
           │   - feedbacks                    │
           │   - trace                        │
           │   - tags                         │
           └──────────────┬───────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────┐
           │   Aggregate Metrics Across All   │
           │   EvalItems                      │
           │   - mean, median, min, max, etc. │
           └──────────────┬───────────────────┘
                          │
                          ▼
           ┌──────────────────────────────────┐
           │   Log Aggregated Metrics to      │
           │   MLflow Run                     │
           └──────────────┬───────────────────┘
                          │
                          ▼
                    ┌──────────────┐
                    │ Final Result │
                    └──────────────┘
```

### Data Flow

**Input → Processing → Output**:

1. **Input**: `Dataset` (list of dicts with inputs, outputs, expectations)
2. **Convert**: Transform to `EvalItem` objects
3. **Predict** (optional): Run prediction function with tracing
4. **Score**: Execute all scorers in parallel
5. **Aggregate**: Compute aggregate metrics
6. **Output**: `EvaluationResult` with per-item and aggregated scores

### Key Files

- **Harness**: `/home/user/mlflow/mlflow/genai/evaluation/harness.py`
- **Entities**: `/home/user/mlflow/mlflow/genai/evaluation/entities.py`
- **Scorers**: `/home/user/mlflow/mlflow/genai/scorers/base.py`
- **Aggregation**: `/home/user/mlflow/mlflow/genai/scorers/aggregation.py`

---

## Trace-Based Judge Pipeline

**Purpose**: Enable LLM judges to analyze agent execution traces using tools for deep behavioral evaluation. The judge acts as an agent with access to tools for inspecting spans, searching content, and gathering performance metrics.

**Main Entry Point**: `InstructionsJudge` when `{{ trace }}` appears in instructions
**File**: `/home/user/mlflow/mlflow/genai/judges/instructions_judge/__init__.py`

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRACE-BASED JUDGE PIPELINE                              │
│                         (Agentic Evaluation)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  Evaluation Input  │
│  - trace           │
│  - instructions    │
│  - rating_fields   │
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Build System Prompt                                        │
│  (INSTRUCTIONS_JUDGE_TRACE_PROMPT_TEMPLATE)                 │
│                                                             │
│  "You are an expert judge...                                │
│   Instructions: {{ instructions }}                          │
│   Rating Fields: {{ rating_fields }}                        │
│                                                             │
│   5-Step Process:                                           │
│   1. Read instructions                                      │
│   2. Use tools to gather trace info                         │
│   3. Analyze information                                    │
│   4. Check if enough info (loop to 2 if not)                │
│   5. Provide JSON evaluation rating"                        │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Register Tools with Trace Context                          │
│  - GetTraceInfoTool                                         │
│  - GetRootSpanTool                                          │
│  - GetSpanTool(span_id)                                     │
│  - ListSpansTool(max_results, page_token)                   │
│  - SearchTraceRegexTool(pattern)                            │
│  - GetSpanPerformanceAndTimingReportTool                    │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│              AGENTIC LOOP (LiteLLM Adapter)                 │
│         Max Iterations: MLFLOW_JUDGE_MAX_ITERATIONS          │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
    ┌─────────────────┐
    │  Iteration Loop │
    └─────────┬───────┘
              │
    ┌─────────▼──────────────────────────────────────────────┐
    │  1. Send messages to LLM with tool definitions         │
    │     - System prompt with instructions                  │
    │     - Conversation history                             │
    │     - Available tools                                  │
    └─────────┬──────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────────┐
    │  2. LLM Response                                │
    └─────┬───────────────────────────────────────────┘
          │
          │  ┌──────────────────────────────────┐
          ├──┤  Has tool_calls?                 │
          │  └──────────────────────────────────┘
          │
          │ YES                              NO
          │                                  │
          ▼                                  ▼
┌─────────────────────────────┐   ┌────────────────────────────┐
│  3. Execute Tools           │   │  6. Extract Final Evaluation│
│                             │   │     - Parse JSON response   │
│  For each tool_call:        │   │     - Validate fields       │
│  ┌───────────────────────┐ │   │     - Return Feedback       │
│  │ Parse tool parameters │ │   └────────────┬───────────────┘
│  └──────────┬────────────┘ │                │
│             │               │                ▼
│             ▼               │   ┌────────────────────────────┐
│  ┌───────────────────────┐ │   │  Final Result              │
│  │ Invoke tool with      │ │   │  Feedback(                 │
│  │ trace context:        │ │   │    name=judge_name,        │
│  │                       │ │   │    value=rating,           │
│  │ Examples:             │ │   │    rationale=explanation   │
│  │ - list_spans()        │ │   │  )                         │
│  │ - get_span(id)        │ │   └────────────────────────────┘
│  │ - search_trace(regex) │ │
│  │ - get_trace_info()    │ │
│  └──────────┬────────────┘ │
│             │               │
│             ▼               │
│  ┌───────────────────────┐ │
│  │ Tool returns JSON:    │ │
│  │ - span data           │ │
│  │ - search results      │ │
│  │ - timing info         │ │
│  │ - error if failed     │ │
│  └──────────┬────────────┘ │
└─────────────┼───────────────┘
              │
              ▼
    ┌─────────────────────────────────────────┐
    │  4. Add Tool Responses to Conversation  │
    │     - Append tool results as messages   │
    │     - Format as JSON strings            │
    └─────────┬───────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────┐
    │  5. Context Window Management           │
    │     - Check total tokens                │
    │     - Prune old tool calls if exceeded  │
    │     - Keep recent conversation          │
    └─────────┬───────────────────────────────┘
              │
              │
              └──────────┐
                         │
                         ▼
                  ┌──────────────┐
                  │ Loop back to │
                  │ step 1       │
                  └──────────────┘
```

### Tool Examples and Data Flow

#### Example: ListSpansTool

**Tool Definition**:
```json
{
  "name": "list_spans",
  "description": "List all spans in the trace with pagination",
  "parameters": {
    "max_results": "int (default: 100)",
    "page_token": "string (optional)"
  }
}
```

**LLM Calls Tool**:
```json
{
  "tool_calls": [
    {
      "function": {
        "name": "list_spans",
        "arguments": "{\"max_results\": 50}"
      }
    }
  ]
}
```

**Tool Execution**:
```python
# File: /home/user/mlflow/mlflow/genai/judges/tools/list_spans.py
def invoke(trace, max_results=100, page_token=None):
    spans_data = []
    for span in trace.data.spans:
        spans_data.append({
            "span_id": span.request_id,
            "name": span.name,
            "span_type": span.span_type,
            "start_time": span.start_time_ns,
            "end_time": span.end_time_ns,
            "parent_id": span.parent_id,
            "status": span.status.status_code,
            "attribute_names": list(span.attributes.keys())
        })
    return {"spans": spans_data, "next_page_token": ...}
```

**Tool Response to LLM**:
```json
{
  "role": "tool",
  "content": "{\"spans\": [{\"span_id\": \"abc123\", \"name\": \"query_processor\", ...}], ...}"
}
```

#### Example: SearchTraceRegexTool

**Use Case**: Find specific patterns in trace content

**Tool Call**:
```json
{
  "function": {
    "name": "search_trace_regex",
    "arguments": "{\"pattern\": \"error|exception\"}"
  }
}
```

**Returns**: All spans where inputs/outputs/attributes match the pattern

### Prompt Flow in Trace-Based Judge

**Initial System Prompt** (from `constants.py`):
```
JUDGE_BASE_PROMPT +
"""Your job is to analyze a trace of the agent's execution...

The instructions refer to {{ trace }}. To read it, use the tools:
1. fetch trace metadata, timing, execution details
2. list all spans with inputs and outputs
3. search for specific text or patterns

Step-by-step:
1. Read instructions to understand criteria
2. Use tools to gather trace information
3. Analyze gathered information
4. Think - do you have enough info? If not, go to step 2
5. Provide evaluation rating as JSON

Evaluation Rating Fields:
{evaluation_rating_fields}

Instructions:
{instructions}
"""
```

**Example Conversation Flow**:

1. **LLM First Message**:
   ```
   "I need to understand the trace structure. Let me list all spans."
   [Calls list_spans tool]
   ```

2. **Tool Response**:
   ```json
   {"spans": [{"span_id": "1", "name": "root"}, {"span_id": "2", "name": "retriever"}, ...]}
   ```

3. **LLM Second Message**:
   ```
   "I see a retriever span. Let me examine it in detail."
   [Calls get_span(span_id="2")]
   ```

4. **Tool Response**:
   ```json
   {"span_id": "2", "inputs": {"query": "..."}, "outputs": {"chunks": [...]}}
   ```

5. **LLM Final Message**:
   ```json
   {
     "result": "yes",
     "rationale": "Let's think step by step. The retriever retrieved 5 relevant chunks..."
   }
   ```

### Key Files

- **Main Judge**: `/home/user/mlflow/mlflow/genai/judges/instructions_judge/__init__.py`
- **Prompts**: `/home/user/mlflow/mlflow/genai/judges/instructions_judge/constants.py`
- **Tool Registry**: `/home/user/mlflow/mlflow/genai/judges/tools/registry.py`
- **LiteLLM Adapter**: `/home/user/mlflow/mlflow/genai/judges/adapters/litellm_adapter.py`
- **Tool Utils**: `/home/user/mlflow/mlflow/genai/judges/utils/tool_calling_utils.py`

### Available Tools

1. **GetTraceInfoTool**: High-level metadata (state, timing, tags)
2. **GetRootSpanTool**: Root span inputs and outputs
3. **GetSpanTool**: Specific span by ID
4. **ListSpansTool**: All spans with pagination
5. **SearchTraceRegexTool**: Search content with regex
6. **GetSpanPerformanceAndTimingReportTool**: Performance metrics

---

