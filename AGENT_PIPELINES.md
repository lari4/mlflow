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

## RAG Evaluation Pipelines

**Purpose**: Evaluate Retrieval-Augmented Generation systems across three dimensions: relevance of retrieved chunks, sufficiency of context, and groundedness of responses.

**Main File**: `/home/user/mlflow/mlflow/genai/scorers/builtin_scorers.py`

### 1. Retrieval Relevance Pipeline

**Purpose**: Evaluate if each retrieved document chunk is relevant to the user query. Returns precision score (fraction of relevant chunks).

**Prompts Used**: `RETRIEVAL_RELEVANCE_PROMPT` from `/home/user/mlflow/mlflow/genai/judges/prompts/retrieval_relevance.py`

#### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL RELEVANCE PIPELINE                             │
│                 (Evaluates per-chunk relevance → precision)                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Input: Trace    │
│  - request       │
│  - retriever     │
│    spans         │
└─────────┬────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  1. Extract Request from Root Span   │
│     - Get root span                  │
│     - Extract request field          │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  2. Find RETRIEVER Spans             │
│     - Filter by span_type=RETRIEVER  │
│     - Use last retriever if multiple │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  3. Extract Retrieved Chunks         │
│     - From retriever span outputs    │
│     - Each chunk is a document       │
└─────────┬────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  4. For Each Chunk: Judge Relevance (Parallel)          │
│                                                          │
│  ┌────────────────────────────────────────────┐         │
│  │  Prompt: RETRIEVAL_RELEVANCE_PROMPT        │         │
│  │                                            │         │
│  │  Consider this question and document.      │         │
│  │  Is the document (fully or partially)     │         │
│  │  relevant to the question?                 │         │
│  │                                            │         │
│  │  <question>{{request}}</question>          │         │
│  │  <document>{{chunk}}</document>            │         │
│  │                                            │         │
│  │  Return JSON:                              │         │
│  │  {                                         │         │
│  │    "rationale": "Let's think step by step...",      │         │
│  │    "result": "yes|no"                      │         │
│  │  }                                         │         │
│  └────────────┬───────────────────────────────┘         │
│               │                                          │
│               ▼                                          │
│  ┌────────────────────────────────┐                     │
│  │  LLM Judge Returns:            │                     │
│  │  Feedback(                     │                     │
│  │    name="retrieval_relevance"  │                     │
│  │    value="yes" or "no"         │                     │
│  │    rationale="..."             │                     │
│  │  )                             │                     │
│  └────────────┬───────────────────┘                     │
└───────────────┼──────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────┐
    │  5. Collect All YES/NO Results  │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌────────────────────────────────────────────┐
    │  6. Calculate Precision Score              │
    │     precision = count(YES) / total_chunks  │
    │                                            │
    │     Example:                               │
    │     - 5 chunks total                       │
    │     - 3 chunks relevant (YES)              │
    │     - 2 chunks not relevant (NO)           │
    │     → precision = 3/5 = 0.6                │
    └────────────┬───────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────────┐
    │  7. Return Final Feedback                  │
    │     Feedback(                              │
    │       name="retrieval_relevance_precision" │
    │       value=0.6                            │
    │       rationale="3 out of 5 chunks..."     │
    │     )                                      │
    └────────────────────────────────────────────┘
```

**Data Flow Summary**:
```
Trace → Extract Request → Find Retriever Spans → Extract Chunks →
For Each Chunk: Judge(request, chunk) → Aggregate YES/NO → Precision Score
```

---

### 2. Retrieval Sufficiency Pipeline

**Purpose**: Determine if the retrieved context provides sufficient information to answer the query given expected facts or response.

**Prompts Used**: `CONTEXT_SUFFICIENCY_PROMPT` from `/home/user/mlflow/mlflow/genai/judges/prompts/context_sufficiency.py`

#### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL SUFFICIENCY PIPELINE                            │
│              (Is retrieved context sufficient to support answer?)            │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Input: Trace    │
│  - request       │
│  - retriever     │
│  - expectations  │
│    or trace      │
│    assessments   │
└─────────┬────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  1. Extract Request from Root Span   │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  2. Extract Retrieved Context        │
│     - Find last RETRIEVER span       │
│     - Get all retrieved chunks       │
│     - Concatenate as context         │
└─────────┬────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  3. Get Expected Facts or Response                     │
│     Priority order:                                    │
│     a) expectations.expected_facts (from dataset)      │
│     b) expectations.expected_response (from dataset)   │
│     c) trace.data.assessments["expected_facts"]        │
│     d) trace.data.assessments["expected_response"]     │
└─────────┬──────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Build Sufficiency Judge Prompt                          │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │  CONTEXT_SUFFICIENCY_PROMPT_INSTRUCTIONS   │             │
│  │                                            │             │
│  │  Consider this claim and document.         │             │
│  │  Is the claim supported by the document?   │             │
│  │                                            │             │
│  │  <claim>                                   │             │
│  │    <question>{{request}}</question>        │             │
│  │    <answer>{{expected_facts}}</answer>     │             │
│  │  </claim>                                  │             │
│  │  <document>{{context}}</document>          │             │
│  │                                            │             │
│  │  + CONTEXT_SUFFICIENCY_PROMPT_OUTPUT       │             │
│  │  Return JSON: {                            │             │
│  │    "rationale": "Let's think step by step...",          │             │
│  │    "result": "yes|no"                      │             │
│  │  }                                         │             │
│  └────────────┬───────────────────────────────┘             │
└───────────────┼──────────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────┐
    │  5. Invoke Judge                │
    │     is_context_sufficient(      │
    │       request=request,           │
    │       context=context,           │
    │       expected_facts=facts,      │
    │       expected_response=response │
    │     )                            │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  6. Return Feedback             │
    │     Feedback(                   │
    │       name="context_sufficiency" │
    │       value="yes" or "no"       │
    │       rationale="..."           │
    │     )                           │
    └─────────────────────────────────┘
```

**Data Flow Summary**:
```
Trace → Extract (Request + Context + Expected) →
Judge(request, context, expected) → YES/NO Feedback
```

**Key Logic**: Checks if the retrieved documents contain enough information to support the expected answer/facts.

---

### 3. Retrieval Groundedness Pipeline

**Purpose**: Verify that the final response is grounded in (supported by) the retrieved context, detecting hallucinations.

**Prompts Used**: `GROUNDEDNESS_PROMPT` from `/home/user/mlflow/mlflow/genai/judges/prompts/groundedness.py`

#### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   RETRIEVAL GROUNDEDNESS PIPELINE                            │
│              (Is the response grounded in retrieved context?)                │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Input: Trace    │
│  - request       │
│  - response      │
│  - retriever     │
└─────────┬────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  1. Extract Request from Root Span   │
│     - Get root span inputs           │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  2. Extract Final Response           │
│     - Get root span outputs          │
│     - Or from outputs parameter      │
└─────────┬────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  3. Extract Retrieved Context        │
│     - Find all RETRIEVER spans       │
│     - Collect all chunks             │
│     - Concatenate as context         │
└─────────┬────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Build Groundedness Judge Prompt                         │
│                                                              │
│  ┌────────────────────────────────────────────┐             │
│  │  GROUNDEDNESS_PROMPT_INSTRUCTIONS          │             │
│  │                                            │             │
│  │  Consider this claim and document.         │             │
│  │  Is the claim supported by the document?   │             │
│  │                                            │             │
│  │  <claim>                                   │             │
│  │    <question>{{request}}</question>        │             │
│  │    <answer>{{response}}</answer>           │             │
│  │  </claim>                                  │             │
│  │  <document>{{context}}</document>          │             │
│  │                                            │             │
│  │  + GROUNDEDNESS_PROMPT_OUTPUT              │             │
│  │  Return JSON: {                            │             │
│  │    "rationale": "Let's think step by step...",          │             │
│  │    "result": "yes|no"                      │             │
│  │  }                                         │             │
│  └────────────┬───────────────────────────────┘             │
└───────────────┼──────────────────────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────────┐
    │  5. Invoke Judge                │
    │     is_grounded(                │
    │       request=request,           │
    │       response=response,         │
    │       context=context            │
    │     )                            │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  6. Return Feedback             │
    │     Feedback(                   │
    │       name="groundedness"       │
    │       value="yes" or "no"       │
    │       rationale="..."           │
    │     )                           │
    └─────────────────────────────────┘
```

**Data Flow Summary**:
```
Trace → Extract (Request + Response + Context) →
Judge(request, response, context) → YES/NO Feedback
```

**Key Logic**: Ensures the response doesn't contain claims that aren't supported by the retrieved documents (anti-hallucination check).

---

### RAG Pipeline Comparison

| Pipeline | Purpose | Inputs | Output | Prompt Used |
|----------|---------|--------|--------|-------------|
| **Retrieval Relevance** | Are retrieved chunks relevant? | Request + Each Chunk | Precision (0-1) | RETRIEVAL_RELEVANCE_PROMPT |
| **Retrieval Sufficiency** | Is context sufficient for answer? | Request + Context + Expected | YES/NO | CONTEXT_SUFFICIENCY_PROMPT |
| **Retrieval Groundedness** | Is response grounded in context? | Request + Response + Context | YES/NO | GROUNDEDNESS_PROMPT |

### Common Pattern

All three RAG pipelines follow this pattern:

1. **Extract from Trace**: Get request, response, and/or context from trace spans
2. **Build Prompt**: Use pre-defined judge prompts with extracted data
3. **Invoke Judge**: Call LLM judge with structured output (JSON with rationale + result)
4. **Return Feedback**: Convert judge response to Feedback object

### Key Files

- **Scorers**: `/home/user/mlflow/mlflow/genai/scorers/builtin_scorers.py`
- **Prompts**:
  - Retrieval Relevance: `/home/user/mlflow/mlflow/genai/judges/prompts/retrieval_relevance.py`
  - Context Sufficiency: `/home/user/mlflow/mlflow/genai/judges/prompts/context_sufficiency.py`
  - Groundedness: `/home/user/mlflow/mlflow/genai/judges/prompts/groundedness.py`

---

## Prompt Optimization Pipeline (GEPA)

**Purpose**: Automatically optimize prompts using Genetic-Pareto algorithm with LLM-based reflection. Iteratively improves prompts by evaluating performance, analyzing traces, and generating better candidates.

**Main Entry Point**: `mlflow.genai.optimize_prompts()`
**Files**:
- `/home/user/mlflow/mlflow/genai/optimize/optimize.py`
- `/home/user/mlflow/mlflow/genai/optimize/optimizers/gepa_optimizer.py`

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEPA PROMPT OPTIMIZATION PIPELINE                         │
│              (Iterative Evaluation → Reflection → Mutation)                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│  Input                           │
│  - predict_fn                    │
│  - train_data                    │
│  - initial_prompt_uris           │
│  - scorers + aggregation         │
│  - optimizer config              │
└─────────────┬────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  1. PREPARATION                                 │
│     ┌──────────────────────────────────┐        │
│     │ Load Prompts by URI              │        │
│     │ - Fetch PromptVersion objects    │        │
│     │ - Extract templates              │        │
│     └──────────────┬───────────────────┘        │
│                    │                             │
│                    ▼                             │
│     ┌──────────────────────────────────┐        │
│     │ Build metric_fn                  │        │
│     │ - Combine scorers                │        │
│     │ - Apply aggregation (mean, etc.) │        │
│     └──────────────┬───────────────────┘        │
│                    │                             │
│                    ▼                             │
│     ┌──────────────────────────────────┐        │
│     │ Create eval_fn                   │        │
│     │ (wraps predict_fn + metric_fn)   │        │
│     └──────────────────────────────────┘        │
└─────────────┬───────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────┐
│  2. ITERATIVE OPTIMIZATION LOOP                 │
│     (Max iterations: max_metric_calls)           │
└─────────────┬───────────────────────────────────┘
              │
    ┌─────────▼────────────────────────────────────────────┐
    │  For each candidate prompt:                          │
    └─────────┬────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────────┐
    │  EVALUATE Phase                                      │
    │  ┌────────────────────────────────────────────┐      │
    │  │  a) Patch PromptVersion.template           │      │
    │  │     - Temporarily replace with candidate   │      │
    │  └────────────┬───────────────────────────────┘      │
    │               │                                       │
    │               ▼                                       │
    │  ┌────────────────────────────────────────────┐      │
    │  │  b) For each training example:             │      │
    │  │     ┌──────────────────────────────┐       │      │
    │  │     │ Call predict_fn(inputs)      │       │      │
    │  │     │ - Uses current candidate     │       │      │
    │  │     │ - Captures trace             │       │      │
    │  │     └──────────┬───────────────────┘       │      │
    │  │                │                            │      │
    │  │                ▼                            │      │
    │  │     ┌──────────────────────────────┐       │      │
    │  │     │ Call metric_fn(inputs,       │       │      │
    │  │     │               outputs,       │       │      │
    │  │     │               expectations,  │       │      │
    │  │     │               trace)         │       │      │
    │  │     │ - Returns aggregated score   │       │      │
    │  │     └──────────┬───────────────────┘       │      │
    │  │                │                            │      │
    │  │                ▼                            │      │
    │  │     ┌──────────────────────────────┐       │      │
    │  │     │ Collect:                     │       │      │
    │  │     │ - inputs                     │       │      │
    │  │     │ - outputs                    │       │      │
    │  │     │ - score                      │       │      │
    │  │     │ - trace                      │       │      │
    │  │     │ - rationales (from scorers)  │       │      │
    │  │     │ - expectations               │       │      │
    │  │     └──────────┬───────────────────┘       │      │
    │  └────────────────┼────────────────────────────┘      │
    │                   │                                    │
    │                   ▼                                    │
    │  ┌────────────────────────────────────────────┐       │
    │  │  c) Return EvaluationBatch                 │       │
    │  │     - outputs: list of outputs             │       │
    │  │     - scores: list of scores               │       │
    │  │     - trajectories: list of                │       │
    │  │       EvaluationResultRecord (if traces)   │       │
    │  └────────────────────────────────────────────┘       │
    └──────────────────┬─────────────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────────────────┐
    │  REFLECT Phase (if capture_traces=True)                  │
    │  ┌────────────────────────────────────────────┐          │
    │  │  make_reflective_dataset():                │          │
    │  │                                            │          │
    │  │  For each trajectory & score:              │          │
    │  │  ┌──────────────────────────────────┐     │          │
    │  │  │ Extract trace spans:             │     │          │
    │  │  │ - span.name                      │     │          │
    │  │  │ - span.inputs                    │     │          │
    │  │  │ - span.outputs                   │     │          │
    │  │  └──────────┬───────────────────────┘     │          │
    │  │             │                              │          │
    │  │             ▼                              │          │
    │  │  ┌──────────────────────────────────┐     │          │
    │  │  │ Build reflection record:         │     │          │
    │  │  │ {                                │     │          │
    │  │  │   "component_name": "prompt_1",  │     │          │
    │  │  │   "current_text": "Answer...",   │     │          │
    │  │  │   "trace": [spans],              │     │          │
    │  │  │   "score": 0.75,                 │     │          │
    │  │  │   "inputs": {...},               │     │          │
    │  │  │   "outputs": {...},              │     │          │
    │  │  │   "expectations": {...},         │     │          │
    │  │  │   "rationales": ["..."],         │     │          │
    │  │  │   "index": 0                     │     │          │
    │  │  │ }                                │     │          │
    │  │  └──────────────────────────────────┘     │          │
    │  │                                            │          │
    │  │  Return: reflective_datasets[prompt_name] │          │
    │  └────────────┬───────────────────────────────┘          │
    └───────────────┼──────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────────────────────────────────┐
    │  MUTATE Phase (GEPA Library)                              │
    │  ┌─────────────────────────────────────────────┐          │
    │  │  GEPA uses reflection_lm to:                │          │
    │  │                                             │          │
    │  │  1. Analyze reflective dataset:             │          │
    │  │     - Current prompt text                   │          │
    │  │     - Execution traces                      │          │
    │  │     - Scores achieved                       │          │
    │  │     - Rationales for scores                 │          │
    │  │                                             │          │
    │  │  2. Generate improved prompt candidates:    │          │
    │  │     - Identify weaknesses                   │          │
    │  │     - Propose modifications                 │          │
    │  │     - Create new prompt text                │          │
    │  │                                             │          │
    │  │  Note: Actual reflection prompts are        │          │
    │  │  internal to GEPA library, not in MLflow    │          │
    │  └─────────────┬───────────────────────────────┘          │
    └────────────────┼──────────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────┐
    │  SELECT Phase (GEPA Library)                       │
    │  - Pareto-aware candidate selection                │
    │  - Balances multiple objectives                    │
    │  - Maintains diversity in population               │
    └────────────────┬───────────────────────────────────┘
                     │
                     │
                     └──────────┐
                                │
                                ▼
                         ┌──────────────┐
                         │ Loop back to │
                         │ EVALUATE     │
                         └──────────────┘
                                │
                                │ (until max_metric_calls reached)
                                │
                                ▼
              ┌─────────────────────────────────┐
              │  3. FINALIZATION                │
              │     ┌───────────────────────┐   │
              │     │ Select best candidate │   │
              │     │ - Highest score       │   │
              │     └───────────┬───────────┘   │
              │                 │               │
              │                 ▼               │
              │     ┌───────────────────────┐   │
              │     │ Register optimized    │   │
              │     │ prompts as new        │   │
              │     │ versions              │   │
              │     └───────────┬───────────┘   │
              │                 │               │
              │                 ▼               │
              │     ┌───────────────────────┐   │
              │     │ Return result:        │   │
              │     │ - optimized_prompts   │   │
              │     │ - initial_score       │   │
              │     │ - final_score         │   │
              │     └───────────────────────┘   │
              └─────────────────────────────────┘
```

### Data Flow Through GEPA Pipeline

**Phase 1: Evaluation**
```
Candidate Prompt → Patch Template → Run predict_fn() → Capture Trace →
Run scorers → Aggregate Score → EvaluationResultRecord
```

**Phase 2: Reflection**
```
EvaluationResultRecord[] → Extract Traces → Build Reflection Data →
{current_text, traces, scores, rationales} → Feed to GEPA
```

**Phase 3: Mutation**
```
Reflection Data → GEPA Reflection LLM → Analyze Performance →
Identify Improvements → Generate New Candidate Prompts
```

**Phase 4: Selection**
```
New Candidates → Evaluate All → Pareto Selection → Keep Best Diverse Set
```

### Example Reflection Dataset Record

```python
{
    "component_name": "qa_prompt",
    "current_text": "Answer the following question: {{question}}",
    "trace": [
        {
            "name": "predict",
            "inputs": {"question": "What is MLflow?"},
            "outputs": {"answer": "MLflow is a platform."}
        },
        {
            "name": "llm_call",
            "inputs": {"prompt": "Answer: What is MLflow?"},
            "outputs": {"text": "MLflow is a platform."}
        }
    ],
    "score": 0.6,
    "inputs": {"question": "What is MLflow?"},
    "outputs": {"answer": "MLflow is a platform."},
    "expectations": {"expected": "MLflow is an open-source platform..."},
    "rationales": ["Answer lacks detail about ML lifecycle management"],
    "index": 0
}
```

GEPA's reflection LLM analyzes this to understand that the prompt should explicitly ask for detailed explanations.

### Key Files

- **Main Optimize**: `/home/user/mlflow/mlflow/genai/optimize/optimize.py`
- **GEPA Optimizer**: `/home/user/mlflow/mlflow/genai/optimize/optimizers/gepa_optimizer.py`
- **Types**: `/home/user/mlflow/mlflow/genai/optimize/types.py`
- **GEPA Library**: External (pip install gepa)

### Configuration Parameters

- **reflection_model**: LLM for reflection (e.g., "openai:/gpt-4o")
- **max_metric_calls**: Evaluation budget (default: 100)
- **display_progress_bar**: Show progress (default: False)
- **use_mlflow**: Enable MLflow tracking (default: True)

---

## Built-in Judge Pipeline

**Purpose**: Simple, direct judge evaluation using pre-defined prompts without tool calling or trace analysis.

**Main File**: `/home/user/mlflow/mlflow/genai/judges/builtin.py`

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUILT-IN JUDGE PIPELINE                              │
│                    (Simple Prompt-based Evaluation)                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  Judge Input       │
│  - request         │
│  - response        │
│  - context/ground  │
│    truth (varies)  │
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  1. Select Pre-defined Prompt       │
│     Based on judge type:            │
│     - is_correct()                  │
│       → CORRECTNESS_PROMPT          │
│     - is_grounded()                 │
│       → GROUNDEDNESS_PROMPT         │
│     - is_safe()                     │
│       → SAFETY_PROMPT               │
│     - meets_guidelines()            │
│       → GUIDELINES_PROMPT           │
│     - etc.                          │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  2. Format Prompt with Variables    │
│     Using format_prompt():          │
│     - Replace {{input}}             │
│     - Replace {{output}}            │
│     - Replace {{context}}           │
│     - etc.                          │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  3. Invoke LLM Judge                │
│     invoke_judge_model(             │
│       messages=[{                   │
│         "role": "user",             │
│         "content": formatted_prompt │
│       }],                           │
│       response_format={             │
│         "type": "json_schema",      │
│         "schema": {                 │
│           "rationale": "string",    │
│           "result": "string"        │
│         }                           │
│       }                             │
│     )                               │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  4. Parse JSON Response             │
│     Extract:                        │
│     - rationale                     │
│     - result (yes/no or value)      │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  5. Create Feedback Object          │
│     Feedback(                       │
│       name=judge_name,              │
│       value=result,                 │
│       rationale=rationale           │
│     )                               │
└─────────┬───────────────────────────┘
          │
          ▼
    ┌─────────────┐
    │   Return    │
    └─────────────┘
```

### Example: Correctness Judge Flow

**Input**:
```python
is_correct(
    request="What is MLflow?",
    response="MLflow is an open-source platform.",
    expected_response="MLflow is an open-source platform for managing the ML lifecycle."
)
```

**Step 1: Select Prompt**
```python
CORRECTNESS_PROMPT from correctness.py
```

**Step 2: Format**
```
Consider the following question, claim and document...

<question>What is MLflow?</question>
<claim>MLflow is an open-source platform for managing the ML lifecycle.</claim>
<document>What is MLflow? - MLflow is an open-source platform.</document>

Please indicate whether each statement in the claim is supported...
{
  "rationale": "...",
  "result": "yes|no"
}
```

**Step 3: Invoke LLM**
```
LLM receives formatted prompt → Generates response
```

**Step 4: Parse Response**
```json
{
  "rationale": "Let's think step by step. The document states MLflow is an open-source platform, which matches the claim. However, the document doesn't mention 'managing the ML lifecycle', so the claim is not fully supported.",
  "result": "no"
}
```

**Step 5: Create Feedback**
```python
Feedback(
    name="correctness",
    value="no",
    rationale="Let's think step by step..."
)
```

### Available Built-in Judges

1. **is_correct()**: Correctness against ground truth
2. **is_grounded()**: Groundedness in context
3. **is_context_sufficient()**: Context sufficiency
4. **is_safe()**: Content safety
5. **meets_guidelines()**: Guideline compliance
6. **is_equivalent()**: Output equivalence

### Key Characteristics

- **Simple**: One LLM call per evaluation
- **Deterministic prompts**: Pre-defined, well-tested
- **Structured output**: JSON with rationale + result
- **Fast**: No tool calling or iteration
- **Reusable**: Same prompts across different judges

---

## Custom Prompt Judge Pipeline

**Purpose**: Allow users to create custom judges with their own prompts and categorical choices.

**Main File**: `/home/user/mlflow/mlflow/genai/judges/custom_prompt_judge.py`

### Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CUSTOM PROMPT JUDGE PIPELINE                           │
│                   (User-defined Categorical Evaluation)                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│  Judge Creation    │
│  - prompt_template │
│  - numeric_values  │
│    (optional)      │
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  1. Extract Choices from Template       │
│     Using regex: \[\[([\w ]+)\]\]      │
│                                         │
│     Example template:                   │
│     "[[formal]]: Very formal response"  │
│     "[[semi_formal]]: Somewhat formal"  │
│     "[[not_formal]]: Not formal"        │
│                                         │
│     Extracted: ["formal",               │
│                 "semi_formal",          │
│                 "not_formal"]           │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  2. Validate Choices                    │
│     If numeric_values provided:         │
│     - Check all choices have values     │
│     - Raise error if mismatch           │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  3. Create Judge Function               │
│     Returns callable that:              │
│     - Accepts keyword args              │
│     - Formats template                  │
│     - Adds structured output            │
│       instructions                      │
│     - Invokes LLM                       │
│     - Returns Feedback                  │
└─────────┬───────────────────────────────┘
          │
          ▼
    ┌─────────────┐
    │ Return Judge│
    └─────────────┘


RUNTIME EXECUTION:
─────────────────

┌────────────────────┐
│  Judge Invocation  │
│  judge(            │
│    request="...",  │
│    response="..."  │
│  )                 │
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  1. Format Template with Variables      │
│     Replace {{request}}, {{response}}   │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  2. Add Structured Output Instructions  │
│     Append JSON format requirements     │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  3. Invoke LLM with Prompt              │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  4. Parse Response                      │
│     Extract chosen category             │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  5. Map to Numeric Value (if provided)  │
│     e.g., "formal" → 1.0                │
└─────────┬───────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────┐
│  6. Create and Return Feedback          │
│     Feedback(                           │
│       name=judge_name,                  │
│       value=category or numeric,        │
│       rationale=explanation             │
│     )                                   │
└─────────────────────────────────────────┘
```

### Example Flow

**Creation**:
```python
judge = custom_prompt_judge(
    name="formality",
    prompt_template="""
<request>{{request}}</request>
<response>{{response}}</response>

Choose:
[[formal]]: Very formal
[[semi_formal]]: Somewhat formal
[[not_formal]]: Not formal
""",
    numeric_values={
        "formal": 1.0,
        "semi_formal": 0.5,
        "not_formal": 0.0
    }
)
```

**Invocation**:
```python
result = judge(
    request="Hi there!",
    response="Greetings, esteemed colleague."
)
```

**Runtime**:
1. Template formatted with request/response
2. LLM chooses category: "formal"
3. Mapped to numeric: 1.0
4. Returns: `Feedback(name="formality", value=1.0, rationale="...")`

---

## Summary

### Pipeline Comparison

| Pipeline | Complexity | Tool Use | Prompts | Output | Use Case |
|----------|------------|----------|---------|--------|----------|
| **Standard Evaluation** | Medium | No | Multiple scorers | Aggregate metrics | General model evaluation |
| **Trace-Based Judge** | High | Yes (6 tools) | Dynamic + Instructions | JSON rating | Complex agent behavior |
| **RAG Evaluation** | Medium | No | Pre-defined | YES/NO or precision | RAG system quality |
| **GEPA Optimization** | High | No | Reflection (external) | Optimized prompts | Prompt improvement |
| **Built-in Judge** | Low | No | Pre-defined | YES/NO | Quick assessments |
| **Custom Judge** | Low | No | User-defined | Categorical/numeric | Domain-specific |

### Common Data Flow Pattern

```
Input Data → Extract/Transform → Build Prompt → Invoke LLM →
Parse Response → Create Feedback → Return/Aggregate
```

### Key Concepts

1. **Feedback**: Universal output format with name, value, rationale
2. **Trace**: Execution record with spans for observability
3. **Structured Output**: JSON schemas for reliable parsing
4. **Chain-of-Thought**: "Let's think step by step" reasoning
5. **Parallel Execution**: ThreadPools for efficiency
6. **Tool Calling**: Agentic behavior for complex analysis

