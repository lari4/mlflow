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

