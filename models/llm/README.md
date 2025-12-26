# LLM Stance Detection

Pipeline for predicting participant agreement using Gemini 3 Pro via Vertex AI batch predictions.

## Quick Start

```bash
# Submit P(shared) batch (predicts agreement probabilities over time)
python -m models.llm.pipeline submit pshared --max-bins 13

# Check job status
python -m models.llm.pipeline status pshared

# Download and parse results
python -m models.llm.pipeline download pshared
```

## Overview

This package submits batch prediction jobs to Google Cloud's Vertex AI. Two main prompt types are supported:

1. **P(shared)** - Directly predicts agreement probability for each question
2. **Chat** - Predicts full probability distributions for each participant

## Requirements

```bash
pip install pandas google-cloud-aiplatform google-cloud-storage vertexai
gcloud auth application-default login
```

## Pipeline Commands

### Submit a batch job

```bash
# P(shared) predictions (recommended)
python -m models.llm.pipeline submit pshared --max-bins 13

# Full distribution predictions
python -m models.llm.pipeline submit chat --max-bins 13

# Options
--max-bins N    # Number of 15-second time bins (default: 13 = ~3 min)
--sample N      # Only process first N groups (for testing)
--version V     # Output version suffix (default: v1)
--dry-run       # Create batch file without submitting
```

### Check job status

```bash
python -m models.llm.pipeline status pshared
python -m models.llm.pipeline status chat
```

### Download and parse results

```bash
python -m models.llm.pipeline download pshared
python -m models.llm.pipeline download chat
```

Results are saved to `data/llm_results/`:
- `pshared_timecourse.csv` - Agreement predictions by group/time/question

## Prompt Templates

### P(shared) Prompt (`create_pshared_prompt`)

Directly asks for agreement probability:

```
Two people "agree" if their Likert responses are within 1 point of each other.
Questions are organized by domain. Questions in the SAME DOMAIN should show
correlated beliefs.
```

Output format:
```json
{
  "agreement_probabilities": {
    "0": 0.85,
    "1": 0.42,
    ...
  }
}
```

### Chat Prompt (`create_chat_prompt`)

Predicts full probability distributions for each participant:

```json
{
  "cat_predictions": {"0": {"1": 0.1, "2": 0.2, ...}, ...},
  "dog_predictions": {"0": {"1": 0.1, "2": 0.2, ...}, ...}
}
```

### Other Prompts

- **`create_prior_prompt()`** - Baseline with no information
- **`create_nochat_prompt()`** - Single observation (for no-chat condition)

## File Structure

```
llm/
├── __init__.py      # Package exports
├── pipeline.py      # Unified submit/status/download pipeline
├── prompts.py       # Prompt templates
├── data.py          # Data loading utilities
├── config.py        # Paths and model configuration
├── batch_requests/  # Saved batch request files
│   └── *_batch_ids.json  # Job tracking info
└── README.md
```

## Output Data

### pshared_timecourse.csv

| Column | Description |
|--------|-------------|
| group_id | Conversation group identifier |
| time_bin | 15-second time bin (0-12) |
| question | Question number (1-35) |
| match_type | high/low (based on initial response similarity) |
| question_category | matched/same_domain/different_domain |
| predicted_agreement | LLM's predicted P(agree) |
| actual_agreement | True if responses within 1 point |

## Programmatic Usage

```python
from models.llm import (
    create_pshared_prompt,
    load_questions,
    submit_batch,
    check_status,
    download_results,
    compute_pshared_metrics,
)

# Create custom batch
questions = load_questions()
prompt = create_pshared_prompt(
    conversation="Cat: I agree!\nDog: Me too!",
    questions_df=questions,
    chat_topic="Do you enjoy reading?"
)
```

## Model Configuration

```python
MODEL_CONFIG = {
    "model": "gemini-3-pro-preview",
    "temperature": 1.0,
    "max_tokens": 65536,
    "thinking_level": "high",
}
```
