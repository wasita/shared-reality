# LLM Stance Detection

Self-contained pipeline for predicting participant Likert responses using Gemini 3 Pro.

## Overview

This package submits batch prediction jobs to Google Cloud's Vertex AI to run Gemini 3 Pro. The model predicts probability distributions over Likert responses (1-5) for each of the 35 survey questions.

All data is loaded from `/paper/data/` for self-containment.

## Requirements

### Python packages
```bash
pip install pandas google-cloud-aiplatform google-cloud-storage python-dotenv
```

### Google Cloud credentials
Set these environment variables (or create a `.env` file):

```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_BUCKET=your-gcs-bucket
GOOGLE_CLOUD_LOCATION=us-central1  # optional, defaults to us-central1
```

You also need to authenticate:
```bash
gcloud auth application-default login
```

## Usage

### Running the no-chat experiment

The no-chat condition is information-matched to the Bayesian factor models—participants observe only their partner's single Likert response to one question.

```bash
# Dry run (creates sample prompt, doesn't submit)
python scripts/run_nochat_batch.py --dry-run

# Submit to Gemini
python scripts/run_nochat_batch.py
```

### Programmatic usage

```python
from paper.models.llm import (
    create_chat_prompt,
    create_prior_prompt,
    create_nochat_prompt,
    load_questions,
    load_unified_data,
    load_nochat_observations
)
from paper.models.llm.batch import create_batch_request, submit_batch

# Load data
questions_df = load_questions()
unified_df = load_unified_data()

# Create a no-chat batch request
nochat_obs = load_nochat_observations()
for _, row in nochat_obs.iterrows():
    prompt = create_nochat_prompt(
        observed_question=row['matched_question_text'],
        partner_response=int(row['partner_response']),
        questions_df=questions_df
    )
    req = create_batch_request(
        custom_id=f"nochat_{row['pid']}",
        prompt=prompt
    )
```

### Prompt templates

Three prompt templates are provided:

1. **`create_chat_prompt()`** - For predicting responses given conversation content
   - Marks the discussed question with `[DISCUSSED]`
   - Instructs model to use population priors + conversation evidence

2. **`create_prior_prompt()`** - Baseline with no information
   - Used to measure LLM's prior knowledge about population distributions
   - No observation provided

3. **`create_nochat_prompt()`** - Single observation (information-matched to Bayesian models)
   - Receives partner's Likert response to ONE question
   - Marks the observed question with `[OBSERVED]`
   - Predicts all 35 questions based on this single observation

## Model Configuration

```python
# config.py
MODEL_CONFIG = {
    "model": "gemini-3-pro-preview",
    "temperature": 1.0,  # Gemini 3's optimized default
    "max_tokens": 65536,
    "thinking_level": "high",  # Extended reasoning
}
```

## Output Format

The model returns JSON with probability distributions:

```json
{
  "predictions": {
    "0": {"1": 0.05, "2": 0.10, "3": 0.20, "4": 0.40, "5": 0.25},
    "1": {"1": 0.10, "2": 0.15, "3": 0.25, "4": 0.35, "5": 0.15},
    ...
  }
}
```

For chat prompts (with two participants):
```json
{
  "cat_predictions": { ... },
  "dog_predictions": { ... }
}
```

Question indices correspond to the shuffled order in the prompt (seeded for reproducibility).

## Data Files

All data is loaded from `/paper/data/`:

- `questions.csv` - 35 survey questions with domains
- `experiment_data.csv` - All participant data (chat + no-chat conditions)

## File Structure

```
llm/
├── __init__.py      # Package exports
├── config.py        # Paths and model configuration
├── prompts.py       # Prompt templates (chat, prior, nochat)
├── data.py          # Data loading utilities
├── batch.py         # Gemini batch submission
├── batch_requests/  # Saved batch request files
└── README.md        # This file
```

## Scripts

- `scripts/run_nochat_batch.py` - Submit no-chat batch predictions
