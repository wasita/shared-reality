#!/usr/bin/env python3
"""
Unified LLM batch prediction pipeline.

This module provides a single entry point for:
1. Creating batch requests from data
2. Submitting to Vertex AI
3. Checking job status
4. Downloading and parsing results

Usage:
    python -m models.llm.pipeline submit pshared --max-bins 13
    python -m models.llm.pipeline status
    python -m models.llm.pipeline download
"""

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
import vertexai
from vertexai.preview.batch_prediction import BatchPredictionJob

from .config import DATA_DIR, BATCH_DIR, MODEL_CONFIG
from .data import load_questions, load_unified_data
from .prompts import create_chat_prompt, create_pshared_prompt

# Constants
RESULTS_DIR = DATA_DIR / "llm_results"
RESULTS_DIR.mkdir(exist_ok=True)

GCS_PROJECT = "709275529646"
GCS_BUCKET = "hs-social-interaction-llm-batches"
GCS_PREFIX = "llm-stance-detection"

STATE_NAMES = {1: 'PENDING', 2: 'RUNNING', 3: 'SUCCEEDED', 4: 'FAILED', 5: 'CANCELLED'}


# =============================================================================
# Batch Request Creation
# =============================================================================

def create_batch_request(custom_id: str, prompt: str) -> dict:
    """Create a single batch request in Gemini format."""
    return {
        "custom_id": custom_id,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": MODEL_CONFIG["temperature"],
                "maxOutputTokens": MODEL_CONFIG["max_tokens"],
                "responseMimeType": "application/json",
                "thinkingConfig": {"thinkingLevel": MODEL_CONFIG.get("thinking_level", "high")}
            }
        }
    }


def build_conversation(messages_df: pd.DataFrame, up_to_time: pd.Timestamp) -> str:
    """Build conversation text up to a given timestamp."""
    subset = messages_df[messages_df['absolute_timestamp'] <= up_to_time].copy()
    subset = subset.sort_values('absolute_timestamp')
    lines = [f"{'Cat' if r['author'] == '🐱' else 'Dog'}: {r['message_string']}"
             for _, r in subset.iterrows()]
    return "\n".join(lines)


def create_chat_timecourse_batch(prompt_type: str = "pshared",
                                  bin_seconds: int = 15,
                                  max_bins: int = 13,
                                  sample_groups: list = None) -> list:
    """Create batch requests for chat timecourse analysis.

    Args:
        prompt_type: "chat" for full distributions, "pshared" for agreement probabilities
        bin_seconds: Size of each time bin in seconds
        max_bins: Maximum number of time bins (13 = ~3 min conversation)
        sample_groups: Optional list of group_ids to process

    Returns:
        List of batch request dicts
    """
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    messages['absolute_timestamp'] = pd.to_datetime(messages['absolute_timestamp'], format='mixed')
    messages['start_time'] = pd.to_datetime(messages['start_time'], format='mixed')
    questions = load_questions()

    if sample_groups:
        messages = messages[messages['group_id'].isin(sample_groups)]

    groups = messages.groupby('group_id').first().reset_index()
    print(f"Processing {len(groups)} groups with {max_bins} time bins each")

    batch_requests = []
    prompt_fn = create_pshared_prompt if prompt_type == "pshared" else create_chat_prompt

    for _, group_row in groups.iterrows():
        group_id = group_row['group_id']
        group_messages = messages[messages['group_id'] == group_id].copy()
        start_time = group_messages['start_time'].iloc[0]
        chat_topic = group_messages['matched_question'].iloc[0]

        for t in range(max_bins):
            bin_end = start_time + pd.Timedelta(seconds=(t + 1) * bin_seconds)
            conversation = build_conversation(group_messages, bin_end)
            prompt = prompt_fn(conversation=conversation, questions_df=questions, chat_topic=chat_topic)
            custom_id = f"{prompt_type}_{group_id}_t{t}"
            batch_requests.append(create_batch_request(custom_id, prompt))

    return batch_requests


# =============================================================================
# Batch Job Management
# =============================================================================

def get_batch_ids_file(experiment: str) -> Path:
    """Get path to batch IDs file for an experiment."""
    return BATCH_DIR / f"{experiment}_batch_ids.json"


def submit_batch(batch_requests: list, experiment: str, version: str = "v1") -> str:
    """Submit batch to Vertex AI.

    Args:
        batch_requests: List of batch request dicts
        experiment: Experiment name (e.g., "pshared", "chat")
        version: Version suffix for output directory

    Returns:
        Job resource name
    """
    vertexai.init(project=GCS_PROJECT, location="global")

    # Save batch file locally
    batch_file = BATCH_DIR / f"{experiment}_timecourse.jsonl"
    with open(batch_file, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')
    print(f"Created {batch_file} ({len(batch_requests)} requests)")

    # Upload to GCS
    gcs_input = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/input/{experiment}_timecourse.jsonl"
    gcs_output = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/output/{experiment}_{version}/"

    subprocess.run(["gsutil", "cp", str(batch_file), gcs_input], check=True)
    print(f"Uploaded to {gcs_input}")

    # Submit job
    job = BatchPredictionJob.submit(
        source_model=f"publishers/google/models/{MODEL_CONFIG['model']}",
        input_dataset=gcs_input,
        output_uri_prefix=gcs_output
    )

    print(f"Job: {job.resource_name}")
    print(f"State: {STATE_NAMES.get(job.state, job.state)}")

    # Save job info
    job_info = {
        f"{experiment}_timecourse": {
            "resource_name": job.resource_name,
            "input_uri": gcs_input,
            "output_uri": gcs_output
        }
    }
    with open(get_batch_ids_file(experiment), 'w') as f:
        json.dump(job_info, f, indent=2)

    return job.resource_name


def check_status(experiment: str) -> dict:
    """Check batch job status.

    Returns:
        Dict mapping job name to status string
    """
    batch_ids_file = get_batch_ids_file(experiment)
    if not batch_ids_file.exists():
        return {"error": "No batch job found"}

    vertexai.init(project=GCS_PROJECT, location="global")

    with open(batch_ids_file) as f:
        batch_ids = json.load(f)

    results = {}
    for name, info in batch_ids.items():
        job = BatchPredictionJob(info['resource_name'])
        results[name] = STATE_NAMES.get(job.state, str(job.state))

    return results


def download_results(experiment: str) -> Path:
    """Download results from completed batch job.

    Returns:
        Path to downloaded raw JSONL file
    """
    batch_ids_file = get_batch_ids_file(experiment)
    if not batch_ids_file.exists():
        raise FileNotFoundError(f"No batch job found for {experiment}")

    with open(batch_ids_file) as f:
        batch_ids = json.load(f)

    info = batch_ids.get(f"{experiment}_timecourse")
    if not info:
        raise KeyError(f"No {experiment}_timecourse job found")

    output_uri = info['output_uri']

    # Find predictions file
    result = subprocess.run(
        ["gsutil", "ls", f"{output_uri}**predictions.jsonl"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise FileNotFoundError("No predictions file found. Job may still be running.")

    predictions_uri = result.stdout.strip().split('\n')[0]
    print(f"Downloading {predictions_uri}...")

    raw_file = RESULTS_DIR / f"{experiment}_raw.jsonl"
    subprocess.run(["gsutil", "cp", predictions_uri, str(raw_file)], check=True)

    return raw_file


# =============================================================================
# Result Parsing
# =============================================================================

def parse_raw_results(raw_file: Path) -> list:
    """Parse raw JSONL results from Vertex AI batch.

    Returns:
        List of dicts with custom_id and predictions
    """
    results = []
    errors = 0

    with open(raw_file) as f:
        for line in f:
            try:
                d = json.loads(line)
                custom_id = d['custom_id']
                response = d.get('response', {})
                candidates = response.get('candidates', [])

                if not candidates:
                    errors += 1
                    continue

                content = candidates[0].get('content', {})
                parts = content.get('parts', [])

                pred_json = None
                for part in parts:
                    if 'text' in part:
                        try:
                            pred_json = json.loads(part['text'])
                            break
                        except json.JSONDecodeError:
                            continue

                if pred_json is None:
                    errors += 1
                    continue

                results.append({'custom_id': custom_id, 'predictions': pred_json})

            except Exception:
                errors += 1

    print(f"Parsed {len(results)} results, {errors} errors")
    return results


def compute_pshared_metrics(results: list) -> pd.DataFrame:
    """Compute P(shared) metrics from parsed results.

    Returns:
        DataFrame with predicted vs actual agreement by group/time/question
    """
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    unified = load_unified_data()
    questions = load_questions()

    # Load responses to get matchedTolerance for stance recoding
    responses_df = pd.read_csv(DATA_DIR / "responses.csv", low_memory=False)
    responses_chat = responses_df[responses_df['experiment'] == 'chat'].copy()

    # Build stance lookup: groupId -> stance (based on matchedTolerance <= 1)
    # 'high' = high match (shared stance), 'low' = low match (opposing stance)
    # This recodes all conditions (including random) based on actual matched tolerance
    stance_lookup = {}
    for group_id in responses_chat['groupId'].unique():
        group_resp = responses_chat[responses_chat['groupId'] == group_id]
        if len(group_resp) > 0:
            matched_tol = group_resp['matchedTolerance'].iloc[0]
            stance_lookup[group_id] = 'high' if matched_tol <= 1 else 'low'

    unified_chat = unified[unified['experiment'] == 'chat'].copy()

    # Build group metadata
    group_meta = {}
    for group_id in messages['group_id'].unique():
        group_msgs = messages[messages['group_id'] == group_id]
        cat_msgs = group_msgs[group_msgs['author'] == '🐱']
        dog_msgs = group_msgs[group_msgs['author'] == '🐶']
        if len(cat_msgs) == 0 or len(dog_msgs) == 0:
            continue
        # Use recoded stance based on matchedTolerance, not raw match_type
        stance = stance_lookup.get(group_id, 'unknown')
        group_meta[group_id] = {
            'cat_pid': cat_msgs['prolific_id'].iloc[0],
            'dog_pid': dog_msgs['prolific_id'].iloc[0],
            'match_type': stance,
        }

    # Build response lookup
    responses = {}
    for _, row in unified_chat.iterrows():
        pid = row['pid']
        if pid not in responses:
            responses[pid] = {}
        responses[pid][int(row['question'])] = int(row['own_response'])

    # Build matched question info
    matched_info = {}
    for _, row in unified_chat[unified_chat['is_matched'] == True].iterrows():
        mq = int(row['matched_question'])
        matched_domain = row.get('matched_domain')
        if pd.isna(matched_domain) and mq - 1 < len(questions):
            matched_domain = questions.iloc[mq - 1]['domain']
        matched_info[row['pid']] = {'matched_question': mq, 'matched_domain': matched_domain}

    records = []
    for result in results:
        custom_id = result['custom_id']
        predictions = result['predictions']

        parts = custom_id.split('_')
        if len(parts) < 3 or parts[0] != 'pshared':
            continue

        group_id = parts[1]
        time_bin = int(parts[2].replace('t', ''))

        if group_id not in group_meta:
            continue

        meta = group_meta[group_id]
        cat_pid, dog_pid = meta['cat_pid'], meta['dog_pid']

        if cat_pid not in matched_info:
            continue

        matched_q = matched_info[cat_pid]['matched_question']
        matched_domain = matched_info[cat_pid]['matched_domain']

        agreement_probs = predictions.get('agreement_probabilities', predictions)

        for q_str, pred_agreement in agreement_probs.items():
            q_idx = int(q_str)
            q_num = q_idx + 1

            if cat_pid not in responses or dog_pid not in responses:
                continue
            if q_num not in responses[cat_pid] or q_num not in responses[dog_pid]:
                continue

            cat_resp = responses[cat_pid][q_num]
            dog_resp = responses[dog_pid][q_num]
            actual_agreement = abs(cat_resp - dog_resp) <= 2  # τ=2.0 from fitted_params

            q_domain = questions.iloc[q_idx]['domain'] if q_idx < len(questions) else None

            if q_num == matched_q:
                category = 'matched'
            elif q_domain and matched_domain and q_domain.lower() == matched_domain.lower():
                category = 'same_domain'
            else:
                category = 'different_domain'

            records.append({
                'group_id': group_id,
                'time_bin': time_bin,
                'bin_seconds': time_bin * 15,
                'question': q_num,
                'match_type': meta['match_type'],
                'question_category': category,
                'question_domain': q_domain,
                'predicted_agreement': float(pred_agreement),
                'actual_agreement': actual_agreement,
            })

    return pd.DataFrame(records)


# =============================================================================
# CLI
# =============================================================================

def cmd_submit(args):
    """Submit batch job."""
    if args.sample:
        messages = pd.read_csv(DATA_DIR / "messages.csv")
        sample_groups = list(messages['group_id'].unique()[:args.sample])
        print(f"Using sample of {len(sample_groups)} groups")
    else:
        sample_groups = None

    batch_requests = create_chat_timecourse_batch(
        prompt_type=args.type,
        max_bins=args.max_bins,
        sample_groups=sample_groups
    )
    print(f"Created {len(batch_requests)} batch requests")

    if args.dry_run:
        # Just save locally
        batch_file = BATCH_DIR / f"{args.type}_timecourse.jsonl"
        with open(batch_file, 'w') as f:
            for req in batch_requests:
                f.write(json.dumps(req) + '\n')
        print(f"[DRY RUN] Saved to {batch_file}")
        return

    submit_batch(batch_requests, args.type, args.version)


def cmd_status(args):
    """Check job status."""
    status = check_status(args.type)
    for name, state in status.items():
        print(f"{name}: {state}")


def cmd_download(args):
    """Download and parse results."""
    raw_file = download_results(args.type)
    print(f"Saved raw results to {raw_file}")

    results = parse_raw_results(raw_file)

    if args.type == "pshared":
        df = compute_pshared_metrics(results)
        output_file = RESULTS_DIR / "pshared_timecourse.csv"
    else:
        raise NotImplementedError("Use 'pshared' type. Chat parsing not implemented in unified pipeline.")

    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total predictions: {len(df)}")
    print(f"Groups: {df['group_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(description="LLM batch prediction pipeline")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Submit
    sub = subparsers.add_parser('submit', help='Submit batch job')
    sub.add_argument('type', choices=['pshared', 'chat'], help='Prompt type')
    sub.add_argument('--max-bins', type=int, default=13, help='Time bins (default: 13)')
    sub.add_argument('--sample', type=int, help='Only process first N groups')
    sub.add_argument('--version', default='v1', help='Output version suffix')
    sub.add_argument('--dry-run', action='store_true', help='Create batch file only')

    # Status
    sub = subparsers.add_parser('status', help='Check job status')
    sub.add_argument('type', choices=['pshared', 'chat'], help='Experiment type')

    # Download
    sub = subparsers.add_parser('download', help='Download and parse results')
    sub.add_argument('type', choices=['pshared', 'chat'], help='Experiment type')

    args = parser.parse_args()

    if args.command == 'submit':
        cmd_submit(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'download':
        cmd_download(args)


if __name__ == "__main__":
    main()
