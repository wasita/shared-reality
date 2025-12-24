"""
Parse LLM batch results and compute accuracy metrics.
"""

import json
import pandas as pd
from pathlib import Path
from .config import PAPER_DIR, DATA_DIR
from .data import load_unified_data, load_questions

RESULTS_DIR = PAPER_DIR / "data" / "llm_results"


def parse_raw_results(raw_file: Path) -> list:
    """Parse raw JSONL results from Vertex AI batch.

    Args:
        raw_file: Path to raw .jsonl file

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

                # Extract predictions from response
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

                # Handle different output formats
                predictions = pred_json.get('predictions', pred_json)

                results.append({
                    'custom_id': custom_id,
                    'predictions': predictions
                })

            except Exception as e:
                errors += 1
                continue

    print(f"Parsed {len(results)} results, {errors} errors")
    return results


def compute_nochat_accuracy(results: list) -> pd.DataFrame:
    """Compute accuracy for nochat results.

    Args:
        results: Parsed results from parse_raw_results

    Returns:
        DataFrame with accuracy by question category
    """
    # Load ground truth
    unified = load_unified_data()
    questions = load_questions()

    # Filter to no-chat only
    nochat = unified[unified['experiment'] == 'no-chat'].copy()

    # Build pid -> question -> response map
    ground_truth = {}
    for _, row in nochat.iterrows():
        pid = row['pid']
        if pid not in ground_truth:
            ground_truth[pid] = {}
        ground_truth[pid][int(row['question'])] = int(row['own_response'])

    # Get matched question info per participant
    matched_info = {}
    matched_rows = nochat[nochat['is_matched'] == True]
    for _, row in matched_rows.iterrows():
        matched_info[row['pid']] = {
            'matched_question': int(row['matched_question']),
            'matched_domain': row.get('matched_domain', questions.iloc[int(row['matched_question'])]['domain'])
        }

    # Compute accuracy for each prediction
    records = []

    for result in results:
        custom_id = result['custom_id']
        predictions = result['predictions']

        # Parse pid from custom_id (format: nochat_{pid}_high)
        parts = custom_id.split('_')
        if len(parts) < 2:
            continue
        pid = parts[1]

        if pid not in ground_truth:
            continue

        if pid not in matched_info:
            continue

        matched_q = matched_info[pid]['matched_question']
        matched_domain = matched_info[pid]['matched_domain']

        # For each question, compute probability assigned to true response
        for q_str, probs in predictions.items():
            q_idx = int(q_str)

            if q_idx not in ground_truth[pid]:
                continue

            true_response = ground_truth[pid][q_idx]
            prob_correct = float(probs.get(str(true_response), 0))

            # Determine question category
            q_domain = questions.iloc[q_idx]['domain'] if q_idx < len(questions) else None

            if q_idx == matched_q:
                category = 'matched'
            elif q_domain and matched_domain and q_domain.lower() == matched_domain.lower():
                category = 'same_domain'
            else:
                category = 'different_domain'

            records.append({
                'pid': pid,
                'question': q_idx,
                'true_response': true_response,
                'prob_correct': prob_correct,
                'category': category,
                'question_domain': q_domain,
                'matched_domain': matched_domain
            })

    return pd.DataFrame(records)


def summarize_accuracy(df: pd.DataFrame) -> dict:
    """Compute summary statistics from accuracy DataFrame.

    Returns:
        Dict with overall and by-category accuracies
    """
    summary = {
        'overall': df['prob_correct'].mean(),
        'n_predictions': len(df),
        'n_participants': df['pid'].nunique(),
        'by_category': df.groupby('category')['prob_correct'].agg(['mean', 'std', 'count']).to_dict('index')
    }
    return summary


def compute_chat_timecourse_accuracy(results: list) -> pd.DataFrame:
    """Compute accuracy for chat timecourse results.

    Args:
        results: Parsed results from parse_raw_results

    Returns:
        DataFrame with accuracy by time_bin, match_type, question_category
    """
    # Load raw chat messages and derive group metadata
    messages = pd.read_csv(DATA_DIR / "chat_messages.csv")
    questions = load_questions()

    # Build group_id -> metadata map from chat messages
    group_meta = {}
    for group_id in messages['group_id'].unique():
        group_msgs = messages[messages['group_id'] == group_id]
        cat_msgs = group_msgs[group_msgs['author'] == '🐱']
        dog_msgs = group_msgs[group_msgs['author'] == '🐶']

        if len(cat_msgs) == 0 or len(dog_msgs) == 0:
            continue

        group_meta[group_id] = {
            'cat_pid': cat_msgs['prolific_id'].iloc[0],
            'dog_pid': dog_msgs['prolific_id'].iloc[0],
            'match_type': group_msgs['match_type'].iloc[0],
        }

    # Load ground truth responses
    unified = load_unified_data()
    unified_chat = unified[unified['experiment'] == 'chat'].copy()

    # Build pid -> question -> response map
    ground_truth = {}
    for _, row in unified_chat.iterrows():
        pid = row['pid']
        if pid not in ground_truth:
            ground_truth[pid] = {}
        ground_truth[pid][int(row['question'])] = int(row['own_response'])

    # Get matched question info per participant
    matched_info = {}
    matched_rows = unified_chat[unified_chat['is_matched'] == True]
    for _, row in matched_rows.iterrows():
        mq = int(row['matched_question'])
        matched_domain = row.get('matched_domain')
        if pd.isna(matched_domain) and mq < len(questions):
            matched_domain = questions.iloc[mq]['domain']
        matched_info[row['pid']] = {
            'matched_question': mq,
            'matched_domain': matched_domain
        }

    records = []

    for result in results:
        custom_id = result['custom_id']
        predictions = result['predictions']

        # Parse custom_id: chat_{group_id}_t{time_bin}
        parts = custom_id.split('_')
        if len(parts) < 3 or not parts[0] == 'chat':
            continue

        group_id = parts[1]
        time_bin = int(parts[2].replace('t', ''))

        if group_id not in group_meta:
            continue

        meta = group_meta[group_id]
        match_type = meta['match_type']

        # Process predictions for both cat and dog
        for author, pid_key in [('cat', 'cat_pid'), ('dog', 'dog_pid')]:
            pid = meta[pid_key]
            pred_key = f'{author}_predictions'

            if pred_key not in predictions:
                continue

            if pid not in ground_truth or pid not in matched_info:
                continue

            matched_q = matched_info[pid]['matched_question']
            matched_domain = matched_info[pid]['matched_domain']

            author_preds = predictions[pred_key]

            for q_str, probs in author_preds.items():
                q_idx = int(q_str)

                if q_idx not in ground_truth[pid]:
                    continue

                true_response = ground_truth[pid][q_idx]
                prob_correct = float(probs.get(str(true_response), 0))

                # Determine question category
                q_domain = questions.iloc[q_idx]['domain'] if q_idx < len(questions) else None

                if q_idx == matched_q:
                    category = 'matched'
                elif q_domain and matched_domain and q_domain.lower() == matched_domain.lower():
                    category = 'same_domain'
                else:
                    category = 'different_domain'

                # Compute if prediction is "correct" (argmax matches)
                probs_list = [float(probs.get(str(i), 0)) for i in range(1, 6)]
                predicted = probs_list.index(max(probs_list)) + 1
                correct = predicted == true_response

                records.append({
                    'group_id': group_id,
                    'time_bin': time_bin,
                    'bin_seconds': time_bin * 15,
                    'author': author,
                    'pid': pid,
                    'question': q_idx,
                    'match_type': match_type,
                    'question_category': category,
                    'true_response': true_response,
                    'predicted': predicted,
                    'correct': correct,
                    'prob_correct': prob_correct,
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Parse nochat results
    nochat_file = RESULTS_DIR / "nochat_raw.jsonl"

    if nochat_file.exists():
        print("Parsing nochat results...")
        results = parse_raw_results(nochat_file)

        print("\nComputing accuracy...")
        accuracy_df = compute_nochat_accuracy(results)

        print("\n=== NOCHAT LLM RESULTS ===")
        summary = summarize_accuracy(accuracy_df)
        print(f"Overall accuracy: {summary['overall']:.1%}")
        print(f"N predictions: {summary['n_predictions']}")
        print(f"N participants: {summary['n_participants']}")

        print("\nBy category:")
        for cat, stats in summary['by_category'].items():
            print(f"  {cat}: {stats['mean']:.1%} (n={stats['count']:.0f})")

        # Save detailed results
        output_file = RESULTS_DIR / "nochat_accuracy.csv"
        accuracy_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
    else:
        print(f"No results file found at {nochat_file}")
