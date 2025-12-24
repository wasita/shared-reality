"""
Data loading utilities for LLM stance detection.

All data is loaded from /paper/data/ for self-containment.
"""

import pandas as pd
from .config import DATA_DIR


def load_questions() -> pd.DataFrame:
    """Load all 35 survey questions.

    Returns:
        DataFrame with columns [questionText, domain]
    """
    df = pd.read_csv(DATA_DIR / "questions.csv")
    # Rename columns for compatibility
    df = df.rename(columns={'text': 'questionText'})
    return df


def load_unified_data() -> pd.DataFrame:
    """Load unified input data for all participants (chat and no-chat).

    Returns:
        DataFrame with columns including experiment, pid, question, own_response,
        partner_response, match_type, matched_question, is_matched, question_type
    """
    df = pd.read_csv(DATA_DIR / "experiment_data.csv", low_memory=False)
    # Add column aliases for backward compatibility
    df['own_response'] = df['preChatResponse']
    df['question_domain'] = df['preChatDomain']
    df['matched_question'] = df['matchedIdx']
    return df


def load_ground_truth() -> dict:
    """Load pre-chat ground truth responses for all participants.

    Returns:
        Nested dict: {pid -> {question_idx -> likert_response}}
    """
    df = load_unified_data()
    questions = load_questions()

    ground_truth = {}
    for _, row in df.iterrows():
        pid = row['pid']
        if pid not in ground_truth:
            ground_truth[pid] = {}
        # Use question index as key
        ground_truth[pid][row['question']] = row['own_response']

    return ground_truth


def load_nochat_observations() -> pd.DataFrame:
    """Load no-chat participant observations (matched question + partner response).

    Returns:
        DataFrame with one row per participant showing the observed question and response
    """
    df = load_unified_data()
    questions = load_questions()

    # Filter to no-chat, matched rows only
    nochat_matched = df[(df['experiment'] == 'no-chat') & (df['is_matched'] == True)].copy()

    # Get question text for each matched question
    q_text = questions.set_index('num')['questionText'].to_dict()
    nochat_matched['matched_question_text'] = nochat_matched['matched_question'].map(q_text)

    return nochat_matched[['pid', 'match_type', 'matched_question', 'matched_question_text', 'partner_response']]


def load_dialogue_data(bin_size: str = "15s") -> pd.DataFrame:
    """Load binned dialogue data for timecourse analysis.

    Args:
        bin_size: '15s' only (other sizes not in paper/data)

    Returns:
        DataFrame with conversation text at each time bin
    """
    if bin_size == "15s":
        path = DATA_DIR / "15s-binned-complete-timecourse.csv"
    else:
        raise ValueError(f"Unsupported bin_size: {bin_size}. Only '15s' available in paper/data/")

    return pd.read_csv(path)


def normalize_domain(domain: str) -> str:
    """Normalize domain names for comparison."""
    if pd.isna(domain):
        return ''
    domain_map = {'moral': 'morality', 'morality': 'morality'}
    d = str(domain).lower().strip()
    return domain_map.get(d, d)


def add_question_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add question category column (matched/same_domain/different_domain)."""
    df = df.copy()
    df['question_domain_norm'] = df['question_domain'].apply(normalize_domain)

    if 'matched_domain' in df.columns:
        df['matched_domain_norm'] = df['matched_domain'].apply(normalize_domain)

        def categorize(row):
            if row['is_matched']:
                return 'matched'
            elif row['question_domain_norm'] == row['matched_domain_norm']:
                return 'same_domain'
            return 'different_domain'

        df['question_category'] = df.apply(categorize, axis=1)

    return df
