#!/usr/bin/env python3
"""
Preprocess raw data files to create responses.csv

This script implements the gold-standard preprocessing pipeline, creating a clean,
reproducible path from raw Firebase exports to the final responses.csv.

RAW DATA SOURCES
================
From shared-reality-dev repo (exported from Firebase via pull_data_from_firebase.ipynb):

Chat conditions:
  - chat/high-match-pre-post-responses.csv
  - chat/low-match-pre-post-responses.csv
  - chat/random-match-pre-post-responses.csv

No-chat condition:
  - no-chat/v1.0.2-no-chat-all-match-pre-post-responses.csv

Question metadata:
  - question-table.csv

SR-G scores:
  - srgi-composite-scores.csv (Interaction-specific Generalized Shared Reality; Rossignac-Milon et al., 2021)

MANUAL DATA FIXES
=================
The following fix was applied in pull_data_from_firebase.ipynb and is reflected
in the individual chat files (but NOT in the partner-joined files):

  Participant 65a3c0ee60b1e46f4e1d5f75:
    - matchedIdx: 15 -> 11
    - matchedQuestion: "Are you married or dating someone?" -> "Are you a parent or caregiver?"
    - Reason: Firebase had incorrect matchedIdx for this participant

DUPLICATE PARTICIPANT HANDLING
==============================
22 participants appear in multiple conditions. They are handled as follows:

1. PARTICIPANTS IN BOTH CHAT AND NO-CHAT (17 total):
   Rule: Keep chat entry, remove no-chat entry

   PIDs: 587418895c17910001ea4e75, 594425f07ccfd00001d3f462, 5acbd1ecf69e940001d9cd4d,
         5b0c65c11e55760001b96e90, 5cfb5233df7d70001619ca90, 5d1c0e818d27eb0016164015,
         5d42486206811e001ada5b4b, 5d76d2f7daf4bf00164d585b, 5d8a29c082fec30001d9c24a,
         5e1f7b2a4c9b832b34e7c9d3, 5e27727328b7b698b037ee6c, 628509147e9e933a0c068b9b,
         6306d39340264f76753706fd, 63474e67a5fd298c6103c409, 63d2eec88a81e2462be70762,
         6438a923683de8fa6662b555, 65603afb8e9cfc182d7b55bc

2. PARTICIPANTS IN MULTIPLE NO-CHAT CONDITIONS (5 total):
   Rule: Keep first participation (determined by preprocess_for_simulations.ipynb)

   5c395df5f5ebd50001850900: in low & high -> keep HIGH (remove low)
   6742d74e9f7b3b029a3791a6: in low & high -> keep HIGH (remove low)
   63ee285c49c51e4f2826a68e: in low & high -> keep LOW (remove high)
   6500a84f0a4f87687fa51a20: in low & high -> keep LOW (remove high)
   672c4e78eeaae1dfe925f5e2: in low & random -> keep LOW (remove random)

TOLERANCE COMPUTATION
=====================
matchedTolerance = |ownResponse - partnerResponse| for the focal (discussed) question

- Chat: Precomputed in raw files from partner's actual pre-chat response
- No-chat: Computed from ownResponse and observedResponse columns

OUTPUT
======
../responses.csv - Combined dataset ready for behavioral analyses

Expected result: 8/21 significant cells for opposing stance (FDR < 0.05)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
RAW_DIR = Path(__file__).parent
DATA_DIR = RAW_DIR.parent
OUTPUT_PATH = DATA_DIR / "responses.csv"

# Domain mapping
DOMAIN_MAP = {
    'arbitrary': 'Lifestyle',
    'background': 'Background',
    'identity': 'Identity',
    'morality': 'Morality',
    'politics': 'Politics',
    'preferences': 'Preferences',
    'religion': 'Religion',
}


def load_chat_data():
    """Load chat condition data from individual corrected files."""
    dfs = []
    for match_type in ['high', 'low', 'random']:
        df = pd.read_csv(RAW_DIR / "chat" / f"{match_type}-match-pre-post-responses.csv")
        dfs.append(df)

    chat_df = pd.concat(dfs, ignore_index=True)
    chat_df['experiment'] = 'chat'
    return chat_df


def load_nochat_data():
    """Load no-chat condition data and compute tolerance."""
    nochat_df = pd.read_csv(RAW_DIR / "no-chat" / "v1.0.2-no-chat-all-match-pre-post-responses.csv")
    nochat_df['experiment'] = 'no-chat'

    # Compute tolerance from ownResponse and observedResponse
    nochat_df['matchedTolerance'] = (nochat_df['ownResponse'] - nochat_df['observedResponse']).abs()

    return nochat_df


def apply_duplicate_filters(chat_df, nochat_df):
    """
    Handle participants who appear in multiple conditions.

    Returns filtered dataframes and a log of actions taken.
    """
    log = []

    # Rule 1: Remove from no-chat if also in chat
    chat_pids = set(chat_df['pid'].unique())
    nochat_pids = set(nochat_df['pid'].unique())
    overlap = chat_pids & nochat_pids

    if overlap:
        log.append(f"Removed {len(overlap)} participants from no-chat (also in chat)")
        nochat_df = nochat_df[~nochat_df['pid'].isin(overlap)]

    # Rule 2: Handle participants in multiple no-chat conditions
    # These rules come from preprocess_for_simulations.ipynb
    nochat_filters = [
        # (pid, matchType_to_REMOVE, reason)
        ('5c395df5f5ebd50001850900', 'low', 'in low & high no-chat, keep high'),
        ('6742d74e9f7b3b029a3791a6', 'low', 'in low & high no-chat, keep high'),
        ('63ee285c49c51e4f2826a68e', 'high', 'in low & high no-chat, keep low'),
        ('6500a84f0a4f87687fa51a20', 'high', 'in low & high no-chat, keep low'),
        ('672c4e78eeaae1dfe925f5e2', 'random', 'in low & random no-chat, keep low'),
    ]

    for pid, remove_matchType, reason in nochat_filters:
        before = len(nochat_df)
        nochat_df = nochat_df[~((nochat_df['pid'] == pid) & (nochat_df['matchType'] == remove_matchType))]
        if len(nochat_df) < before:
            log.append(f"Removed {pid[:12]}... {remove_matchType}: {reason}")

    return chat_df, nochat_df, log


def compute_question_type(row):
    """Classify question relative to focal question."""
    if row['question'] == row['matchedIdx']:
        return 'observed'
    elif row['preChatDomain'] == row['matchedDomain']:
        return 'same_domain'
    else:
        return 'diff_domain'


def load_srgi_scores():
    """Load SRGI composite scores."""
    srgi = pd.read_csv(RAW_DIR / "srgi-composite-scores.csv")
    return srgi


def compute_partner_responses(chat_df):
    """
    Compute partner's actual response on each question for chat participants.

    For each participant, find their partner in the same group and get the
    partner's preChatResponse for each question.
    """
    # Create a lookup of pid -> preChatResponse for each question
    response_lookup = chat_df.set_index(['groupId', 'pid', 'question'])['preChatResponse'].to_dict()

    # For each group, find the two participants
    group_partners = {}
    for group_id in chat_df['groupId'].unique():
        pids = chat_df[chat_df['groupId'] == group_id]['pid'].unique()
        if len(pids) == 2:
            group_partners[group_id] = {pids[0]: pids[1], pids[1]: pids[0]}

    def get_partner_response(row):
        group_id = row['groupId']
        pid = row['pid']
        question = row['question']

        if group_id not in group_partners:
            return np.nan
        if pid not in group_partners[group_id]:
            return np.nan

        partner_pid = group_partners[group_id][pid]
        key = (group_id, partner_pid, question)
        return response_lookup.get(key, np.nan)

    chat_df = chat_df.copy()
    chat_df['partner_response'] = chat_df.apply(get_partner_response, axis=1)
    return chat_df


def create_responses_csv(chat_df, nochat_df):
    """Combine and format data for analysis."""

    # Common columns
    cols = [
        'pid', 'question', 'preChatQuestion', 'preChatResponse', 'preChatDomain',
        'postChatQuestion', 'postChatResponse', 'postChatDomain', 'predictShared',
        'matchType', 'matchedDomain', 'matchedIdx', 'matchedQuestion',
        'matchedTolerance', 'experiment'
    ]

    # Chat has groupId and partner_response
    chat_subset = chat_df[cols + ['groupId', 'partner_response']].copy()

    # No-chat needs empty groupId and NaN partner_response
    nochat_subset = nochat_df[cols].copy()
    nochat_subset['groupId'] = ''
    nochat_subset['partner_response'] = np.nan

    # Combine
    combined = pd.concat([chat_subset, nochat_subset], ignore_index=True)

    # Add derived columns
    combined['question_type'] = combined.apply(compute_question_type, axis=1)
    combined['is_matched'] = combined['question'] == combined['matchedIdx']

    # Compatibility columns (same data, different names for legacy code)
    combined['participant_binary_prediction'] = combined['predictShared']
    combined['match_type'] = combined['matchType'].str.lower()

    # Join SRGI scores
    srgi = load_srgi_scores()
    combined = combined.merge(
        srgi[['pid', 'experiment', 'matchType', 'srgiResponse']],
        on=['pid', 'experiment', 'matchType'],
        how='left'
    )

    return combined


def validate(df):
    """Validate output and print summary."""

    # Check for issues
    nan_tol = df['matchedTolerance'].isna().sum()
    if nan_tol > 0:
        print(f"WARNING: {nan_tol} rows have NaN matchedTolerance")

    nan_srgi = df['srgiResponse'].isna().sum()
    if nan_srgi > 0:
        print(f"Note: {nan_srgi} rows have NaN srgiResponse (expected for some no-chat)")

    # Summary stats
    chat_n = df[df['experiment'] == 'chat']['pid'].nunique()
    nochat_n = df[df['experiment'] == 'no-chat']['pid'].nunique()

    print(f"\n=== OUTPUT SUMMARY ===")
    print(f"Chat participants: {chat_n}")
    print(f"No-chat participants: {nochat_n}")
    print(f"Total participants: {df['pid'].nunique()}")
    print(f"Total rows: {len(df)}")

    # Stance distribution (computed for display, not saved)
    stance = df['matchedTolerance'].apply(lambda x: 'opposing' if x > 1 else 'shared')
    print(f"\n=== STANCE DISTRIBUTION ===")
    stance_df = df.copy()
    stance_df['stance'] = stance
    stance_counts = stance_df.groupby(['experiment', 'stance'])['pid'].nunique()
    print(stance_counts)


def main():
    print("Loading raw data...")
    chat_df = load_chat_data()
    nochat_df = load_nochat_data()

    print(f"  Chat: {len(chat_df)} rows, {chat_df['pid'].nunique()} participants")
    print(f"  No-chat: {len(nochat_df)} rows, {nochat_df['pid'].nunique()} participants")

    print("\nApplying duplicate filters...")
    chat_df, nochat_df, log = apply_duplicate_filters(chat_df, nochat_df)
    for entry in log:
        print(f"  {entry}")

    print("\nComputing partner responses for chat participants...")
    chat_df = compute_partner_responses(chat_df)

    print("\nCreating responses.csv...")
    responses_df = create_responses_csv(chat_df, nochat_df)

    validate(responses_df)

    # Save
    responses_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")

    return responses_df


if __name__ == "__main__":
    main()
