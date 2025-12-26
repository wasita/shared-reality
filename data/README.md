# Data Documentation

This folder contains the behavioral data from our experiments.

## Files

### `responses.csv`
Main trial-level data with participant responses and predictions. Each row is one question for one participant.

| Column | Description |
|--------|-------------|
| `experiment` | Condition: `chat` (conversation) or `no-chat` (minimal info) |
| `pid` | Participant ID (Prolific ID) |
| `groupId` | Unique identifier for participant pair |
| `question` | Question number (1-35) |
| `preChatDomain` | Domain of question (arbitrary, background, identity, morality, politics, preferences, religion) |
| `preChatQuestion` | Question text |
| `preChatResponse` | Participant's own response (1-7 Likert scale) |
| `postChatResponse` | Participant's response after chat (chat condition only) |
| `predictShared` | Binary prediction: does partner share this attitude? (0=no, 1=yes) |
| `matchType` | Partner match condition: `high` (similar) or `low` (dissimilar) |
| `matchedIdx` | Index of the matched (discussed) question |
| `matchedDomain` | Domain of the matched question |
| `matchedQuestion` | Text of the matched question |
| `matchedTolerance` | Tolerance used for high/low matching |
| `srgiResponse` | Self-reported group identification (SRGI) scale response |
| `participant_binary_prediction` | Same as `predictShared` (derived column) |
| `match_type` | Same as `matchType` (lowercase, derived) |
| `is_matched` | Boolean: is this the matched/discussed question? |
| `partner_response` | Partner's actual response to this question |
| `question_type` | Category: `observed` (matched), `same_domain`, or `different_domain` |

### `questions.csv`
The 35 survey questions used in the study.

| Column | Description |
|--------|-------------|
| `num` | Question number (1-35) |
| `text` | Question text |
| `domain` | Domain category (7 domains, 5 questions each) |

### `messages.csv`
Conversation transcripts from the chat condition.

| Column | Description |
|--------|-------------|
| `absolute_timestamp` | Message timestamp (UTC) |
| `group_id` | Pair identifier |
| `author` | Anonymized sender (emoji) |
| `prolific_id` | Participant ID |
| `message_string` | Message content |
| `match_type` | Partner match condition |
| `task_ver` | Task version |
| `matched_domain` | Domain of discussed question |
| `matched_idx` | Index of discussed question |
| `matched_question` | Text of discussed question |
| `start_time` | Conversation start time |

### `llm_results/`
Pre-computed LLM (Gemini) prediction results for supplementary analyses.
