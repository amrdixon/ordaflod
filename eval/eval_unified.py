from inspect_ai import Task, task, eval
from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.scorer import scorer, Score, mean, stderr
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from typing import Any, Dict, Tuple
import json
import uuid
import yaml
import random
import argparse
import datetime
from litellm import acompletion
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

# Import bot interfaces and adapters
from bot_interface import VocabBotInterface
from traditional_bot import TraditionalBot
from realtime_bot import RealtimeBot

# Load configuration
with open("eval_unified_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Load API keys from .env file specified in config
load_dotenv(CONFIG['env_file'])

from langfuse import get_client as _get_langfuse_client
from langfuse.api import LangfuseAPI

_langfuse_enabled = bool(os.getenv("LANGFUSE_SECRET_KEY"))
_lf = _get_langfuse_client() if _langfuse_enabled else None
_lf_api = (
    LangfuseAPI(
        base_url=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        username=os.getenv("LANGFUSE_PUBLIC_KEY"),
        password=os.getenv("LANGFUSE_SECRET_KEY"),
    )
    if _langfuse_enabled
    else None
)


def _push_langfuse_score(trace_id, name, value, comment=None):
    if not _langfuse_enabled or not _lf or not trace_id:
        return
    try:
        _lf.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type="NUMERIC",
            comment=comment[:1000] if comment else None,
        )
    except Exception:
        logger.warning(f"Failed to push Langfuse score '{name}' for trace {trace_id}", exc_info=True)


def format_conversation(conversation: list) -> str:
    """Format a conversation list as a readable transcript string."""
    lines = []
    for msg in conversation:
        role = "User" if msg['role'] == 'user' else "Bot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def create_vocab_bot() -> VocabBotInterface:
    """Factory function to create the appropriate bot based on configuration.

    Returns:
        VocabBotInterface: Instance of TraditionalBot or RealtimeBot

    Raises:
        ValueError: If architecture is not recognized
    """
    architecture = CONFIG['architecture']

    if architecture == "traditional":
        model = CONFIG['traditional']['model']
        return TraditionalBot(model=model)

    elif architecture == "realtime":
        api_key = os.getenv(CONFIG['realtime']['api_key_env'])
        return RealtimeBot(api_key=api_key)

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Must be 'traditional' or 'realtime'"
        )


def load_bot_config() -> Tuple[str, Dict[str, str]]:
    """Load bot configuration (prompt and vocabulary) based on architecture.

    Returns:
        Tuple of (prompt_template, vocab_dict)
    """
    architecture = CONFIG['architecture']
    bot_config = CONFIG[architecture]

    # Load prompt template
    prompt_fp = bot_config['prompt_fp']
    with open(prompt_fp, 'r') as f:
        prompt = f.read()

    # Load vocabulary list
    with open(CONFIG['vocab_fp'], 'r') as f:
        vocab_data = json.load(f)
        vocab_dict = vocab_data[CONFIG['vocab_key']]

    # Prepend a unique session ID to bust prompt caching
    prompt = f"<!-- session-id: {uuid.uuid4()} -->\n\n" + prompt

    return prompt, vocab_dict


def setup_user_bot() -> Tuple[str, str, Dict[str, str]]:
    """Load user bot resources (model, prompt template, vocab).

    Returns:
        Tuple of (user_model, prompt_template, vocab_words)

    Note: The prompt template still contains {{KNOWN_WORDS}} and {{UNKNOWN_WORDS}}
    placeholders. Call _build_user_prompt() inside each solver to get a fresh
    random known/unknown split per epoch.
    """
    template = open(CONFIG['user_bot_prompt_fp'], 'r').read()

    with open(CONFIG['vocab_fp'], 'r') as f:
        vocab_data = json.load(f)
        vocab_words = vocab_data[CONFIG['vocab_key']]

    user_model = CONFIG['user_bot_llm']
    return user_model, template, vocab_words


def fetch_production_sessions(limit: int = 50, since: datetime.datetime | None = None) -> list:
    """Fetch completed production sessions from Langfuse.

    Reads conversation and vocab_dict from trace metadata (stored by bots.py
    end_session()). Skips traces that pre-date this metadata schema.
    """
    if not _langfuse_enabled or not _lf:
        return []

    traces = _lf_api.trace.list(
        name="vocab-quiz-session",
        limit=limit,
        from_timestamp=since,
    )
    sessions = []

    for trace in traces.data:
        metadata = trace.metadata or {}
        vocab_dict = metadata.get("vocab_dict")
        if not vocab_dict:
            continue  # pre-refactor trace; no vocab data stored

        conversation = metadata.get("conversation")
        if not conversation:
            continue  # conversation not stored; skip this trace

        sessions.append({
            "trace_id": trace.id,
            "conversation": conversation,
            "vocab_dict": vocab_dict,
        })

    return sessions


def _build_user_prompt(template: str, vocab_words: Dict[str, str]) -> Tuple[str, list, list]:
    """Build a filled user bot prompt with a fresh random known/unknown split.

    Called once per epoch inside each solver so that every conversation
    uses a different split of known vs unknown words.

    Args:
        template: Raw prompt template containing {{KNOWN_WORDS}} and
                  {{UNKNOWN_WORDS}} placeholders (and optionally extra
                  suffixes like the review instruction).
        vocab_words: Full vocabulary dict {word: definition}.

    Returns:
        Tuple of (filled_prompt, known_words, unknown_words)
    """
    vocab_keys = list(vocab_words.keys())
    random.shuffle(vocab_keys)

    split_point = int(len(vocab_keys) * CONFIG['user_bot_percent_words_correct'])
    known_words = vocab_keys[:split_point]
    unknown_words = vocab_keys[split_point:]

    prompt = template.replace('{{KNOWN_WORDS}}', json.dumps(known_words, indent=2))
    prompt = prompt.replace('{{UNKNOWN_WORDS}}', json.dumps(unknown_words, indent=2))

    # Prepend a unique session ID to bust prompt caching
    prompt = f"<!-- session-id: {uuid.uuid4()} -->\n\n" + prompt

    return prompt, known_words, unknown_words


@task
def unified_completion_rate() -> Task:
    """Unified evaluation task for both architectures.

    Tests vocabulary bot completion rate using simulated conversations.
    """
    user_model, user_prompt_template, vocab_words = setup_user_bot()

    arch_cfg = CONFIG.get(CONFIG["architecture"], {})
    return Task(
        name=f"Unified Completion Rate Evaluation ({CONFIG['architecture']})",
        dataset=[Sample(input=json.dumps(vocab_words))],
        epochs=CONFIG['epochs'],
        metadata={
            "architecture": CONFIG["architecture"],
            "model": arch_cfg.get("model", "n/a"),
            "eval_llm_model": CONFIG["eval_llm_model"],
            "judge_llm_model": CONFIG["judge_llm_model"],
            "user_bot_llm": CONFIG["user_bot_llm"],
            "user_bot_percent_words_correct": CONFIG["user_bot_percent_words_correct"],
            "bot_prompt_fp": arch_cfg.get("prompt_fp", "n/a"),
            "user_bot_prompt_fp": CONFIG["user_bot_prompt_fp"],
        },
        solver=[
            unified_simulated_conversation(user_prompt_template, vocab_words, user_model),
            unified_final_word_list_request(),
            cleanup_bot()  # Clean up bot resources at the end
        ],
        scorer=[
            words_covered_rate_bot_perception(),
            words_covered_rate_ground_truth(),
            hint_quality_scorer()
        ]
    )


@task
def unified_review_accuracy() -> Task:
    """Evaluation task testing which missed words are reviewed after the quiz.

    The simulated user is configured to accept the bot's offer to review
    missed/tricky words. The conversation runs past 'beep boop' (end of quiz)
    and ends when 'end of line' is detected (end of review phase).

    The scorer measures how well the bot covered the student's actual unknown
    words during the review segment (between 'beep boop' and 'end of line').
    """
    user_model, user_prompt_template, vocab_words = setup_user_bot()

    # Append review instruction to template before passing to solver.
    # The {{KNOWN_WORDS}}/{{UNKNOWN_WORDS}} placeholders are still present
    # and will be filled per-epoch inside the solver.
    review_template = (
        user_prompt_template
        + "\n\nIMPORTANT: If the bot asks whether you would like to review "
        "missed or tricky words, always reply with a simple yes."
    )

    arch_cfg = CONFIG.get(CONFIG["architecture"], {})
    return Task(
        name=f"Unified Review Accuracy Evaluation ({CONFIG['architecture']})",
        dataset=[Sample(input=json.dumps(vocab_words))],
        epochs=CONFIG['epochs'],
        metadata={
            "architecture": CONFIG["architecture"],
            "model": arch_cfg.get("model", "n/a"),
            "eval_llm_model": CONFIG["eval_llm_model"],
            "judge_llm_model": CONFIG["judge_llm_model"],
            "user_bot_llm": CONFIG["user_bot_llm"],
            "user_bot_percent_words_correct": CONFIG["user_bot_percent_words_correct"],
            "bot_prompt_fp": arch_cfg.get("prompt_fp", "n/a"),
            "user_bot_prompt_fp": CONFIG["user_bot_prompt_fp"],
        },
        solver=[
            review_simulated_conversation(review_template, vocab_words, user_model),
            cleanup_bot()
        ],
        scorer=[
            reviewed_words_accuracy(),
            bot_perceived_review_accuracy()
        ]
    )


@task
def production_session_quality(since: datetime.datetime | None = None) -> Task:
    """Score real production sessions fetched from Langfuse.

    No simulated user needed — conversations are read from trace metadata
    (stored by bots.py end_session() alongside vocab_dict). Scores are pushed back to each
    session's Langfuse trace by the scorers, and also recorded in inspect_ai logs.
    """
    sessions = fetch_production_sessions(limit=CONFIG.get("production_eval_limit", 50), since=since)
    if not sessions:
        raise RuntimeError(
            "No production sessions found in Langfuse. "
            "Make sure Langfuse is configured and sessions have been completed."
        )

    samples = []
    for s in sessions:
        conversation = s["conversation"]
        beep_boop_index = next(
            (i for i, m in enumerate(conversation)
             if m["role"] == "assistant" and "beep boop" in m["content"].lower()),
            None,
        )
        samples.append(Sample(
            input=json.dumps(s["vocab_dict"]),
            metadata={
                "conversation": conversation,
                "langfuse_trace_id": s["trace_id"],
                "conversation_transcript": format_conversation(conversation),
                "beep_boop_index": beep_boop_index,
            }
        ))

    return Task(
        name="Production Session Quality",
        dataset=samples,
        solver=[passthrough_solver()],
        scorer=[
            words_covered_rate_ground_truth(),
            hint_quality_scorer(),
            bot_perceived_review_accuracy(),
        ]
    )


@solver
def passthrough_solver():
    """No-op solver for tasks where the dataset already contains the conversation."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return state
    return solve


# @task
# def unified_recall_accuracy() -> Task:
#     """Unified evaluation task testing bot's ability to recall missed words.

#     Tests whether the bot can accurately remember which words the student
#     struggled with during the conversation.
#     """
#     user_model, user_prompt, known_words, unknown_words = setup_user_bot()

#     return Task(
#         name=f"Unified Recall Accuracy Evaluation ({CONFIG['architecture']})",
#         dataset=[Sample(input=json.dumps(vocab_words))],
#         epochs=CONFIG['epochs'],
#         solver=[
#             unified_simulated_conversation(user_prompt, user_model, known_words, unknown_words),
#             recall_missed_words_request(),
#             cleanup_bot()  # Clean up bot resources at the end
#         ],
#         scorer=[
#             missed_words_recall_accuracy()
#         ]
#     )


@solver
def unified_simulated_conversation(
    user_prompt_template: str,
    vocab_words: Dict[str, str],
    user_model: str = "openai/gpt-4o",
):
    """Unified solver that simulates conversation with both bot architectures.

    Args:
        user_prompt_template: Raw system prompt template for the simulated user
                              bot (still contains {{KNOWN_WORDS}} and
                              {{UNKNOWN_WORDS}} placeholders).
        vocab_words: Full vocabulary dict {word: definition}.
        user_model: LLM model to use for user simulation.

    Returns:
        Solver function for Inspect AI
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Build a fresh known/unknown split for this epoch
        user_prompt, known_words, unknown_words = _build_user_prompt(
            user_prompt_template, vocab_words
        )

        # Create bot instance via factory
        bot = create_vocab_bot()

        # Load bot configuration
        prompt, vocab_dict = load_bot_config()

        # Initialize bot with prompt and vocabulary
        await bot.initialize(prompt, vocab_dict)
        state.metadata['langfuse_trace_id'] = bot.get_trace_id()

        # Store bot instance in metadata for reuse in next solver
        state.metadata['bot_instance'] = bot

        # Store ground truth known/unknown words for scorers
        state.metadata['known_words'] = [w.lower() for w in known_words]
        state.metadata['unknown_words'] = [w.lower() for w in unknown_words]

        # Initialize conversation tracking
        turn = 0
        max_turns = CONFIG['max_turns']
        session_complete = False

        # Store conversation in state metadata for scoring
        state.metadata['conversation'] = []

        # Initial user message
        user_message = "Hi!"

        print(f"\n=== Starting conversation with {CONFIG['architecture']} bot ===")

        while not session_complete and turn < max_turns:
            turn += 1

            # Send user message and get bot response
            bot_response = await bot.send_message(user_message)

            # Store messages in metadata
            state.metadata['conversation'].append({
                'role': 'user',
                'content': user_message
            })
            state.metadata['conversation'].append({
                'role': 'assistant',
                'content': bot_response
            })

            print(f"Turn {turn}:")
            print(f"  User: {user_message}")
            print(f"  Bot: {bot_response[:100]}...")
            if not bot_response:
                raise RuntimeError(
                    f"Bot returned empty response on turn {turn}. "
                    f"Last user message: {user_message!r}"
                )

            # Check if session is complete
            session_complete = bot.is_session_complete()

            if session_complete:
                state.metadata['token_usage_initial_pass'] = bot.get_token_usage()
                print("  [Session complete - 'beep boop' detected]")
                break

            # Generate user response using simulated user bot
            conversation_history = bot.get_history()

            # Flip roles for user's perspective (user <-> assistant)
            flipped_roles = {'assistant': 'user', 'user': 'assistant'}
            user_messages = [{'role': 'system', 'content': user_prompt}]
            user_messages.extend([
                {
                    'role': flipped_roles[msg['role']],
                    'content': msg['content']
                }
                for msg in conversation_history
            ])

            # Get user response from LiteLLM (async to avoid blocking the event loop)
            user_response = await acompletion(
                model=user_model,
                messages=user_messages,
                num_retries=5,
            )
            user_message = user_response['choices'][0]['message']['content']
            if user_message is None:
                raise RuntimeError(
                    f"Simulated user LLM returned null content on turn {turn}. "
                    f"Full response: {user_response['choices'][0]['message']}"
                )

        # Store final state information
        state.metadata['session_complete'] = session_complete
        state.metadata['turns'] = turn
        state.metadata['conversation_transcript'] = format_conversation(
            state.metadata['conversation']
        )

        print(f"\nConversation ended after {turn} turns")
        print(f"Session complete: {session_complete}\n")

        return state

    return solve


@solver
def unified_final_word_list_request():
    """Request final word list from bot to measure self-reported coverage.

    Reuses the bot instance from the previous solver.

    Returns:
        Solver function for Inspect AI
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Retrieve bot instance from metadata (created in previous solver)
        bot = state.metadata.get('bot_instance')

        if not bot:
            raise RuntimeError(
                "Bot instance not found in state. "
                "unified_simulated_conversation must run first."
            )

        # Request final word list from the already-initialized bot
        final_request = (
            "Thanks for the quiz! Can you please provide a list of all the words "
            "you quizzed me on during our conversation? Make sure the words are "
            "returned as a Python list in lowercase. Only return the list, do not "
            "include any other text."
        )

        print(f"\nRequesting final word list from bot...")
        final_response = await bot.send_message(final_request)
        print(f"Bot's word list response: {final_response}\n")

        # Store final request and response in metadata
        state.metadata['final_word_list_request'] = final_request
        state.metadata['final_word_list'] = final_response

        return state

    return solve


@solver
def recall_missed_words_request():
    """Request bot to recall which words the student struggled with.

    Returns:
        Solver function for Inspect AI
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Retrieve bot instance from metadata
        bot = state.metadata.get('bot_instance')

        if not bot:
            raise RuntimeError(
                "Bot instance not found in state. "
                "unified_simulated_conversation must run first."
            )

        # Request list of missed words
        recall_request = (
            "Thanks for the quiz! Can you please tell me which words I struggled with "
            "or got wrong during our conversation? Make sure the words are returned as "
            "a Python list in lowercase. Only return the list, do not include any other text."
        )

        print(f"\nRequesting missed words from bot...")
        recall_response = await bot.send_message(recall_request)
        print(f"Bot's missed words response: {recall_response}\n")

        # Store recall request and response in metadata
        state.metadata['missed_words_recall_request'] = recall_request
        state.metadata['missed_words_recall'] = recall_response

        return state

    return solve


@solver
def cleanup_bot():
    """Clean up bot resources after evaluation completes.

    Returns:
        Solver function for Inspect AI
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Retrieve and cleanup bot instance
        bot = state.metadata.get('bot_instance')

        if bot:
            print("Cleaning up bot resources...")
            await bot.cleanup()
            # Remove from metadata to avoid keeping reference
            del state.metadata['bot_instance']

        return state

    return solve


@solver
def review_simulated_conversation(
    user_prompt_template: str,
    vocab_words: Dict[str, str],
    user_model: str = "openai/gpt-4o",
):
    """Solver that runs a conversation through the review phase.

    Like unified_simulated_conversation but:
    - Does NOT stop at 'beep boop'; instead records its position in metadata
    - Stops when 'end of line' is detected (end of review phase)

    Args:
        user_prompt_template: Raw system prompt template for the simulated user
                              bot (still contains {{KNOWN_WORDS}} and
                              {{UNKNOWN_WORDS}} placeholders). Should already
                              include the review-acceptance instruction suffix.
        vocab_words: Full vocabulary dict {word: definition}.
        user_model: LLM model to use for user simulation.

    Returns:
        Solver function for Inspect AI
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Build a fresh known/unknown split for this epoch
        user_prompt, known_words, unknown_words = _build_user_prompt(
            user_prompt_template, vocab_words
        )

        bot = create_vocab_bot()
        prompt, vocab_dict = load_bot_config()
        await bot.initialize(prompt, vocab_dict)
        state.metadata['langfuse_trace_id'] = bot.get_trace_id()

        state.metadata['bot_instance'] = bot
        state.metadata['known_words'] = [w.lower() for w in known_words]
        state.metadata['unknown_words'] = [w.lower() for w in unknown_words]

        turn = 0
        max_turns = CONFIG['max_turns']
        session_complete = False
        beep_boop_index = None  # conversation list index of the 'beep boop' message

        state.metadata['conversation'] = []

        user_message = "Hi!"

        print(f"\n=== Starting review conversation with {CONFIG['architecture']} bot ===")

        while not session_complete and turn < max_turns:
            turn += 1

            bot_response = await bot.send_message(user_message)

            state.metadata['conversation'].append({'role': 'user', 'content': user_message})
            state.metadata['conversation'].append({'role': 'assistant', 'content': bot_response})

            print(f"Turn {turn}:")
            print(f"  User: {user_message}")
            print(f"  Bot: {bot_response[:100]}...")
            if not bot_response:
                raise RuntimeError(
                    f"Bot returned empty response on turn {turn}. "
                    f"Last user message: {user_message!r}"
                )

            # Record the first 'beep boop' occurrence (start of review phase)
            if beep_boop_index is None and 'beep boop' in bot_response.lower():
                beep_boop_index = len(state.metadata['conversation']) - 1
                state.metadata['token_usage_initial_pass'] = bot.get_token_usage()
                print("  [Beep boop detected - review phase begins]")

            # Session ends on 'end of line'
            if 'end of line' in bot_response.lower():
                state.metadata['token_usage_full_session'] = bot.get_token_usage()
                print("  [End of line detected - session complete]")
                session_complete = True
                break

            # Generate next user response using simulated user bot
            conversation_history = bot.get_history()
            flipped_roles = {'assistant': 'user', 'user': 'assistant'}
            user_messages = [{'role': 'system', 'content': user_prompt}]
            user_messages.extend([
                {'role': flipped_roles[msg['role']], 'content': msg['content']}
                for msg in conversation_history
            ])

            logger.debug(
                f"Turn {turn} — simulated user messages payload:\n"
                + json.dumps(user_messages, indent=2, ensure_ascii=False)
            )

            null_content_attempts = 0
            user_message = None
            while user_message is None:
                user_response = await acompletion(model=user_model, messages=user_messages, num_retries=5)
                user_message = user_response['choices'][0]['message']['content']
                if user_message is None:
                    null_content_attempts += 1
                    logger.warning(
                        f"Simulated user LLM returned null content on turn {turn}, "
                        f"attempt {null_content_attempts}. "
                        f"Full response: {user_response['choices'][0]['message']}"
                    )
                    if null_content_attempts >= 3:
                        raise RuntimeError(
                            f"Simulated user LLM returned null content on turn {turn} "
                            f"after {null_content_attempts} attempts. "
                            f"Full response: {user_response['choices'][0]['message']}"
                        )

        state.metadata['session_complete'] = session_complete
        state.metadata['turns'] = turn
        state.metadata['beep_boop_index'] = beep_boop_index
        state.metadata['conversation_transcript'] = format_conversation(
            state.metadata['conversation']
        )

        print(f"\nConversation ended after {turn} turns")
        print(f"Session complete (end of line): {session_complete}")
        print(f"Beep boop at conversation index: {beep_boop_index}\n")

        return state

    return solve


@scorer(metrics=[mean(), stderr()])
def reviewed_words_accuracy():
    """Score based on which vocabulary words were reviewed after 'beep boop'.

    Inspects bot messages between the 'beep boop' message and 'end of line'
    to find which vocabulary words were mentioned during the review phase.
    Computes precision and recall against the student's actual unknown words.
    Returns recall as the primary metric.

    Returns:
        Scorer function for Inspect AI
    """

    async def score(state: TaskState, target: Any):
        conversation = state.metadata.get('conversation', [])
        beep_boop_index = state.metadata.get('beep_boop_index')
        actual_missed = set(state.metadata.get('unknown_words', []))
        original_words = [w.lower() for w in json.loads(state.input).keys()]

        # Extract vocab words mentioned in bot messages after 'beep boop'
        reviewed_words = set()
        if beep_boop_index is not None:
            review_messages = conversation[beep_boop_index + 1:]
            for msg in review_messages:
                if msg['role'] == 'assistant':
                    for word in original_words:
                        if word in msg['content'].lower():
                            reviewed_words.add(word)

        # Precision: of words the bot reviewed, how many were actually missed?
        if reviewed_words:
            correct_set = reviewed_words.intersection(actual_missed)
            precision = len(correct_set) / len(reviewed_words)
        else:
            correct_set = set()
            precision = 0.0

        # Recall: of actually missed words, how many did the bot review?
        if actual_missed:
            correct_set = reviewed_words.intersection(actual_missed)
            recall = len(correct_set) / len(actual_missed)
        else:
            recall = 1.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        correct_set = reviewed_words.intersection(actual_missed)
        false_positives = reviewed_words - actual_missed
        false_negatives = actual_missed - reviewed_words

        explanation = (
            f'Words reviewed after beep boop: {", ".join(sorted(reviewed_words)) if reviewed_words else "none"}\n'
            f'Actual unknown words (ground truth): {", ".join(sorted(actual_missed)) if actual_missed else "none"}\n'
            f'Correctly reviewed: {", ".join(sorted(correct_set)) if correct_set else "none"}\n'
            f'Reviewed but already known (false positives): {", ".join(sorted(false_positives)) if false_positives else "none"}\n'
            f'Unknown but not reviewed (false negatives): {", ".join(sorted(false_negatives)) if false_negatives else "none"}\n'
            f'Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}'
        )

        _push_langfuse_score(
            state.metadata.get('langfuse_trace_id'),
            name="reviewed_words_accuracy_recall",
            value=recall,
            comment=f"Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}",
        )
        return Score(
            value=recall,
            explanation=explanation,
            metadata={
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'reviewed_words': list(reviewed_words),
                'actual_missed': list(actual_missed),
                'correct': list(correct_set),
                'false_positives': list(false_positives),
                'false_negatives': list(false_negatives),
                'beep_boop_found': beep_boop_index is not None
            }
        )

    return score


@scorer(metrics=[mean(), stderr()])
def bot_perceived_review_accuracy():
    """Score review accuracy using bot-perceived quiz performance as ground truth.

    Unlike reviewed_words_accuracy() which uses the simulated user's configured
    known/unknown lists as ground truth, this scorer uses an LLM judge to
    determine which words the bot perceived the student as struggling with
    during the quiz phase. This avoids penalising the bot when a student
    unexpectedly answers a word correctly despite being configured to 'not know' it.

    The scorer:
    1. Extracts the quiz-phase conversation (up to and including beep boop).
    2. Calls an LLM judge to classify each vocabulary word as
       'adequately_answered' or 'struggled' based on bot behaviour.
    3. Computes precision, recall, and F1 against that bot-perceived ground truth.
    4. Returns recall as the primary metric (matching reviewed_words_accuracy).

    Returns:
        Scorer function for Inspect AI
    """

    async def score(state: TaskState, target: Any):
        conversation = state.metadata.get('conversation', [])
        beep_boop_index = state.metadata.get('beep_boop_index')
        vocab_dict = {k.lower(): v for k, v in json.loads(state.input).items()}
        original_words = list(vocab_dict.keys())

        # Split conversation at beep_boop_index
        if beep_boop_index is not None:
            quiz_messages = conversation[:beep_boop_index + 1]
            review_messages = conversation[beep_boop_index + 1:]
        else:
            quiz_messages = conversation
            review_messages = []

        # Format quiz-phase transcript for LLM judge
        quiz_lines = []
        for msg in quiz_messages:
            role = "Student" if msg['role'] == 'user' else "Bot"
            quiz_lines.append(f"{role}: {msg['content']}")
        quiz_transcript = "\n".join(quiz_lines)

        # Call LLM judge to classify each word
        judge_prompt = (
            "You are evaluating a vocabulary quiz bot. "
            "Your job is to determine which words the student adequately answered "
            "during the quiz phase, based solely on bot behaviour.\n\n"
            "The vocabulary words and their definitions are:\n"
            f"{json.dumps(vocab_dict, indent=2)}\n\n"
            "Here is the quiz-phase transcript (before the review segment):\n"
            f"{quiz_transcript}\n\n"
            "For each vocabulary word, classify it as EITHER:\n"
            "- 'adequately_answered': The bot explicitly confirmed the student "
            "was correct WITHOUT first giving hints or corrections. "
            "Look for bot phrases like 'You've got it!', 'Exactly!', 'Spot on!', "
            "'Perfect!', 'Correct!', 'Nicely done!', 'Yes!', 'Right!' "
            "appearing as the FIRST substantive response to the student's answer "
            "(not after hints). The confirmation should come before any teaching.\n"
            "- 'struggled': The bot had to give hints, said 'Not quite', "
            "offered corrections, asked follow-up questions to guide the student, "
            "or had to explain/reveal the meaning before the student got it.\n\n"
            "IMPORTANT: Judge by what the BOT DID, not by whether the student "
            "technically knew the word. If the bot confirmed immediately without "
            "hints, classify as 'adequately_answered' even if the student's "
            "explanation was imprecise. If the bot gave any hint before confirming, "
            "classify as 'struggled'.\n\n"
            "Words not quizzed at all should be classified as 'struggled'.\n\n"
            "Return ONLY a JSON array (no other text, no markdown fences):\n"
            "[\n"
            "  {\n"
            '    "word": "vocabulary word (lowercase)",\n'
            '    "status": "adequately_answered" | "struggled",\n'
            '    "evidence": "brief quote or description from transcript supporting this"\n'
            "  }\n"
            "]\n\n"
            f"Include ALL of these vocabulary words: {original_words}"
        )

        response = await acompletion(
            model=CONFIG.get('judge_llm_model', CONFIG['eval_llm_model']),
            messages=[{"role": "user", "content": judge_prompt}],
            num_retries=5,
        )
        judge_response = response['choices'][0]['message']['content']

        # Parse JSON response, stripping markdown code fences if present
        word_assessments = []
        parse_failed = False
        try:
            clean = judge_response.strip()
            if clean.startswith('```'):
                clean = clean.split('```')[1]
                if clean.startswith('json'):
                    clean = clean[4:]
            word_assessments = json.loads(clean)
        except (json.JSONDecodeError, IndexError, ValueError):
            parse_failed = True

        if parse_failed:
            _push_langfuse_score(
                state.metadata.get('langfuse_trace_id'),
                name="bot_perceived_review_accuracy_recall",
                value=0.0,
                comment="LLM judge response could not be parsed as JSON.",
            )
            return Score(
                value=0.0,
                explanation=(
                    "LLM judge response could not be parsed as JSON. "
                    "Raw response:\n" + judge_response
                ),
                metadata={
                    'parse_failed': True,
                    'beep_boop_found': beep_boop_index is not None,
                    'sim_unknown_words': list(state.metadata.get('unknown_words', [])),
                }
            )

        # Build bot-perceived missed set
        bot_perceived_missed = set()
        for assessment in word_assessments:
            word = assessment.get('word', '').lower().strip()
            status = assessment.get('status', '')
            if word in original_words and status == 'struggled':
                bot_perceived_missed.add(word)

        # Find words reviewed in review phase (same logic as reviewed_words_accuracy)
        reviewed_words = set()
        for msg in review_messages:
            if msg['role'] == 'assistant':
                for word in original_words:
                    if word in msg['content'].lower():
                        reviewed_words.add(word)

        # Calculate precision, recall, F1
        if reviewed_words:
            correct_set = reviewed_words.intersection(bot_perceived_missed)
            precision = len(correct_set) / len(reviewed_words)
        else:
            correct_set = set()
            precision = 0.0

        if bot_perceived_missed:
            correct_set = reviewed_words.intersection(bot_perceived_missed)
            recall = len(correct_set) / len(bot_perceived_missed)
        else:
            recall = 1.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        correct_set = reviewed_words.intersection(bot_perceived_missed)
        false_positives = reviewed_words - bot_perceived_missed
        false_negatives = bot_perceived_missed - reviewed_words

        explanation = (
            f'Bot-perceived struggled words (LLM judge ground truth): '
            f'{", ".join(sorted(bot_perceived_missed)) if bot_perceived_missed else "none"}\n'
            f'Words reviewed after beep boop: '
            f'{", ".join(sorted(reviewed_words)) if reviewed_words else "none"}\n'
            f'Correctly reviewed: '
            f'{", ".join(sorted(correct_set)) if correct_set else "none"}\n'
            f'Reviewed but bot-perceived as known (false positives): '
            f'{", ".join(sorted(false_positives)) if false_positives else "none"}\n'
            f'Struggled but not reviewed (false negatives): '
            f'{", ".join(sorted(false_negatives)) if false_negatives else "none"}\n'
            f'Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}\n\n'
            f'--- LLM judge word-by-word assessment ---\n'
            + "\n".join(
                f'  {a.get("word","?")}: {a.get("status","?")} — {a.get("evidence","")}'
                for a in word_assessments
            )
        )

        _push_langfuse_score(
            state.metadata.get('langfuse_trace_id'),
            name="bot_perceived_review_accuracy_recall",
            value=recall,
            comment=f"Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}",
        )
        return Score(
            value=recall,
            explanation=explanation,
            metadata={
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'reviewed_words': list(reviewed_words),
                'bot_perceived_missed': list(bot_perceived_missed),
                'sim_unknown_words': list(state.metadata.get('unknown_words', [])),
                'correct': list(correct_set),
                'false_positives': list(false_positives),
                'false_negatives': list(false_negatives),
                'beep_boop_found': beep_boop_index is not None,
                'word_assessments': word_assessments,
                'parse_failed': False,
            }
        )

    return score


@scorer(metrics=[mean(), stderr()])
def words_covered_rate_bot_perception():
    """Score based on bot's self-reported vocabulary coverage.

    Returns:
        Scorer function for Inspect AI
    """

    async def score(state: TaskState, target: Any):
        # Extract word list from bot's final response
        final_response = state.metadata.get('final_word_list', '')

        # Get original vocabulary words
        original_words = json.loads(state.input).keys()
        original_words = [word.lower() for word in original_words]

        # Check which original words appear in the bot's final response
        covered_words = set()
        for word in original_words:
            if word in final_response.lower():
                covered_words.add(word)

        words_covered_rate = (
            len(covered_words) / len(original_words)
            if original_words else 0
        )

        conversation_transcript = state.metadata.get('conversation_transcript', '')
        final_request = state.metadata.get('final_word_list_request', '')

        _push_langfuse_score(
            state.metadata.get('langfuse_trace_id'),
            name="words_covered_rate_bot_perception",
            value=words_covered_rate,
            comment=f"Covered {len(covered_words)}/{len(original_words)} words",
        )
        return Score(
            value=words_covered_rate,
            explanation=(
                f'Bot reports covering: {", ".join(covered_words)}\n\n'
                f'--- Final request sent to bot ---\n{final_request}\n\n'
                f'--- Bot response to final request ---\n{final_response}\n\n'
                f'--- Quiz conversation transcript ---\n{conversation_transcript}'
            )
        )

    return score


@scorer(metrics=[mean(), stderr()])
def words_covered_rate_ground_truth():
    """Score based on actual vocabulary mentions in conversation.

    Returns:
        Scorer function for Inspect AI
    """

    async def score(state: TaskState, target: Any):
        # Get original vocabulary words
        original_words = json.loads(state.input).keys()
        original_words = [word.lower() for word in original_words]

        # Find words mentioned in conversation
        covered_words = set()
        conversation = state.metadata.get('conversation', [])

        for msg in conversation:
            if msg['role'] == 'assistant':
                for word in original_words:
                    if word.lower() in msg['content'].lower():
                        covered_words.add(word.lower())

        # Calculate coverage rate
        words_covered_rate = (
            len(covered_words) / len(original_words)
            if original_words else 0
        )

        conversation_transcript = state.metadata.get('conversation_transcript', '')

        _push_langfuse_score(
            state.metadata.get('langfuse_trace_id'),
            name="words_covered_rate_ground_truth",
            value=words_covered_rate,
            comment=f"Mentioned {len(covered_words)}/{len(original_words)} words",
        )
        return Score(
            value=words_covered_rate,
            explanation=(
                f'Bot mentioned during session: {", ".join(covered_words)}\n\n'
                f'--- Quiz conversation transcript ---\n{conversation_transcript}'
            )
        )

    return score


@scorer(metrics=[mean(), stderr()])
def missed_words_recall_accuracy():
    """Score based on bot's ability to recall which words student missed.

    Uses the known/unknown words from the simulated user setup as ground truth.
    Compares the bot's reported missed words against the unknown words.
    Uses recall as the primary metric.

    Returns:
        Scorer function for Inspect AI
    """

    async def score(state: TaskState, target: Any):
        # Get bot's recall of missed words
        recall_response = state.metadata.get('missed_words_recall', '')

        # Parse the word list
        bot_recalled_missed = list(
            recall_response.strip()
            .replace('[', '').replace(']', '')
            .replace("'", '').replace('"', '')
            .replace(' ', '').split(',')
        )
        bot_recalled_missed = {w.lower() for w in bot_recalled_missed if w}

        # Get ground truth unknown words from metadata
        # These are the words the simulated user was configured not to know
        actual_missed = set(state.metadata.get('unknown_words', []))

        # Calculate precision and recall
        if len(bot_recalled_missed) > 0:
            correct_recalls = bot_recalled_missed.intersection(actual_missed)
            precision = len(correct_recalls) / len(bot_recalled_missed)
        else:
            precision = 0.0

        if len(actual_missed) > 0:
            correct_recalls = bot_recalled_missed.intersection(actual_missed)
            recall = len(correct_recalls) / len(actual_missed)
        else:
            recall = 1.0  # If no words were missed, perfect recall is not recalling any

        # F1 score combines precision and recall
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        correct_set = bot_recalled_missed.intersection(actual_missed)
        false_positives = bot_recalled_missed - actual_missed
        false_negatives = actual_missed - bot_recalled_missed

        conversation_transcript = state.metadata.get('conversation_transcript', '')
        recall_request = state.metadata.get('missed_words_recall_request', '')
        recall_response = state.metadata.get('missed_words_recall', '')

        explanation = (
            f'Bot recalled as missed: {", ".join(sorted(bot_recalled_missed)) if bot_recalled_missed else "none"}\n'
            f'Actually missed (ground truth): {", ".join(sorted(actual_missed)) if actual_missed else "none"}\n'
            f'Correctly identified: {", ".join(sorted(correct_set)) if correct_set else "none"}\n'
            f'False positives: {", ".join(sorted(false_positives)) if false_positives else "none"}\n'
            f'False negatives: {", ".join(sorted(false_negatives)) if false_negatives else "none"}\n'
            f'Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1_score:.2f}\n\n'
            f'--- Recall request sent to bot ---\n{recall_request}\n\n'
            f'--- Bot response to recall request ---\n{recall_response}\n\n'
            f'--- Quiz conversation transcript ---\n{conversation_transcript}'
        )

        # Return recall as the primary metric
        # (What fraction of actually-missed words did the bot correctly identify?)
        _push_langfuse_score(
            state.metadata.get('langfuse_trace_id'),
            name="missed_words_recall_accuracy_recall",
            value=recall,
            comment=f"Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1_score:.2f}",
        )
        return Score(
            value=recall,
            explanation=explanation,
            metadata={
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'bot_recalled': list(bot_recalled_missed),
                'actual_missed': list(actual_missed),
                'correct': list(correct_set),
                'false_positives': list(false_positives),
                'false_negatives': list(false_negatives)
            }
        )

    return score


@scorer(metrics=[mean(), stderr()])
def hint_quality_scorer():
    """Score based on how indirect (non-definition-revealing) hints are.

    Uses an LLM judge to scan the conversation transcript, identify every
    hint the bot gave, and flag any that essentially state or closely
    paraphrase the dictionary definition.

    Score = fraction of hints that are appropriately indirect (0–1).
    A score of 1.0 means all hints were well-crafted; lower scores indicate
    the bot is giving away definitions instead of guiding the student.
    If no hints were given the score is 1.0.

    Returns:
        Scorer function for Inspect AI
    """

    async def score(state: TaskState, target: Any):
        vocab_dict = {k.lower(): v for k, v in json.loads(state.input).items()}
        conversation_transcript = state.metadata.get('conversation_transcript', '')

        # Ask LLM judge to extract all hints and assess quality in one call
        judge_prompt = (
            "You are evaluating the hint quality of a vocabulary quiz bot.\n\n"
            "The bot is quizzing a student on these vocabulary words and definitions:\n"
            f"{json.dumps(vocab_dict, indent=2)}\n\n"
            "Here is the conversation transcript:\n"
            f"{conversation_transcript}\n\n"
            'A "hint" is any bot message where the bot helps the student figure out '
            "a word's meaning without directly asking them to define it. This includes:\n"
            "- Giving a contextual clue or example sentence\n"
            "- Pointing to etymology or word parts\n"
            "- Describing the concept without naming it\n"
            "- Offering a related concept or analogy\n\n"
            'A hint is "too direct" if it essentially states or closely paraphrases '
            'the dictionary definition (e.g. for "predilection" whose definition is '
            '"an established preference for something", a too-direct hint would be '
            '"think about a strong preference you have for something").\n\n'
            "Identify ALL hints in the transcript and rate each one.\n"
            "Return ONLY a JSON array (no other text):\n"
            "[\n"
            "  {\n"
            '    "word": "the vocabulary word being hinted at",\n'
            '    "hint_text": "the exact hint text from the conversation",\n'
            '    "too_direct": true or false,\n'
            '    "reasoning": "brief explanation"\n'
            "  }\n"
            "]\n\n"
            "If no hints were given, return an empty array: []"
        )

        response = await acompletion(
            model=CONFIG['eval_llm_model'],
            messages=[{"role": "user", "content": judge_prompt}],
            num_retries=5,
        )
        judge_response = response['choices'][0]['message']['content']

        # Parse JSON response, stripping markdown code fences if present
        try:
            clean = judge_response.strip()
            if clean.startswith('```'):
                clean = clean.split('```')[1]
                if clean.startswith('json'):
                    clean = clean[4:]
            hints = json.loads(clean)
        except (json.JSONDecodeError, IndexError, ValueError):
            hints = []

        # Score = fraction of hints that are appropriately indirect
        if not hints:
            score_val = 1.0
            explanation = "No hints were identified in the conversation."
        else:
            too_direct = [h for h in hints if h.get('too_direct', False)]
            indirect = [h for h in hints if not h.get('too_direct', False)]
            score_val = len(indirect) / len(hints)

            lines = [
                f"Total hints: {len(hints)}",
                f"Appropriately indirect: {len(indirect)}",
                f"Too direct (reveals definition): {len(too_direct)}",
                f"Hint quality score: {score_val:.2f}",
                "",
            ]
            if too_direct:
                lines.append("TOO DIRECT hints:")
                for h in too_direct:
                    lines.append(f"  Word: {h.get('word', '?')}")
                    lines.append(f"  Hint: {h.get('hint_text', '?')[:200]}")
                    lines.append(f"  Reason: {h.get('reasoning', '')}")
                    lines.append("")
            if indirect:
                lines.append("GOOD (indirect) hints:")
                for h in indirect:
                    lines.append(f"  Word: {h.get('word', '?')}")
                    lines.append(f"  Hint: {h.get('hint_text', '?')[:200]}")
                    lines.append(f"  Reason: {h.get('reasoning', '')}")
                    lines.append("")
            lines.append(f"--- Quiz conversation transcript ---\n{conversation_transcript}")
            explanation = "\n".join(lines)

        _push_langfuse_score(
            state.metadata.get('langfuse_trace_id'),
            name="hint_quality",
            value=score_val,
            comment=explanation,
        )
        return Score(
            value=score_val,
            explanation=explanation,
            metadata={
                'hints': hints,
                'total_hints': len(hints),
                'too_direct_count': len([h for h in hints if h.get('too_direct', False)]),
                'indirect_count': len([h for h in hints if not h.get('too_direct', False)]),
            }
        )

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Orðaflóð evaluations")
    parser.add_argument(
        "--production-only",
        action="store_true",
        help="Score recent production traces from Langfuse only (skips simulated evals)",
    )
    parser.add_argument(
        "--lookback-days",
        type=float,
        default=None,
        help="How many days back to fetch production traces (overrides config). "
             "Only used with --production-only. Supports floats, e.g. 0.5 = 12 hours.",
    )
    args = parser.parse_args()

    log_subdir = CONFIG.get('log_subdir')
    eval_kwargs = {"log_dir": f"logs/{log_subdir}"} if log_subdir else {}

    if args.production_only:
        print(f"\n{'='*60}")
        print("Running Production Session Quality Evaluation")
        print(f"{'='*60}\n")
        if not _langfuse_enabled:
            raise RuntimeError("LANGFUSE_SECRET_KEY is not set — cannot fetch production traces.")
        lookback_days = args.lookback_days if args.lookback_days is not None else CONFIG.get("production_lookback_days", 1)
        since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=lookback_days)
        print(f"Fetching traces from the last {lookback_days} day(s) (since {since.isoformat()})\n")
        result_prod = eval(production_session_quality(since=since), **eval_kwargs)
    else:
        print(f"\n{'='*60}")
        print(f"Running Unified Evaluation")
        print(f"Architecture: {CONFIG['architecture']}")
        print(f"{'='*60}\n")

        print("\n--- Running Completion Rate Evaluation ---\n")
        result1 = eval(unified_completion_rate(), **eval_kwargs)

        print("\n--- Running Review Accuracy Evaluation ---\n")
        result3 = eval(unified_review_accuracy(), **eval_kwargs)

        if _langfuse_enabled:
            print("\n--- Running Production Session Quality Evaluation ---\n")
            result_prod = eval(production_session_quality(), **eval_kwargs)

    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}\n")
