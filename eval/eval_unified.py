from inspect_ai import Task, task, eval
from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.scorer import scorer, Score, mean, stderr
from inspect_ai.dataset import csv_dataset
from inspect_ai.model import ChatMessageUser
from typing import Any, Dict, Tuple
import json
import yaml
import random
from litellm import completion
from dotenv import load_dotenv
import os

# Import bot interfaces and adapters
from bot_interface import VocabBotInterface
from traditional_bot import TraditionalBot
from realtime_bot import RealtimeBot

# Load configuration
with open("eval_unified_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# Load API keys from .env file
load_dotenv('../.env')


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
    vocab_list_fp = bot_config['vocab_list_fp']
    vocab_list_key = bot_config['vocab_list_key']

    with open(vocab_list_fp, 'r') as f:
        vocab_data = json.load(f)
        vocab_dict = vocab_data[vocab_list_key]

    return prompt, vocab_dict


def setup_user_bot() -> Tuple[str, str, list, list]:
    """Setup simulated user bot with known/unknown words.

    Returns:
        Tuple of (user_model, user_prompt, known_words, unknown_words)
    """
    # Load user bot prompt template
    user_prompt = open(CONFIG['user_bot_prompt_fp'], 'r').read()

    # Load vocabulary to split into known/unknown
    architecture = CONFIG['architecture']
    bot_config = CONFIG[architecture]

    vocab_list_fp = bot_config['vocab_list_fp']
    vocab_list_key = bot_config['vocab_list_key']

    with open(vocab_list_fp, 'r') as f:
        vocab_data = json.load(f)
        vocab_words = vocab_data[vocab_list_key]

    # Shuffle and split words
    vocab_words_keys = list(vocab_words.keys())
    random.shuffle(vocab_words_keys)

    split_point = int(len(vocab_words) * CONFIG['user_bot_percent_words_correct'])
    known_words = vocab_words_keys[:split_point]
    unknown_words = vocab_words_keys[split_point:]

    # Replace template variables
    user_prompt = user_prompt.replace(
        '{{KNOWN_WORDS}}',
        json.dumps(known_words, indent=2)
    )
    user_prompt = user_prompt.replace(
        '{{UNKNOWN_WORDS}}',
        json.dumps(unknown_words, indent=2)
    )

    user_model = CONFIG['user_bot_llm']
    return user_model, user_prompt, known_words, unknown_words


@task
def unified_completion_rate() -> Task:
    """Unified evaluation task for both architectures.

    Tests vocabulary bot completion rate using simulated conversations.
    """
    user_model, user_prompt, known_words, unknown_words = setup_user_bot()

    return Task(
        name=f"Unified Completion Rate Evaluation ({CONFIG['architecture']})",
        dataset=csv_dataset(CONFIG['eval_dataset_fp']),
        epochs=1,
        solver=[
            unified_simulated_conversation(user_prompt, user_model, known_words, unknown_words),
            unified_final_word_list_request(),
            cleanup_bot()  # Clean up bot resources at the end
        ],
        scorer=[
            words_covered_rate_bot_perception(),
            words_covered_rate_ground_truth()
        ]
    )


@task
def unified_recall_accuracy() -> Task:
    """Unified evaluation task testing bot's ability to recall missed words.

    Tests whether the bot can accurately remember which words the student
    struggled with during the conversation.
    """
    user_model, user_prompt, known_words, unknown_words = setup_user_bot()

    return Task(
        name=f"Unified Recall Accuracy Evaluation ({CONFIG['architecture']})",
        dataset=csv_dataset(CONFIG['eval_dataset_fp']),
        epochs=1,
        solver=[
            unified_simulated_conversation(user_prompt, user_model, known_words, unknown_words),
            recall_missed_words_request(),
            cleanup_bot()  # Clean up bot resources at the end
        ],
        scorer=[
            missed_words_recall_accuracy()
        ]
    )


@solver
def unified_simulated_conversation(
    user_prompt: str,
    user_model: str = "openai/gpt-4o",
    known_words: list = None,
    unknown_words: list = None
):
    """Unified solver that simulates conversation with both bot architectures.

    Args:
        user_prompt: System prompt for simulated user bot
        user_model: LLM model to use for user simulation
        known_words: Words the simulated user knows
        unknown_words: Words the simulated user doesn't know

    Returns:
        Solver function for Inspect AI
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Create bot instance via factory
        bot = create_vocab_bot()

        # Load bot configuration
        prompt, vocab_dict = load_bot_config()

        # Initialize bot with prompt and vocabulary
        await bot.initialize(prompt, vocab_dict)

        # Store bot instance in metadata for reuse in next solver
        state.metadata['bot_instance'] = bot

        # Store ground truth known/unknown words for scorers
        state.metadata['known_words'] = [w.lower() for w in (known_words or [])]
        state.metadata['unknown_words'] = [w.lower() for w in (unknown_words or [])]

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

            # Check if session is complete
            session_complete = bot.is_session_complete()

            if session_complete:
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

            # Get user response from LiteLLM
            user_response = completion(
                model=user_model,
                messages=user_messages
            )
            user_message = user_response['choices'][0]['message']['content']

        # Store final state information
        state.metadata['bot_history'] = bot.get_history()
        state.metadata['session_complete'] = session_complete
        state.metadata['turns'] = turn

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

        # Store final response in metadata
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

        # Store recall response in metadata
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


@scorer(metrics=[mean(), stderr()])
def words_covered_rate_bot_perception():
    """Score based on bot's self-reported vocabulary coverage.

    Returns:
        Scorer function for Inspect AI
    """

    async def score(state: TaskState, target: Any):
        # Extract word list from bot's final response
        final_response = state.metadata.get('final_word_list', '')

        # Parse the word list (expecting Python list format)
        bot_words = list(
            final_response.strip()
            .replace('[', '').replace(']', '')
            .replace("'", '').replace('"', '')
            .replace(' ', '').split(',')
        )

        # Get original vocabulary words
        original_words = json.loads(state.input).keys()
        original_words = [word.lower() for word in original_words]

        # Calculate coverage
        covered_words = set(bot_words).intersection(set(original_words))
        words_covered_rate = (
            len(covered_words) / len(original_words)
            if original_words else 0
        )

        return Score(
            value=words_covered_rate,
            explanation=(
                f'Bot reports covering: {", ".join(bot_words)}\n'
                f'Actually covered: {", ".join(covered_words)}'
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

        return Score(
            value=words_covered_rate,
            explanation=f'Bot mentioned during session: {", ".join(covered_words)}'
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

        explanation = (
            f'Bot recalled as missed: {", ".join(sorted(bot_recalled_missed)) if bot_recalled_missed else "none"}\n'
            f'Actually missed (ground truth): {", ".join(sorted(actual_missed)) if actual_missed else "none"}\n'
            f'Correctly identified: {", ".join(sorted(correct_set)) if correct_set else "none"}\n'
            f'False positives: {", ".join(sorted(false_positives)) if false_positives else "none"}\n'
            f'False negatives: {", ".join(sorted(false_negatives)) if false_negatives else "none"}\n'
            f'Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1_score:.2f}'
        )

        # Return recall as the primary metric
        # (What fraction of actually-missed words did the bot correctly identify?)
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


if __name__ == "__main__":
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Running Unified Evaluation")
    print(f"Architecture: {CONFIG['architecture']}")
    print(f"{'='*60}\n")

    # Run completion rate evaluation
    print("\n--- Running Completion Rate Evaluation ---\n")
    result1 = eval(unified_completion_rate())

    # Run recall accuracy evaluation
    print("\n--- Running Recall Accuracy Evaluation ---\n")
    result2 = eval(unified_recall_accuracy())

    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}\n")
