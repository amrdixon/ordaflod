"""Post-session LLM-based evaluation for production vocab quiz sessions.

Scores are pushed to Langfuse asynchronously (background thread) so end_session()
returns immediately. Three scorers that work without simulated ground truth:

  - words_covered_rate_ground_truth : string match
  - hint_quality                    : LLM judge
  - bot_perceived_review_accuracy   : LLM judge (skipped if no "beep boop")
"""

import json
import logging
import os

from dotenv import load_dotenv
from langfuse import get_client as _get_langfuse_client
from litellm import completion

load_dotenv()

logger = logging.getLogger(__name__)

_langfuse_enabled = bool(os.getenv("LANGFUSE_SECRET_KEY"))
_lf = _get_langfuse_client() if _langfuse_enabled else None


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


def _format_transcript(conversation):
    """Format conversation as a readable transcript, excluding system messages."""
    lines = []
    for msg in conversation:
        if msg['role'] == 'system':
            continue
        role = "Student" if msg['role'] == 'user' else "Bot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _score_words_covered_rate(conversation, vocab_dict, trace_id):
    """Push words_covered_rate_ground_truth score based on string matching."""
    original_words = [w.lower() for w in vocab_dict.keys()]
    covered_words = set()

    for msg in conversation:
        if msg['role'] == 'assistant':
            for word in original_words:
                if word in msg['content'].lower():
                    covered_words.add(word)

    rate = len(covered_words) / len(original_words) if original_words else 0.0
    comment = f"Mentioned {len(covered_words)}/{len(original_words)} words"
    _push_langfuse_score(trace_id, "words_covered_rate_ground_truth", rate, comment)


def _score_hint_quality(conversation, vocab_dict, config, trace_id):
    """Push hint_quality score using an LLM judge (synchronous)."""
    vocab_lower = {k.lower(): v for k, v in vocab_dict.items()}
    conversation_transcript = _format_transcript(conversation)

    judge_prompt = (
        "You are evaluating the hint quality of a vocabulary quiz bot.\n\n"
        "The bot is quizzing a student on these vocabulary words and definitions:\n"
        f"{json.dumps(vocab_lower, indent=2)}\n\n"
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

    response = completion(
        model=config['eval_llm_model'],
        messages=[{"role": "user", "content": judge_prompt}],
        num_retries=5,
    )
    judge_response = response['choices'][0]['message']['content']

    try:
        clean = judge_response.strip()
        if clean.startswith('```'):
            clean = clean.split('```')[1]
            if clean.startswith('json'):
                clean = clean[4:]
        hints = json.loads(clean)
    except (json.JSONDecodeError, IndexError, ValueError):
        hints = []

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
        explanation = "\n".join(lines)

    _push_langfuse_score(trace_id, "hint_quality", score_val, explanation)


def _score_bot_perceived_review_accuracy(conversation, vocab_dict, config, trace_id):
    """Push bot_perceived_review_accuracy_recall score using an LLM judge.

    Skipped entirely if "beep boop" is not detected (no review phase).
    """
    vocab_lower = {k.lower(): v for k, v in vocab_dict.items()}
    original_words = list(vocab_lower.keys())

    # Find beep boop boundary
    beep_boop_index = None
    for i, msg in enumerate(conversation):
        if msg['role'] == 'assistant' and 'beep boop' in msg['content'].lower():
            beep_boop_index = i
            break

    if beep_boop_index is None:
        logger.info("No beep boop detected; skipping bot_perceived_review_accuracy scorer")
        return

    quiz_messages = conversation[:beep_boop_index + 1]
    review_messages = conversation[beep_boop_index + 1:]

    # Format quiz-phase transcript for LLM judge
    quiz_lines = []
    for msg in quiz_messages:
        if msg['role'] == 'system':
            continue
        role = "Student" if msg['role'] == 'user' else "Bot"
        quiz_lines.append(f"{role}: {msg['content']}")
    quiz_transcript = "\n".join(quiz_lines)

    judge_prompt = (
        "You are evaluating a vocabulary quiz bot. "
        "Your job is to determine which words the student adequately answered "
        "during the quiz phase, based solely on bot behaviour.\n\n"
        "The vocabulary words and their definitions are:\n"
        f"{json.dumps(vocab_lower, indent=2)}\n\n"
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

    response = completion(
        model=config['judge_llm_model'],
        messages=[{"role": "user", "content": judge_prompt}],
        num_retries=5,
    )
    judge_response = response['choices'][0]['message']['content']

    try:
        clean = judge_response.strip()
        if clean.startswith('```'):
            clean = clean.split('```')[1]
            if clean.startswith('json'):
                clean = clean[4:]
        word_assessments = json.loads(clean)
    except (json.JSONDecodeError, IndexError, ValueError):
        logger.warning("bot_perceived_review_accuracy: LLM judge response could not be parsed")
        _push_langfuse_score(trace_id, "bot_perceived_review_accuracy_recall", 0.0,
                             "LLM judge response could not be parsed as JSON.")
        return

    # Words the bot perceived student struggled with
    bot_perceived_missed = set()
    for assessment in word_assessments:
        word = assessment.get('word', '').lower().strip()
        if word in original_words and assessment.get('status') == 'struggled':
            bot_perceived_missed.add(word)

    # Words mentioned in review phase
    reviewed_words = set()
    for msg in review_messages:
        if msg['role'] == 'assistant':
            for word in original_words:
                if word in msg['content'].lower():
                    reviewed_words.add(word)

    if not bot_perceived_missed:
        recall = 1.0
        precision = 1.0
        f1 = 1.0
    else:
        correct_set = bot_perceived_missed & reviewed_words
        recall = len(correct_set) / len(bot_perceived_missed)
        precision = len(correct_set) / len(reviewed_words) if reviewed_words else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    comment = f"Recall: {recall:.2f}, Precision: {precision:.2f}, F1: {f1:.2f}"
    _push_langfuse_score(trace_id, "bot_perceived_review_accuracy_recall", recall, comment)


def run_session_evals(conversation, vocab_dict, trace_id, config):
    """Run all applicable production eval scorers and push results to Langfuse.

    Intended to be called in a background daemon thread from end_session().
    Calls _lf.flush() at the end to ensure scores are sent.

    Args:
        conversation: Snapshot of bot.conversation_history (list of role/content dicts)
        vocab_dict: Snapshot of bot.vocab_dict {word: definition}
        trace_id: Langfuse trace ID captured during session initialisation
        config: CONFIG dict from config.yaml (needs eval_llm_model, judge_llm_model)
    """
    for name, fn, args in [
        ("words_covered_rate", _score_words_covered_rate, (conversation, vocab_dict, trace_id)),
        ("hint_quality", _score_hint_quality, (conversation, vocab_dict, config, trace_id)),
        ("bot_perceived_review_accuracy", _score_bot_perceived_review_accuracy, (conversation, vocab_dict, config, trace_id)),
    ]:
        try:
            fn(*args)
            print(f"  ✓ session eval: {name}")
        except Exception as e:
            import traceback
            print(f"  ✗ session eval: {name} FAILED — {e}")
            traceback.print_exc()
    if _lf:
        _lf.flush()
    print("Post-session eval complete.")
