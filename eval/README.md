# Evaluation Framework - Orðaflóð

A unified evaluation framework using [Inspect AI](https://inspect.ai-safety-institute.org.uk/) to assess both bot architectures (traditional and realtime) through automated simulated conversations.

## Overview

This evaluation framework provides a consistent way to test and compare the two different Orðaflóð implementations:
- **Traditional architecture**: Claude Sonnet 4.5 via LiteLLM with text-based interaction
- **Realtime architecture**: OpenAI Realtime API (gpt-4o-realtime) with voice-to-voice interaction

The framework simulates realistic student-bot conversations by using a simulated user bot that knows some vocabulary words and doesn't know others, mirroring how a real student would interact with the system.

## Features

- **Unified Interface**: Abstract bot interface allows both architectures to be tested identically
- **Simulated User Bot**: GPT-4o-powered student simulator with configurable knowledge levels
- **Automated Conversations**: Multi-turn conversations that mimic real study sessions
- **Configurable Evaluation**: Switch between architectures via simple YAML configuration
- **Session Completion Detection**: Detects when the bot signals the session is complete
- **Comprehensive Logging**: All conversations logged to `logs/` directory for analysis
- **Observability**: Optional Langfuse tracing to monitor eval sessions separately from production

## Architecture

### Bot Interface Pattern

The framework uses an adapter pattern to unify different bot architectures:

```
VocabBotInterface (abstract)
├── TraditionalBot (LiteLLM/Claude adapter)
└── RealtimeBot (OpenAI Realtime API adapter)
```

All bots implement the same interface:
- `initialize()` - Set up the bot with prompt and vocabulary
- `send_message()` - Send user message, get bot response
- `get_history()` - Get conversation history
- `is_session_complete()` - Check if bot signaled completion
- `cleanup()` - Clean up resources (e.g., close WebSocket connections)

### Key Components

- **[bot_interface.py](bot_interface.py)** - Abstract interface defining the bot contract
- **[traditional_bot.py](traditional_bot.py)** - Adapter for Claude-based implementation using LiteLLM
- **[realtime_bot.py](realtime_bot.py)** - Adapter for OpenAI Realtime API implementation
- **[eval_unified.py](eval_unified.py)** - Main evaluation script using Inspect AI framework
- **[eval_unified_config.yaml](eval_unified_config.yaml)** - Configuration for evaluation runs
- **[simulated_user_prompt.txt](simulated_user_prompt.txt)** - System prompt for the simulated student

## Prerequisites

- Python 3.13 or higher
- API keys for the architecture you want to test (see Configuration section)
- Virtual environment (recommended)

## Setup Instructions

### 1. Create and activate virtual environment

```bash
cd eval
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Ensure you have a `.env` file in the project root (one level up) with the necessary API keys:

```bash
# In ordaflod/.env
OPENAI_API_KEY=your_openai_key_here        # Required for realtime architecture and user bot
ANTHROPIC_API_KEY=your_anthropic_key_here  # Required for traditional architecture and eval LLM
MW_API_KEY=your_merriam_webster_key_here   # Required for both architectures

# Optional: Langfuse observability (omit to disable tracing)
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Langfuse is optional — evals run normally without it. When enabled, each eval conversation is traced as a parent span (`eval-vocab-quiz-session`) tagged with `source=eval` and the model name, keeping eval traces distinct from production traces in the Langfuse UI.

### 4. Configure evaluation settings

Edit [eval_unified_config.yaml](eval_unified_config.yaml) to customize your evaluation:

```yaml
# Choose which architecture to test
architecture: "traditional"  # or "realtime"

# Vocabulary dataset
vocab_fp: "../traditional_speech_art/data/vocab_words3.json"
vocab_key: "list3"

# Traditional architecture settings
traditional:
  api_key_env: "OPENAI_API_KEY"
  model: "openai/gpt-4o"
  prompt_fp: "../traditional_speech_art/prompt.md"

# Realtime architecture settings
realtime:
  api_key_env: "OPENAI_API_KEY"
  prompt_fp: "../voice2voice_web/public/prompt2.md"

# Evaluation LLM and judge
eval_llm_model: "gemini/gemini-2.5-flash"
judge_llm_model: "anthropic/claude-sonnet-4-6"

# Simulated user bot settings
user_bot_llm: "gemini/gemini-2.5-flash"
user_bot_prompt_fp: "simulated_user_prompt.txt"
user_bot_percent_words_correct: 0.7  # 70% of words known, 30% unknown

# Conversation settings
max_turns: 40
epochs: 10
```

## Running Evaluations

### Basic Evaluation Run

```bash
inspect eval eval_unified.py
```

This will:
1. Load the configuration from `eval_unified_config.yaml`
2. Create the appropriate bot (traditional or realtime) based on the `architecture` setting
3. Initialize the simulated user bot with known/unknown word splits
4. Run automated conversations for each example in the dataset
5. Log all conversations to `logs/` directory
6. Display evaluation results

### View Results

Evaluation results are displayed in the terminal and include:
- Individual conversation scores
- Mean and standard error across all conversations
- Conversation transcripts in the logs directory

### Log Files

All conversations are logged to the `logs/` directory with timestamps:
- `logs/vocab_eval_YYYYMMDD_HHMMSS.log`

Each log contains:
- Configuration used
- Full conversation history
- Evaluation scores
- Session completion status

## Configuration Guide

### Switching Between Architectures

To test the **traditional** architecture:
```yaml
architecture: "traditional"
```

To test the **realtime** architecture:
```yaml
architecture: "realtime"
```

### Adjusting User Bot Knowledge

Control how many words the simulated user knows:
```yaml
user_bot_percent_words_correct: 0.7  # 70% known, 30% unknown
```

This simulates different student knowledge levels:
- `0.0` - Student knows no words (challenging for bot)
- `0.5` - Student knows half the words (balanced)
- `1.0` - Student knows all words (easy for bot)

### Customizing Conversation Length

Adjust maximum conversation turns:
```yaml
max_turns: 40  # Maximum back-and-forth exchanges
```

### Using Different Vocabulary Lists

To test with different word lists, update the vocab configuration:
```yaml
vocab_list_fp: "../traditional_speech_art/data/vocab_words.json"
vocab_list_key: "list2"  # Use a different list from the JSON file
```

## API Keys Required

Depending on which architecture you're testing:

### For Traditional Architecture Testing
- **Anthropic API Key** - For Claude Sonnet 4.5 (both the bot and evaluation LLM)
- **OpenAI API Key** - For GPT-4o (simulated user bot)
- **Merriam-Webster Dictionary API Key** - For word definitions

### For Realtime Architecture Testing
- **OpenAI API Key** - For both Realtime API (the bot) and GPT-4o (simulated user)
- **Anthropic API Key** - For Claude Sonnet 4.5 (evaluation LLM)
- **Merriam-Webster Dictionary API Key** - For word definitions

### For Observability (optional)
- **Langfuse Secret Key**, **Public Key**, and **Host** - For session tracing in Langfuse

## Project Structure

```
eval/
├── README.md                      # This file
├── bot_interface.py               # Abstract bot interface
├── traditional_bot.py             # Claude/LiteLLM adapter
├── realtime_bot.py                # OpenAI Realtime API adapter
├── eval_unified.py                # Main evaluation script (Inspect AI)
├── eval_unified_config.yaml       # Evaluation configuration
├── simulated_user_prompt.txt      # System prompt for simulated student
├── requirements.txt               # Python dependencies
├── logs/                          # Conversation logs (created at runtime)
└── venv/                          # Virtual environment (created during setup)
```

## How It Works

### Evaluation Flow

1. **Configuration Loading**: Load settings from `eval_unified_config.yaml`
2. **Bot Creation**: Factory function creates the appropriate bot based on architecture
3. **User Bot Setup**: Simulated student initialized with split of known/unknown words
4. **Conversation Loop**:
   - User bot generates student-like response
   - Bot receives message and responds
   - History tracked for both sides
   - Continue until max_turns or bot signals completion ("beep boop")
5. **Scoring**: Evaluate conversation quality using the eval LLM
6. **Logging**: Save full conversation transcript and results

### Simulated User Bot

The simulated user bot (Gemini 2.5 Flash by default) behaves like a real student:
- Gives short, natural answers (1-2 sentences)
- Paraphrases definitions for known words (never dictionary-perfect)
- Expresses genuine uncertainty for unknown words
- Reacts naturally to feedback ("oh nice!", "ah I see")
- Stays in character as a student

### Session Completion

Bots signal completion by saying "beep boop" (case-insensitive). The evaluation framework detects this and can end the conversation early.

## Development Notes

### Customizing Evaluation Criteria

The evaluation scoring is handled by Inspect AI's built-in scorers. To customize:
- Modify the scorer configuration in `eval_unified.py`
- Add custom scoring logic if needed

## Related Projects

- **[traditional_speech_art/](../traditional_speech_art/)** - The traditional Claude-based implementation
- **[voice2voice_web/](../voice2voice_web/)** - The OpenAI Realtime API web implementation

## Additional Resources

- [The Demo Trap: Why I Spent More Time Evaluating My AI Than Building It](https://datafoss.ai/blog/demo-trap.html) — blog post on why this eval framework exists
- [Inspect AI Documentation](https://inspect.ai-safety-institute.org.uk/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [OpenAI Realtime API Documentation](https://platform.openai.com/docs/guides/realtime)
