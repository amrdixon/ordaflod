# Traditional Speech Art - Orðaflóð

A Python-based desktop voice agent for vocabulary study. This implementation uses local speech recognition (Whisper), text-to-speech (pyttsx3), and Claude Sonnet 4.5 for intelligent conversation.

## Overview

This is a conversational AI assistant that helps students study vocabulary words through natural voice interaction. The bot:
- Takes a list of vocabulary words to study
- Quizzes students on word meanings via voice
- Uses Merriam-Webster definitions for accurate information
- Provides Claude-powered intelligent responses
- Supports real-time voice interaction with interruption detection

## Features

- **Speech-to-Text**: OpenAI Whisper for accurate transcription
- **Text-to-Speech**: pyttsx3 for natural voice output
- **AI Model**: Claude Sonnet 4.5 via LiteLLM
- **Voice Activity Detection**: Real-time silence detection and interruption support
- **Dictionary Integration**: Merriam-Webster API for authoritative definitions

## Prerequisites

Before running this project, you need to download the Whisper model:

1. Download the Whisper base model (base.pt) from [OpenAI's Whisper repository](https://github.com/openai/whisper)
2. Place it in: `traditional_speech_art/models/whisper/base.pt`

The model file is too large to include in the repository and is required for speech recognition to work.

## Setup Instructions

### 1. Create and activate virtual environment

```bash
cd traditional_speech_art
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root (one level up) with:

```
ANTHROPIC_API_KEY=your_anthropic_key_here
MW_API_KEY=your_merriam_webster_key_here
```

### 4. Configure application settings (optional)

The `config.yaml` file contains application configuration:

```yaml
llm_model: "anthropic/claude-sonnet-4-5-20250929"
prompt_fp: "prompt.md"
vocab_list_fp: "data/vocab_words.json"
vocab_list_key: "list1"
```

Adjust these settings to use different models, prompts, or vocabulary lists as needed.

### 5. Download Whisper model

Place the Whisper base model at: `models/whisper/base.pt`

### 6. Run the application

```bash
python main.py
```

## Project Structure

- `main.py` - Entry point for the application
- `bots.py` - VocabStudyBot and SpeechRecognitionBot classes
- `util.py` - Utilities including Merriam-Webster lookup and prompt loading
- `config.yaml` - Configuration file (model, prompt path, vocab list settings)
- `prompt.md` - System prompt template for the AI assistant (used by bots.py)
- `data/` - Directory containing vocabulary word lists and evaluation datasets
  - `vocab_words.json` - Vocabulary word lists with definitions
  - `vocab_eval_dataset.csv` - Dataset for evaluation
- `models/whisper/` - Directory for Whisper model (download separately)
- `requirements.txt` - Python dependencies
- `venv/` - Virtual environment (created during setup)

## Usage

Once running, the bot will:
1. Load the vocabulary words
2. Start a voice conversation
3. Quiz you on word meanings
4. Provide feedback and explanations
5. Listen for your voice responses and support natural interruptions

## Troubleshooting

**SSL Certificate Issues (macOS)**:
If you encounter SSL errors when fetching dictionary definitions, run:
```bash
python temp.py
```

This fixes SSL certificate validation for macOS.

## Dependencies

Key packages:
- `openai-whisper` - Speech recognition
- `pyttsx3` - Text-to-speech
- `litellm` - Claude API integration
- `sounddevice` / `soundfile` - Audio handling
- `webrtcvad` - Voice activity detection
- `requests` - HTTP requests for dictionary API
- `python-dotenv` - Environment variable management

See `requirements.txt` for complete list.

## API Keys Required

1. **Anthropic API Key** - For Claude Sonnet 4.5 access
2. **Merriam-Webster Dictionary API Key** - For word definitions

Both should be set in the `.env` file in the parent directory.
