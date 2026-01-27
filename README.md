# Orðaflóð

Orðaflóð is an AI-powered vocabulary study assistant intended for secondary students. It is a pupil-facing conversational agent whose primary function is to take a list of vocabulary words and quiz the pupil on their meaning through natural voice interaction. The study assistant uses definitions from Merriam-Webster to ensure high-quality, authoritative information.

## Projects

This repository contains two different implementations of the Orðaflóð voice agent:

### [voice2voice_web/](voice2voice_web/)
**Web-based Voice Agent**

A browser-based implementation using TypeScript, Vite, and OpenAI's Realtime API. Features include:
- Web interface with access code authentication
- Real-time voice interaction using OpenAI's Realtime API
- Word definition lookup via Merriam-Webster API
- Serverless API functions for token generation and dictionary lookups
- Deployed on Vercel

**Tech Stack**: TypeScript, Vite, OpenAI Realtime API, Vercel

See [voice2voice_web/](voice2voice_web/) directory for implementation details and deployment configuration.

### [traditional_speech_art/](traditional_speech_art/)
**Desktop Python Voice Agent**

A local Python application with full control over speech processing. Features include:
- Local speech recognition using OpenAI Whisper
- Text-to-speech using pyttsx3
- Claude Sonnet 4.5 for intelligent conversation via LiteLLM
- Voice activity detection (VAD) and silence detection
- Real-time interruption support
- Merriam-Webster dictionary integration

**Tech Stack**: Python, Whisper, pyttsx3, Claude API, LiteLLM

See [traditional_speech_art/README.md](traditional_speech_art/README.md) for setup instructions.

## Configuration

Both projects share API keys stored in a `.env` file at the root level.

### Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   MW_API_KEY=your_merriam_webster_key_here
   ACCESS_CODES=your-access-codes-comma-separated
   ```

### Required Keys by Project

- **voice2voice_web**: Requires `OPENAI_API_KEY`, `MW_API_KEY`, and `ACCESS_CODES`
- **traditional_speech_art**: Requires `ANTHROPIC_API_KEY` and `MW_API_KEY`

**Note**: `.env` files are gitignored and will never be committed to version control.

## Getting Started

Each project is self-contained with its own dependencies and setup instructions:

1. **For the web version**: Navigate to [voice2voice_web/](voice2voice_web/) and see the [README](voice2voice_web/README.md)
2. **For the desktop version**: Navigate to [traditional_speech_art/](traditional_speech_art/) and see the [README](traditional_speech_art/README.md)

## API Keys Required

- **OpenAI API Key** - For voice2voice_web's Realtime API
- **Anthropic API Key** - For traditional_speech_art's Claude integration
- **Merriam-Webster Dictionary API Key** - For both projects' word definitions

Register for API keys at:
- OpenAI: https://platform.openai.com/
- Anthropic: https://www.anthropic.com/
- Merriam-Webster: https://dictionaryapi.com/