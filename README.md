# Orðaflóð

Orðaflóð is an AI-powered vocabulary study assistant intended for secondary students. It is a pupil-facing conversational agent whose primary function is to take a list of vocabulary words and quiz the pupil on their meaning through natural voice interaction. The study assistant uses definitions from Merriam-Webster to ensure high-quality, authoritative information.

> **Blog post:** [The Demo Trap: Why I Spent More Time Evaluating My AI Than Building It](https://datafoss.ai/blog/demo-trap.html) — the story behind this project, including how formal evaluation revealed that the bot was silently skipping required review steps and worked correctly only ~72–77% of the time despite looking great in demos. It also uncovers how rigorous evaluation transformed my coding agent into a research collaborator by giving it something to reason about.

## Why Evaluation Matters Here

Demos are deceptive. When building Orðaflóð, informal testing looked promising — but structured evaluation told a different story. The bot would silently skip protocol steps with no error, no warning, just a plausible-sounding response that was functionally wrong.

As AI shifts from assisting humans to driving them, the failure mode that matters most is the one that's invisible. The asymmetry between appearance in a demo and actual performance is a central challenge of deploying AI in high-stakes domains. The `eval/` framework in this repo addresses this.

## Projects

This repository contains two different implementations of the Orðaflóð voice agent and an evaluation framework:

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
**Desktop Voice Agent**

A local Python application with full control over speech processing. Features include:
- Local speech recognition using OpenAI Whisper
- Text-to-speech using pyttsx3
- Claude Sonnet 4.5 for intelligent conversation via LiteLLM
- Voice activity detection (VAD) and silence detection
- Real-time interruption support
- Merriam-Webster dictionary integration

**Tech Stack**: Python, Whisper, pyttsx3, Claude API, LiteLLM

See [traditional_speech_art/README.md](traditional_speech_art/README.md) for setup instructions.

### [eval/](eval/)
**Evaluation Framework**

A unified evaluation framework using Inspect AI to assess both bot architectures. Features include:
- Abstracted bot interface for testing both implementations
- Simulated user bot for automated testing
- Configurable evaluation scenarios
- Support for switching between traditional and realtime architectures
- Automated conversation logging and scoring

**Tech Stack**: Python, Inspect AI, LiteLLM, OpenAI SDK

See [eval/README.md](eval/README.md) for setup and usage instructions.

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
   ACCESS_CODES=your-access-codes-comma-separated  # Local dev only
   ```

### Required Keys by Project

- **voice2voice_web**: Requires `OPENAI_API_KEY` and `MW_API_KEY`
  - For local dev: Also needs `ACCESS_CODES` in `.env`
  - For Vercel deployment: Set `ACCESS_CODES` in Vercel Dashboard (Settings → Environment Variables)
- **traditional_speech_art**: Requires `ANTHROPIC_API_KEY` and `MW_API_KEY`
- **eval**: Requires keys for whichever architecture you're testing (see `eval_unified_config.yaml`)

**Note**: `.env` files are gitignored and will never be committed to version control.

## Getting Started

Each project is self-contained with its own dependencies and setup instructions:

1. **For the web version**: Navigate to [voice2voice_web/](voice2voice_web/) and see the [README](voice2voice_web/README.md)
2. **For the desktop version**: Navigate to [traditional_speech_art/](traditional_speech_art/) and see the [README](traditional_speech_art/README.md)
3. **For evaluation**: Navigate to [eval/](eval/) and see the [README](eval/README.md)

## API Keys Required

- **OpenAI API Key** - For voice2voice_web's Realtime API
- **Anthropic API Key** - For traditional_speech_art's Claude integration
- **Merriam-Webster Dictionary API Key** - For both projects' word definitions

Register for API keys at:
- OpenAI: https://platform.openai.com/
- Anthropic: https://www.anthropic.com/
- Merriam-Webster: https://dictionaryapi.com/