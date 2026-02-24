# Voice2Voice Web - Orðaflóð

A browser-based voice agent for vocabulary study using OpenAI's Realtime API. This web application provides real-time voice interaction for helping students learn vocabulary words through natural conversation.

## Overview

This is a TypeScript/Vite web application that runs in the browser and connects to OpenAI's Realtime API for voice-to-voice conversation. The bot:
- Takes a list of vocabulary words to study
- Quizzes students on word meanings via voice
- Uses Merriam-Webster definitions for accurate information
- Provides real-time voice interaction powered by OpenAI
- Includes access code authentication for controlled access

## Features

- **Real-time Voice**: OpenAI Realtime API for natural voice conversation
- **Web Interface**: Simple browser-based UI with login and word input
- **Serverless Functions**: Vercel API routes for token generation and dictionary lookups
- **Dictionary Integration**: Merriam-Webster API for word definitions
- **Access Control**: Configurable access code for student authentication
- **Production Ready**: Configured for deployment on Vercel

## Tech Stack

- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool and dev server
- **OpenAI Realtime API** - Voice-to-voice conversation
- **Vercel** - Serverless deployment platform
- **Zod** - Runtime validation

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- OpenAI API key
- Merriam-Webster Dictionary API key

## Setup Instructions

### 1. Install dependencies

```bash
cd voice2voice_web
npm install
```

### 2. Configure environment variables

**For Local Development:**

Create a `.env.local` file in this directory with:

```
OPENAI_API_KEY=your_openai_key_here
MW_API_KEY=your_merriam_webster_key_here
ACCESS_CODES=your-access-code-1,your-access-code-2,your-access-code-3
```

**For Vercel Deployment:**

Set environment variables in the Vercel Dashboard:
1. Go to your project → Settings → Environment Variables
2. Add: `OPENAI_API_KEY`, `MW_API_KEY`, and `ACCESS_CODES`
3. Set scope to Production/Preview/Development as needed
4. Redeploy after adding variables

**Note**:
- `ACCESS_CODES` should be a comma-separated list of valid access codes
- The parent directory also has a `.env` file that you can use, or create a local `.env.local` for this project specifically
- Never commit `.env` or `.env.local` files to version control
- Local `.env.local` files are NOT deployed to Vercel - you must set variables in the dashboard

### 3. Run the development server

**Important**: Use Vercel CLI to run locally (not `npm run dev`) so the API endpoints work:

```bash
vercel dev
```

The app will be available at `http://localhost:3000`.

**Why `vercel dev`?** The API endpoints are Vercel serverless functions that only work in the Vercel environment. Using `vercel dev` simulates this environment locally.

## Available Scripts

- **`npm run dev`** - Start development server with hot reload
- **`npm run build`** - Build for production (output to `dist/`)
- **`npm run preview`** - Preview production build locally

## Project Structure

```
voice2voice_web/
├── index.html              # Main HTML file with UI
├── src/
│   ├── main.ts            # TypeScript entry point
│   ├── counter.ts         # Counter utility
│   └── style.css          # Styles
├── api/
│   ├── token.js           # Vercel function: Generate OpenAI session tokens
│   └── definitions.js     # Vercel function: Fetch word definitions
├── public/
│   └── prompt2.md         # System prompt for the voice agent (loaded by main.ts)
├── package.json           # Dependencies and scripts
├── tsconfig.json          # TypeScript configuration
├── vite.config.ts         # Vite configuration
└── vercel.json            # Vercel deployment config
```

## How It Works

1. **Login**: User enters an access code (configurable in the code)
2. **Word Input**: User provides a list of vocabulary words to study
3. **Voice Session**: App requests a session token from OpenAI via `/api/token`
4. **Real-time Conversation**: OpenAI's Realtime API handles voice-to-voice interaction
5. **Dictionary Lookups**: Word definitions fetched via `/api/definitions` from Merriam-Webster

## Deployment

This project is configured for deployment on Vercel:

### Deploy to Vercel

1. Install Vercel CLI (if not already installed):
```bash
npm install -g vercel
```

2. **IMPORTANT: Set environment variables FIRST** (before deploying)
   - Go to Vercel Dashboard → Your Project → Settings → Environment Variables
   - Add these variables:
     - `OPENAI_API_KEY` - Your OpenAI API key
     - `MW_API_KEY` - Your Merriam-Webster API key
     - `ACCESS_CODES` - Comma-separated list (e.g., `code1,code2,code3`)
   - Set scope to Production (and Preview/Development if needed)

3. Deploy to production:
```bash
vercel --prod
```

### Vercel Configuration

The `vercel.json` file configures:
- Build command: `npm run build`
- Output directory: `dist/`
- API routes under `/api/*`

## Access Codes

Access codes are configured via the `ACCESS_CODES` environment variable in your `.env.local` file (or `.env` in the parent directory).

**Format**: Comma-separated list
```
ACCESS_CODES=code1,code2,code3
```

**Example**:
```
ACCESS_CODES=demo-banana-potato,pokemon-spensa-banana,student-test-2024
```

To add or change access codes, update the `ACCESS_CODES` variable in your environment file.

## Customization

### Change System Prompt

Edit `public/prompt2.md` to customize how the voice agent behaves and responds to students. This file is loaded by the application at runtime.

### Change Access Code

In `index.html`, find the login form handler and update the hardcoded access code check.

### Modify UI

The UI is contained in `index.html` with inline styles. Edit the HTML structure and CSS to customize the appearance.

## API Keys Required

1. **OpenAI API Key** - For Realtime API access
   - Sign up at: https://platform.openai.com/
   - Requires payment method (Realtime API is a paid feature)

2. **Merriam-Webster Dictionary API Key** - For word definitions
   - Register at: https://dictionaryapi.com/
   - Free tier available

## Development Notes

- The app uses ES modules (`"type": "module"` in package.json)
- TypeScript is compiled by Vite during build
- API functions in `api/` are serverless functions for Vercel
- Hot reload is enabled during development with `npm run dev`

## Related Projects

See [../traditional_speech_art/](../traditional_speech_art/) for a Python-based desktop implementation using local Whisper and Claude instead of OpenAI's hosted API.
