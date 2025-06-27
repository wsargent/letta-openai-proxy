# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Letta OpenAI Proxy that makes Letta agents available through an OpenAI-compatible API. The project uses Hayhooks as the web framework and serves Letta agents as OpenAI models.

## Setup and Development Commands

```bash
# Environment setup
uv sync
uv venv
source .venv/bin/activate

# Copy environment configuration
cp env_example .env
# Edit .env to set LETTA_BASE_URL and LETTA_API_TOKEN

# Run the server
uv run python app.py

# Run the CLI client for testing
uv run python cli_client.py
```

## Architecture

### Core Components

1. **app.py** - Main FastAPI application that:
   - Overrides Hayhooks OpenAI endpoints (`/models` and `/chat/completions`)
   - Fetches Letta agents and presents them as OpenAI models
   - Routes chat requests to the Letta pipeline wrapper
   - Handles both streaming and non-streaming responses

2. **pipelines/letta_proxy/pipeline_wrapper.py** - Hayhooks pipeline wrapper containing:
   - `LettaChatGenerator` - Haystack component that communicates with Letta client
   - `PipelineWrapper` - Implements the Hayhooks interface for chat completions
   - Streaming response handling with reasoning/tool call formatting

### Key Integration Points

- **Hayhooks Integration**: The app patches Hayhooks OpenAI router endpoints to redirect to custom implementations
- **Letta Client**: Uses `letta-client` library to communicate with Letta server
- **OpenAI Compatibility**: Maintains OpenAI API format for models and chat completions

### Response Flow

1. Client calls `/v1/models` → Returns Letta agents as OpenAI models
2. Client calls `/v1/chat/completions` → Routed to `chat_completions_override()`
3. Request forwarded to `letta_proxy` pipeline wrapper
4. Pipeline uses `LettaChatGenerator` to communicate with Letta
5. Streaming responses formatted with `<think>` tags and timestamped reasoning

## Environment Variables

- `LETTA_BASE_URL` - Letta server URL (default: `http://letta:8283`)
- `LETTA_API_TOKEN` - Authentication token for Letta server
- `LETTA_CHAT_DEBUG_TOOL_STATEMENTS` - Enable detailed tool call logging

## Server Configuration

- Default port: 1416 (configured via Hayhooks settings)
- Supports both streaming and non-streaming chat completions
- 5-minute timeout for Letta requests with 3 retries