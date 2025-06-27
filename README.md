# Letta OpenAI Proxy

This project makes [Letta](https://docs.letta.com) agents available through an OpenAI-compatible API.  Agents are listed as models, and sending messages through the chat completion API will send messages to the selected Letta agent and receive reasoning messages.

## Set up

This project uses `uv` to run the application.  The usual uv methods apply:

```
uv sync
uv venv
source .venv/bin/activate
```

Copy the`env_example` to `.env` and set up your credentials:

```

LETTA_BASE_URL=http://your-letta-server

LETTA_API_TOKEN=your-letta-password-if-any
```

## Running

The server uses [Hayhooks](https://docs.haystack.deepset.ai/docs/hayhooks) to run:

```
uv run python app.py
```

The server will come up at http://localhost:1416

Please see the Hayhooks documentation for logging and configuration options.

## Using the Client

You can use any OpenAI API compatible client to chat with the agent.  I prefer [Open WebUI](https://docs.openwebui.com) but there are many options.

For your convenience, a simple command line client is included that you can run standalone:

```
uv run python cli_client.py
```

## Limitations

You cannot use tools or upload data sources with an agent currently.