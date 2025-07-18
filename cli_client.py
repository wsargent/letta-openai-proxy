#!/usr/bin/env python3

import click
import requests
import json
import sys
import re
import textwrap
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.document import Document


class OpenAIClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        # Only add Authorization header if api_key is provided and non-empty
        if api_key and api_key.strip():
            self.headers["Authorization"] = f"Bearer {api_key}"

    def list_models(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/models", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            click.echo(f"Error listing models: {e}", err=True)
            return {}

    def chat_completion(self, model: str, messages: list, stream: bool = False, **kwargs) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        # Add agent_id if not already present (use model name as agent_id for Letta)
        if "agent_id" not in payload:
            payload["agent_id"] = model
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return {"stream": response}
            else:
                return response.json()
        except requests.exceptions.RequestException as e:
            click.echo(f"Error in chat completion: {e}", err=True)
            return {}

    def process_stream_response(self, response):
        """Process streaming response and yield content chunks"""
        full_content = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    
                    if data == '[DONE]':
                        break
                    
                    try:
                        json_data = json.loads(data)
                        if 'choices' in json_data and json_data['choices']:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                content = delta['content']
                                full_content += content
                                yield content
                    except json.JSONDecodeError:
                        continue
        
        return full_content


class DynamicCompleter(Completer):
    """Custom completer that provides dynamic completions for commands"""
    
    def __init__(self, client: OpenAIClient):
        self.client = client
        self._models_cache = []
        self._models_cache_timestamp = 0
        
        # Base slash commands
        self.base_commands = [
            '/help', '/quit', '/exit', '/q', '/models', '/model', '/clear', 
            '/history', '/temperature', '/max-tokens', '/stream', '/settings', '/refresh', '/url'
        ]
    
    def _get_models(self, force_refresh: bool = False):
        """Get models with caching (cache for 30 seconds)"""
        import time
        current_time = time.time()
        
        if force_refresh or current_time - self._models_cache_timestamp > 30:  # 30 second cache
            models_data = self.client.list_models()
            if models_data and "data" in models_data:
                # Store only combined "name (id)" format
                self._models_cache = []
                for model in models_data['data']:
                    model_id = model.get('id', '')
                    model_name = model.get('name', '')
                    
                    # Add only the combined format
                    if model_name and model_id:
                        combined = f"{model_name} ({model_id})"
                        self._models_cache.append(combined)
                        
                self._models_cache_timestamp = current_time
        
        return self._models_cache
    
    def get_completions(self, document: Document, complete_event):
        text = document.text
        
        # Handle slash commands
        if text.startswith('/'):
            parts = text[1:].split(' ')
            command = parts[0]
            
            # If we're completing the command itself
            if len(parts) == 1:
                for cmd in self.base_commands:
                    if cmd[1:].startswith(command):
                        yield Completion(cmd[1:], start_position=-len(command))
            
            # If we're completing arguments for /model command
            elif len(parts) >= 2 and command == 'model':
                partial_model = parts[-1] if len(parts) > 1 else ''
                models = self._get_models()
                
                for model in models:
                    if model.startswith(partial_model):
                        yield Completion(model, start_position=-len(partial_model))


class SettingsManager:
    def __init__(self):
        self.config_dir = Path.home() / ".letta-cli"
        self.settings_file = self.config_dir / "settings.json"
        self.config_dir.mkdir(exist_ok=True)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file, return defaults if file doesn't exist"""
        default_settings = {
            "model": None,
            "temperature": 0.7,
            "max_tokens": None,
            "stream": True,
            "base_url": "http://localhost:1416/v1",
            "api_key": ""
        }
        
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    # Merge with defaults to handle new settings
                    default_settings.update(saved_settings)
            except (json.JSONDecodeError, IOError) as e:
                click.echo(f"Warning: Could not load settings: {e}", err=True)
        
        return default_settings
    
    def save_settings(self, settings: Dict[str, Any]) -> None:
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except IOError as e:
            click.echo(f"Warning: Could not save settings: {e}", err=True)


class REPLContext:
    def __init__(self, client: OpenAIClient, model: Optional[str] = None):
        self.client = client
        self.conversation = []
        self.settings_manager = SettingsManager()
        
        # Load saved settings
        saved_settings = self.settings_manager.load_settings()
        self.model = model or saved_settings.get("model")
        self.settings = {
            "temperature": saved_settings.get("temperature", 0.7),
            "max_tokens": saved_settings.get("max_tokens"),
            "stream": saved_settings.get("stream", True)
        }
        
        # Initialize rich console for markdown rendering
        self.console = Console(force_terminal=True, legacy_windows=False)
        
        # Show loaded model if any
        if self.model:
            self._display_loaded_model()
    
    def _display_loaded_model(self):
        """Display the loaded model with friendly name if available"""
        try:
            # Get the friendly display name for the loaded model
            models_data = self.client.list_models()
            if models_data and "data" in models_data:
                for m in models_data["data"]:
                    if m.get('id') == self.model:
                        model_name = m.get('name', '')
                        if model_name:
                            click.echo(f"Loaded saved model: {model_name} ({self.model})")
                        else:
                            click.echo(f"Loaded saved model: {self.model}")
                        return
                # Model not found in current list
                click.echo(f"Loaded saved model: {self.model} (not currently available)")
            else:
                click.echo(f"Loaded saved model: {self.model}")
        except Exception:
            click.echo(f"Loaded saved model: {self.model}")

    def save_current_settings(self):
        """Save current settings to file"""
        settings_to_save = {
            "model": self.model,
            "temperature": self.settings["temperature"],
            "max_tokens": self.settings["max_tokens"],
            "stream": self.settings["stream"],
            "base_url": "http://localhost:1416/v1",  # Could be made configurable
            "api_key": ""  # Could be made configurable
        }
        self.settings_manager.save_settings(settings_to_save)
    
    def format_timestamp(self) -> str:
        """Get current time in HH:MM:SS format with bold formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"\033[1m[{timestamp}]\033[0m"
    
    def format_think_line(self, text: str, terminal_width: int = 80) -> str:
        """Format a thinking line with proper indentation, wrapping, and timestamp highlighting"""
        if not text.strip():
            return ""
        
        # Check if line starts with a timestamp pattern (- HH:MM:SS format)
        timestamp_pattern = r'^(\s*-\s*)(\d{2}:\d{2}:\d{2})\s*(.*)$'
        timestamp_match = re.match(timestamp_pattern, text.strip())
        
        if timestamp_match:
            prefix = timestamp_match.group(1)  # "- "
            timestamp = timestamp_match.group(2)  # "HH:MM:SS"
            content = timestamp_match.group(3)  # rest of the text
            
            # Format timestamp in bold cyan and content in dim italic
            formatted_timestamp = f"\033[1;36m{prefix}{timestamp}\033[0m"
            
            # Wrap the content part, accounting for the timestamp prefix length
            indent = "    "
            timestamp_prefix_len = len(prefix + timestamp + " ")
            content_indent = indent + " " * timestamp_prefix_len
            
            # Use much more of the terminal width for thinking content
            available_width = max(80, terminal_width - len(indent) - 8)  # Just leave small margin
            
            if content:
                wrapped_content = textwrap.fill(
                    content,
                    width=available_width,
                    subsequent_indent=""
                ).split('\n')
                
                # First line includes timestamp, subsequent lines are just content
                formatted_lines = []
                for i, line in enumerate(wrapped_content):
                    if i == 0:
                        # First line with timestamp
                        formatted_lines.append(f"{indent}{formatted_timestamp} \033[2;3m{line}\033[0m")
                    else:
                        # Continuation lines
                        formatted_lines.append(f"{content_indent}\033[2;3m{line}\033[0m")
                
                return '\n'.join(formatted_lines)
            else:
                # Just timestamp, no content
                return f"{indent}{formatted_timestamp}"
        else:
            # No timestamp, treat as regular thinking content
            indent = "    "
            # Use much more of the terminal width for thinking content
            available_width = max(80, terminal_width - len(indent) - 8)  # Just leave small margin
            
            wrapped_lines = textwrap.fill(
                text.strip(), 
                width=available_width,
                subsequent_indent=""
            ).split('\n')
            
            # Apply indentation and dim italics to each wrapped line
            formatted_lines = []
            for line in wrapped_lines:
                formatted_lines.append(f"{indent}\033[2;3m{line}\033[0m")
            
            return '\n'.join(formatted_lines)
    
    def format_markdown(self, content: str) -> str:
        """Convert markdown content to ANSI formatted text with clickable links"""
        # Always use basic formatting to ensure URLs are expanded
        return self.basic_markdown_format(content)
    
    def basic_markdown_format(self, content: str) -> str:
        """Enhanced markdown formatting using regex with URL expansion"""
        # Process line by line to handle bullet points and other formatting
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Handle bullet points (•, -, *, +)
            line = re.sub(r'^(\s*)[•\-\*\+]\s+', r'\1• ', line)
            
            # Bold: **text** or __text__
            line = re.sub(r'\*\*(.*?)\*\*', r'\033[1m\1\033[0m', line)
            line = re.sub(r'__(.*?)__', r'\033[1m\1\033[0m', line)
            
            # Italic: *text* or _text_ (but not if it's a bullet point)
            line = re.sub(r'(?<![\s•])\*(.*?)\*(?!\s)', r'\033[3m\1\033[0m', line)
            line = re.sub(r'(?<![\s•])_(.*?)_(?!\s)', r'\033[3m\1\033[0m', line)
            
            # Headers: # Header
            line = re.sub(r'^#{1}\s+(.*?)$', r'\033[1;36m\1\033[0m', line)
            line = re.sub(r'^#{2}\s+(.*?)$', r'\033[1;35m\1\033[0m', line)
            line = re.sub(r'^#{3}\s+(.*?)$', r'\033[1;34m\1\033[0m', line)
            
            # Code blocks: `code`
            line = re.sub(r'`([^`]+)`', r'\033[37;100m\1\033[0m', line)
            
            # Links: [text](url) - expand to show both text and URL for easy clicking
            def expand_link(match):
                text = match.group(1)
                url = match.group(2)
                # Always show both text and URL for maximum clickability
                return f'{text} - \033[4;34m{url}\033[0m'
            
            line = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', expand_link, line)
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def format_response(self, content: str) -> str:
        """Format response content with proper think block formatting and markdown"""
        # Find and format <think> blocks first
        def format_think_block(match):
            think_content = match.group(1).strip()
            # Add proper indentation and italics styling for think blocks
            lines = think_content.split('\n')
            formatted_lines = []
            for line in lines:
                if line.strip():
                    # Add indentation and italic formatting using ANSI codes
                    formatted_lines.append(f"    \033[3m{line.strip()}\033[0m")
                else:
                    formatted_lines.append("")
            return f"\n{chr(10).join(formatted_lines)}\n"
        
        # Replace <think>...</think> blocks with formatted versions (removing XML tags)
        formatted = re.sub(r'<think>(.*?)</think>', format_think_block, content, flags=re.DOTALL)
        
        # Then apply markdown formatting to the rest (but not to think blocks)
        # Split by think blocks to avoid double-formatting
        parts = re.split(r'(\n    \033\[2;3m.*?\033\[0m\n)', formatted)
        result_parts = []
        
        for part in parts:
            if '\033[2;3m' in part:  # This is a think block part
                result_parts.append(part)
            else:
                # Apply markdown formatting to non-think content
                result_parts.append(self.format_markdown(part))
        
        return ''.join(result_parts)

    def set_model(self, model: str):
        model = model.strip()
        # Resolve model name to ID if needed
        resolved_model = self.resolve_model_name(model)
        self.model = resolved_model
        
        # Show a nice display of what was set
        models_data = self.client.list_models()
        if models_data and "data" in models_data:
            for m in models_data["data"]:
                if m.get('id') == resolved_model:
                    model_name = m.get('name', '')
                    if model_name:
                        click.echo(f"Model set to: {model_name} ({resolved_model})")
                    else:
                        click.echo(f"Model set to: {resolved_model}")
                    # Save settings when model is changed
                    self.save_current_settings()
                    return
        
        click.echo(f"Model set to: {resolved_model}")
        # Save settings even if model not found in list
        self.save_current_settings()

    def send_message(self, message: str):
        if not self.model:
            click.echo("No model selected. Use /model <name> to set a model.", err=True)
            return

        # Add timestamp to user message display
        timestamp = self.format_timestamp()
        click.echo(f"{timestamp} You: {message}")

        self.conversation.append({"role": "user", "content": message})
        
        # Extract stream setting
        stream = self.settings.get("stream", True)
        settings_without_stream = {k: v for k, v in self.settings.items() if k != "stream" and v is not None}
        
        response = self.client.chat_completion(
            self.model, 
            self.conversation, 
            stream=stream,
            **settings_without_stream
        )
        
        response_timestamp = self.format_timestamp()
        
        if stream and response and "stream" in response:
            click.echo(f"{response_timestamp} Assistant: ")
            assistant_message = ""
            in_think_block = False
            line_buffer = ""
            
            # Get terminal width for proper wrapping
            try:
                terminal_width = click.get_terminal_size().columns
            except (OSError, AttributeError):
                terminal_width = 80  # fallback
            
            try:
                for chunk in self.client.process_stream_response(response["stream"]):
                    assistant_message += chunk
                    line_buffer += chunk
                    
                    # Check for think block start/end
                    if '<think>' in line_buffer:
                        in_think_block = True
                        # Remove the <think> tag from display
                        line_buffer = line_buffer.replace('<think>', '')
                    
                    if '</think>' in line_buffer:
                        in_think_block = False
                        # Remove the </think> tag and process any remaining content
                        parts = line_buffer.split('</think>')
                        if parts[0].strip():
                            # Format and display the last think line with wrapping
                            formatted = self.format_think_line(parts[0], terminal_width)
                            click.echo(formatted)
                        line_buffer = parts[1] if len(parts) > 1 else ""
                        # Continue processing the buffer instead of displaying immediately
                    
                    # Process complete lines
                    while '\n' in line_buffer:
                        line, line_buffer = line_buffer.split('\n', 1)
                        if line.strip():
                            if in_think_block:
                                # Display thinking lines with proper wrapping and indentation
                                formatted = self.format_think_line(line, terminal_width)
                                click.echo(formatted)
                            else:
                                # Display regular content with markdown formatting
                                formatted_line = self.format_markdown(line)
                                click.echo(formatted_line)
                
                # Handle any remaining content in buffer
                if line_buffer.strip():
                    if in_think_block:
                        formatted = self.format_think_line(line_buffer, terminal_width)
                        click.echo(formatted)
                    else:
                        formatted_buffer = self.format_markdown(line_buffer)
                        click.echo(formatted_buffer)
                elif not line_buffer and not assistant_message.endswith('\n'):
                    click.echo()  # Add final newline only if needed
                
                if assistant_message:
                    self.conversation.append({"role": "assistant", "content": assistant_message})
            except Exception as e:
                click.echo(f"\nError processing stream: {e}", err=True)
                
        elif not stream and response and "choices" in response:
            assistant_message = response["choices"][0]["message"]["content"]
            formatted_message = self.format_response(assistant_message)
            click.echo(f"{response_timestamp} Assistant: {formatted_message}\n")
            self.conversation.append({"role": "assistant", "content": assistant_message})
        else:
            click.echo("Error: No response received", err=True)

    def clear_conversation(self):
        self.conversation = []
        click.echo("Conversation cleared.")

    def show_conversation(self):
        if not self.conversation:
            click.echo("No conversation history.")
            return
        
        for i, msg in enumerate(self.conversation):
            role = msg["role"].title()
            content = msg["content"]
            # Add mock timestamps for history (since we don't store them)
            timestamp = f"\033[1m[{i:02d}:{30+i:02d}:{15+i*2:02d}]\033[0m"
            
            if role == "Assistant":
                formatted_content = self.format_response(content)
                click.echo(f"{timestamp} {role}: {formatted_content}")
            else:
                click.echo(f"{timestamp} {role}: {content}")
            click.echo()

    def set_temperature(self, temp: float):
        if 0 <= temp <= 2:
            self.settings["temperature"] = temp
            click.echo(f"Temperature set to: {temp}")
            self.save_current_settings()
        else:
            click.echo("Temperature must be between 0 and 2", err=True)

    def set_max_tokens(self, tokens: Optional[int]):
        self.settings["max_tokens"] = tokens
        if tokens:
            click.echo(f"Max tokens set to: {tokens}")
        else:
            click.echo("Max tokens limit removed")
        self.save_current_settings()

    def set_streaming(self, stream: bool):
        self.settings["stream"] = stream
        click.echo(f"Streaming {'enabled' if stream else 'disabled'}")
        self.save_current_settings()
    
    def show_settings(self):
        """Display current settings"""
        click.echo("Current settings:")
        click.echo(f"  Model: {self.model if self.model else 'None'}")
        click.echo(f"  Temperature: {self.settings['temperature']}")
        tokens = self.settings['max_tokens']
        click.echo(f"  Max tokens: {tokens if tokens else 'unlimited'}")
        click.echo(f"  Streaming: {'enabled' if self.settings['stream'] else 'disabled'}")
        click.echo(f"  Base URL: {self.client.base_url}")
        click.echo(f"  API Key: {'***' if self.client.api_key else 'None'}")
        click.echo(f"  Settings file: {self.settings_manager.settings_file}")
    
    def reload_client_from_env(self):
        """Reload the client with fresh environment variables"""
        env_base_url = os.getenv('LETTA_BASE_URL')
        if env_base_url and not env_base_url.endswith('/v1'):
            env_base_url = env_base_url.rstrip('/') + '/v1'
        
        env_api_key = os.getenv('LETTA_API_TOKEN', '')
        
        # Use environment variables if available, otherwise keep current values
        new_base_url = env_base_url or self.client.base_url
        new_api_key = env_api_key if env_api_key != '' else self.client.api_key
        
        # Create new client
        self.client = OpenAIClient(new_base_url, new_api_key)
        
        click.echo(f"Reloaded client configuration:")
        click.echo(f"  Base URL: {self.client.base_url}")
        click.echo(f"  API Key: {'***' if self.client.api_key else 'None'}")
    
    def resolve_model_name(self, model_name_or_id: str) -> str:
        """Resolve model name to ID, return the ID if it's already an ID or name if not found"""
        models_data = self.client.list_models()
        if models_data and "data" in models_data:
            for model in models_data["data"]:
                model_id = model.get('id', '')
                model_name = model.get('name', '')
                
                # Check if input matches the combined format "name (id)"
                combined_format = f"{model_name} ({model_id})" if model_name and model_id else ""
                if model_name_or_id == combined_format:
                    return model_id
                
                # Check individual components
                if model.get('name') == model_name_or_id:
                    return model.get('id', model_name_or_id)
                elif model.get('id') == model_name_or_id:
                    return model_name_or_id
        return model_name_or_id  # Return as-is if not found


@click.group(invoke_without_command=True)
@click.option('--base-url', default=None, help='API base URL (overrides saved setting)')
@click.option('--api-key', default=None, help='API key for authentication (overrides saved setting)')
@click.pass_context
def cli(ctx, base_url, api_key):
    """OpenAI CLI Client for Letta - Interactive REPL"""
    ctx.ensure_object(dict)
    
    # Load saved settings for base_url and api_key if not provided
    settings_manager = SettingsManager()
    saved_settings = settings_manager.load_settings()
    
    # Check environment variables, then saved settings, then defaults
    env_base_url = os.getenv('LETTA_BASE_URL')
    if env_base_url and not env_base_url.endswith('/v1'):
        env_base_url = env_base_url.rstrip('/') + '/v1'
    
    env_api_key = os.getenv('LETTA_API_TOKEN', '')
    
    final_base_url = base_url or env_base_url or saved_settings.get('base_url', 'http://localhost:1416/v1')
    final_api_key = api_key or env_api_key or saved_settings.get('api_key', '')
    
    ctx.obj['client'] = OpenAIClient(final_base_url, final_api_key)
    
    if ctx.invoked_subcommand is None:
        start_repl(ctx.obj['client'])


@cli.command()
@click.pass_context
def models(ctx):
    """List available models"""
    client = ctx.obj['client']
    models_data = client.list_models()
    
    if models_data and "data" in models_data:
        click.echo("Available models:")
        for model in models_data["data"]:
            model_id = model.get('id', 'Unknown')
            model_name = model.get('name', '')
            if model_name:
                click.echo(f"  - {model_name} ({model_id})")
            else:
                click.echo(f"  - {model_id}")
    else:
        click.echo("No models found or error occurred")


@cli.command()
@click.argument('model')
@click.argument('message')
@click.option('--temperature', type=float, default=0.7, help='Temperature for generation')
@click.option('--max-tokens', type=int, help='Maximum tokens to generate')
@click.option('--stream/--no-stream', default=True, help='Enable/disable streaming')
@click.pass_context
def complete(ctx, model, message, temperature, max_tokens, stream):
    """Send a single completion request"""
    client = ctx.obj['client']
    messages = [{"role": "user", "content": message}]
    
    kwargs = {"temperature": temperature, "stream": stream}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
        
    response = client.chat_completion(model, messages, **kwargs)
    
    if stream and response and "stream" in response:
        try:
            for chunk in client.process_stream_response(response["stream"]):
                click.echo(chunk, nl=False)
            click.echo()  # Final newline
        except Exception as e:
            click.echo(f"Error processing stream: {e}", err=True)
            sys.exit(1)
    elif not stream and response and "choices" in response:
        click.echo(response["choices"][0]["message"]["content"])
    else:
        click.echo("Error: No response received", err=True)
        sys.exit(1)


def start_repl(client: OpenAIClient):
    """Start the interactive REPL"""
    ctx = REPLContext(client)
    
    # Set up command history and dynamic autocomplete
    history = InMemoryHistory()
    completer = DynamicCompleter(client)
    
    click.echo("🤖 OpenAI CLI REPL - Type /help for commands, /quit to exit")
    click.echo("Arrow keys for history, Tab for command completion")
    click.echo("=" * 50)
    
    while True:
        try:
            user_input = prompt(
                "> ",
                history=history,
                completer=completer,
                complete_style=CompleteStyle.MULTI_COLUMN
            ).strip()
            
            if not user_input:
                continue
                
            # Handle REPL commands
            if user_input.startswith('/'):
                parts = user_input[1:].split(' ', 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if command in ['quit', 'exit', 'q']:
                    click.echo("Goodbye! 👋")
                    break
                elif command == 'help':
                    show_help()
                elif command == 'models':
                    # Force refresh models when explicitly requested
                    completer._models_cache = []
                    completer._models_cache_timestamp = 0
                    models_data = client.list_models()
                    if models_data and "data" in models_data:
                        click.echo("Available models:")
                        for model in models_data["data"]:
                            model_id = model.get('id', 'Unknown')
                            model_name = model.get('name', '')
                            if model_name:
                                click.echo(f"  - {model_name} ({model_id})")
                            else:
                                click.echo(f"  - {model_id}")
                elif command == 'model':
                    if args:
                        ctx.set_model(args)
                    else:
                        if ctx.model:
                            click.echo(f"Current model: {ctx.model}")
                        else:
                            click.echo("No model selected")
                elif command == 'clear':
                    ctx.clear_conversation()
                elif command == 'history':
                    ctx.show_conversation()
                elif command == 'temperature':
                    if args:
                        try:
                            temp = float(args)
                            ctx.set_temperature(temp)
                        except ValueError:
                            click.echo("Invalid temperature value", err=True)
                    else:
                        click.echo(f"Current temperature: {ctx.settings['temperature']}")
                elif command == 'max-tokens':
                    if args:
                        try:
                            tokens = int(args) if args != "none" else None
                            ctx.set_max_tokens(tokens)
                        except ValueError:
                            click.echo("Invalid max tokens value", err=True)
                    else:
                        tokens = ctx.settings['max_tokens']
                        click.echo(f"Current max tokens: {tokens if tokens else 'unlimited'}")
                elif command == 'stream':
                    if args:
                        if args.lower() in ['true', 'on', 'yes', '1']:
                            ctx.set_streaming(True)
                        elif args.lower() in ['false', 'off', 'no', '0']:
                            ctx.set_streaming(False)
                        else:
                            click.echo("Invalid streaming value. Use true/false, on/off, yes/no, or 1/0", err=True)
                    else:
                        stream = ctx.settings['stream']
                        click.echo(f"Streaming: {'enabled' if stream else 'disabled'}")
                elif command == 'settings':
                    ctx.show_settings()
                elif command == 'refresh':
                    # Force refresh models cache and other cached data
                    completer._models_cache = []
                    completer._models_cache_timestamp = 0
                    click.echo("Refreshed models cache. Use /models to see updated list.")
                elif command == 'url':
                    if args:
                        # Set a new URL directly
                        new_url = args.strip()
                        if not new_url.endswith('/v1'):
                            new_url = new_url.rstrip('/') + '/v1'
                        ctx.client = OpenAIClient(new_url, ctx.client.api_key)
                        completer._models_cache = []
                        completer._models_cache_timestamp = 0
                        click.echo(f"Updated Base URL to: {new_url}")
                    else:
                        # Reload from environment variables
                        ctx.reload_client_from_env()
                        completer._models_cache = []
                        completer._models_cache_timestamp = 0
                else:
                    click.echo(f"Unknown command: {command}. Type /help for available commands.", err=True)
            else:
                # Regular chat message
                ctx.send_message(user_input)
                
        except KeyboardInterrupt:
            click.echo("\nUse /quit to exit")
        except EOFError:
            click.echo("\nGoodbye! 👋")
            break
        except Exception as e:
            click.echo(f"Error: {e}", err=True)


def show_help():
    """Show REPL help"""
    help_text = """
Available commands:
  /help                 - Show this help message
  /quit, /exit, /q      - Exit the REPL
  /models               - List available models
  /model <name>         - Set the current model
  /model                - Show current model
  /clear                - Clear conversation history
  /history              - Show conversation history
  /temperature <value>  - Set temperature (0-2)
  /temperature          - Show current temperature
  /max-tokens <value>   - Set max tokens limit
  /max-tokens none      - Remove max tokens limit
  /max-tokens           - Show current max tokens
  /stream <true/false>  - Enable/disable streaming responses
  /stream               - Show current streaming setting
  /settings             - Show all current settings and config file location
  /refresh              - Refresh models cache (useful after changing LETTA_BASE_URL)
  /url                  - Reload client from environment variables (LETTA_BASE_URL)
  /url <new-url>        - Set a new base URL directly

Just type your message to chat with the selected model.
Settings are automatically saved when changed.
"""
    click.echo(help_text)


if __name__ == "__main__":
    cli()