# NeverLiie AI SDK

A minimal, unified Python SDK for interacting with multiple AI/LLM providers. Optimized for Nuitka compilation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/neverliie-ai-sdk)](https://pypi.org/project/neverliie-ai-sdk/)

## Features

- **Multi-provider support** - OpenAI, Anthropic (Claude), Google (Gemini), Mistral
- **Minimal dependencies** - Only requires `requests>=2.28.0`
- **Streaming responses** - Real-time content with SSE support
- **Tool calling** - Function/tool calling across all providers
- **OpenAI-compatible** - Generic provider for any OpenAI API-compatible endpoint
- **Type safety** - Full TypedDict support for messages, tools, and responses
- **Nuitka optimized** - Designed for compiling to standalone executables

## Installation

```bash
pip install neverliie-ai-sdk
```

## Quick Start

```python
from neverliie_ai_sdk import Mistral

client = Mistral(api_key="your-api-key")
response = client.chat(messages="Hello, world!")
print(response["choices"][0]["message"]["content"])
client.close()
```

## Supported Providers

| Provider | Import | Default Model | Base URL |
|----------|--------|---------------|----------|
| OpenAI | `from neverliie_ai_sdk import OpenAI` | gpt-4o-mini | https://api.openai.com/v1 |
| Anthropic | `from neverliie_ai_sdk import Anthropic` | claude-3-haiku-20240307 | https://api.anthropic.com/v1 |
| Google | `from neverliie_ai_sdk import Google` | gemini-1.5-flash | https://generativelanguage.googleapis.com/v1beta |
| Mistral | `from neverliie_ai_sdk import Mistral` | mistral-small-latest | https://api.mistral.ai/v1 |
| OpenAI Compatible | `from neverliie_ai_sdk import OpenAICompatible` | (configurable) | (configurable) |

## Usage Examples

### Simple Chat

```python
from neverliie_ai_sdk import OpenAI

client = OpenAI(api_key="your-api-key")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
]

response = client.chat(messages=messages, model="gpt-4o-mini")
print(response["choices"][0]["message"]["content"])

client.close()
```

### Streaming Responses

```python
from neverliie_ai_sdk import Anthropic

client = Anthropic(api_key="your-api-key")

for event in client.chat_stream(
    messages="Tell me a short story",
    model="claude-3-haiku-20240307"
):
    if event["type"] == "content":
        print(event["content"], end="")
    elif event["type"] == "tool_call":
        print("Tool call:", event["tool_call"])

client.close()
```

### Tool Calling

```python
from neverliie_ai_sdk import Google

client = Google(api_key="your-api-key")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country"
                }
            },
            "required": ["location"]
        }
    }
}]

response = client.chat(
    messages="What's the weather in Paris?",
    model="gemini-1.5-flash",
    tools=tools,
    tool_choice="auto"
)

if response["choices"][0]["message"].get("tool_calls"):
    for tool_call in response["choices"][0]["message"]["tool_calls"]:
        print(f"Function: {tool_call['function']['name']}")
        print(f"Arguments: {tool_call['function']['arguments']}")

client.close()
```

### OpenAI-Compatible Endpoints

```python
from neverliie_ai_sdk import OpenAICompatible

# For OpenRouter, NVIDIA NIM, or any OpenAI-compatible API
client = OpenAICompatible(
    base_url="https://api.openrouter.com/api/v1",
    api_key="your-api-key"
)

response = client.chat(
    messages="Hello!",
    model="meta-llama/llama-3.1-8b-instruct"
)
print(response["choices"][0]["message"]["content"])

client.close()
```

## API Reference

### Client Initialization

All providers accept:
- `api_key` (str): Your API key for the service
- `base_url` (str, optional): Override the default base URL

### Methods

#### `chat(messages, model=None, tools=None, tool_choice=None, **kwargs)`

Send a chat completion request.

**Parameters:**
- `messages` (str | list[dict]): User message string or list of message dicts
- `model` (str, optional): Model name (uses provider default if not specified)
- `tools` (list[dict], optional): List of tool definitions
- `tool_choice` (str, optional): Tool choice strategy ("auto", "required", or "none")

**Returns:** Response dict with normalized format

#### `chat_stream(messages, model=None, tools=None, tool_choice=None, **kwargs)`

Send a streaming chat completion request.

**Returns:** Iterator of event dicts with `type` field ("content" or "tool_calls")

#### `close()`

Close the HTTP session.

## Error Handling

```python
from neverliie_ai_sdk import Mistral
from neverliie_ai_sdk._exceptions import APIError, AuthenticationError, RateLimitError

client = Mistral(api_key="invalid-key")

try:
    response = client.chat(messages="Hello")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except APIError as e:
    print(f"API error: {e}")

client.close()
```

## License

MIT License - see LICENSE file for details.
