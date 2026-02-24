import json
from typing import List, Dict, Any, Iterator, Union
from ._base import BaseProvider
from .._client import HttpClient
from .._types import Message, ChatCompletionResponse, Tools, ToolChoice, StreamingEvent


class OpenAICompatible(BaseProvider):
    """Generic OpenAI-compatible provider for custom API endpoints.
    
    Supports any API that follows the OpenAI chat completions format including:
    - NVIDIA NIM (https://integrate.api.nvidia.com/v1/)
    - OpenRouter (https://openrouter.ai/api/v1)
    - OpenCode Zen (https://opencode.ai/zen/v1)
    - Custom self-hosted models
    """
    
    default_base_url = ""
    default_model = ""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str = None,
        extra_headers: Dict[str, str] = None
    ):
        """Initialize OpenAI-compatible provider.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for the API endpoint (required)
            default_model: Default model to use (optional, required if not set via class)
            extra_headers: Additional headers to include in requests (e.g., OpenRouter headers)
        """
        super().__init__(api_key, base_url)
        self._default_model = default_model or self.default_model
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if extra_headers:
            headers.update(extra_headers)
            
        self._client = HttpClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=headers,
        )

    def chat(
        self,
        messages: Union[str, List[Message]],
        model: str = None,
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> ChatCompletionResponse:
        normalized_messages = self._normalize_messages(messages)
        
        payload: Dict[str, Any] = {
            "model": model or self._default_model,
            "messages": normalized_messages,
        }
        
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)
        
        return self._client.post("/chat/completions", data=payload)

    def chat_stream(
        self,
        messages: Union[str, List[Message]],
        model: str = None,
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Iterator[StreamingEvent]:
        """Stream chat completion with support for tool calls.
        
        Yields StreamingEvent objects:
        - {"type": "content", "content": "..."} for text content
        - {"type": "tool_call", "tool_call": {...}, "finish_reason": "..."} for complete tool calls
        """
        normalized_messages = self._normalize_messages(messages)

        payload: Dict[str, Any] = {
            "model": model or self._default_model,
            "messages": normalized_messages,
            "stream": True,
        }

        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        # Buffer for accumulating tool calls during streaming
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
        has_tool_calls = tools is not None

        for chunk_json in self._client.post_stream("/chat/completions", data=payload):
            chunk = json.loads(chunk_json)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")

            if delta.get("content"):
                yield {
                    "type": "content",
                    "content": self._normalize_streaming_content(delta["content"])
                }

            # Handle tool calls - accumulate them
            if has_tool_calls and delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_buffer:
                        tool_calls_buffer[idx] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        }

                    func = tc.get("function", {})
                    if func.get("name"):
                        tool_calls_buffer[idx]["function"]["name"] += func["name"]
                    if func.get("arguments"):
                        tool_calls_buffer[idx]["function"]["arguments"] += func["arguments"]

            # When streaming ends with tool calls, yield them
            if finish_reason == "tool_calls" and tool_calls_buffer:
                for tool_call in tool_calls_buffer.values():
                    yield {
                        "type": "tool_call",
                        "tool_call": tool_call,
                        "finish_reason": "tool_calls"
                    }

    def close(self):
        self._client.close()
