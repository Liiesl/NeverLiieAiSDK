import json
from typing import List, Dict, Any, Iterator, Union, Optional
from ._base import BaseProvider
from .._client import HttpClient
from .._types import Message, ChatCompletionResponse, Tools, ToolChoice, StreamingEvent


class Mistral(BaseProvider):
    default_base_url = "https://api.mistral.ai/v1"

    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key, base_url)
        self._client = HttpClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={"Authorization": f"Bearer {self.api_key}"},
        )

    def chat(
        self,
        messages: Union[str, List[Message]],
        model: str = "mistral-small-latest",
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> ChatCompletionResponse:
        normalized_messages = self._normalize_messages(messages)
        
        payload: Dict[str, Any] = {
            "model": model,
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
        model: str = "mistral-small-latest",
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
            "model": model,
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
            
            # Handle regular content
            if delta.get("content"):
                yield {
                    "type": "content",
                    "content": delta["content"]
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
