import json
from typing import List, Dict, Any, Iterator, Union
from ._base import BaseProvider
from .._client import HttpClient
from .._types import Message, ChatCompletionResponse, Tools, ToolChoice, StreamingEvent


class Anthropic(BaseProvider):
    default_base_url = "https://api.anthropic.com/v1"

    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key, base_url)
        self._client = HttpClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    def _to_anthropic_messages(self, messages: List[Message]) -> tuple:
        system = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "tool":
                # Convert tool response to Anthropic format
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": msg["content"]
                    }]
                })
            else:
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    # Convert tool_calls to Anthropic tool_use blocks
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tool_call in tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "input": tool_call["function"].get("arguments", {})
                        })
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": content_blocks
                    })
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": content,
                    })

        return system, anthropic_messages

    def _convert_tools_to_anthropic(self, tools: Tools) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return anthropic_tools

    def _normalize_response(self, response: Dict[str, Any]) -> ChatCompletionResponse:
        content = ""
        tool_calls = []

        if response.get("content"):
            for block in response["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": block.get("input", {})
                        }
                    })

        message = {
            "role": "assistant",
            "content": content if content else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": response.get("id"),
            "model": response.get("model"),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": response.get("stop_reason"),
            }],
            "usage": {
                "prompt_tokens": response.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("output_tokens", 0),
                "total_tokens": response.get("usage", {}).get("input_tokens", 0) + response.get("usage", {}).get("output_tokens", 0),
            },
        }

    def chat(
        self,
        messages: Union[str, List[Message]],
        model: str = "claude-3-haiku-20240307",
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        max_tokens: int = 1024,
        temperature: float = None,
        **kwargs
    ) -> ChatCompletionResponse:
        normalized_messages = self._normalize_messages(messages)
        system, anthropic_messages = self._to_anthropic_messages(normalized_messages)

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }

        if system:
            payload["system"] = system
        if tools is not None:
            payload["tools"] = self._convert_tools_to_anthropic(tools)
        if tool_choice is not None:
            # Anthropic uses tool_choice with different format
            if tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
            elif tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                payload["tool_choice"] = {
                    "type": "tool",
                    "name": tool_choice["function"]["name"]
                }
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(kwargs)

        response = self._client.post("/messages", data=payload)

        return self._normalize_response(response)

    def chat_stream(
        self,
        messages: Union[str, List[Message]],
        model: str = "claude-3-haiku-20240307",
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        max_tokens: int = 1024,
        temperature: float = None,
        **kwargs
    ) -> Iterator[StreamingEvent]:
        """Stream chat completion with support for tool calls.
        
        Yields StreamingEvent objects:
        - {"type": "content", "content": "..."} for text content
        - {"type": "tool_call", "tool_call": {...}, "finish_reason": "..."} for complete tool calls
        """
        normalized_messages = self._normalize_messages(messages)
        system, anthropic_messages = self._to_anthropic_messages(normalized_messages)

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }

        if system:
            payload["system"] = system
        if tools is not None:
            payload["tools"] = self._convert_tools_to_anthropic(tools)
        if tool_choice is not None:
            if tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
            elif tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                payload["tool_choice"] = {
                    "type": "tool",
                    "name": tool_choice["function"]["name"]
                }
        if temperature is not None:
            payload["temperature"] = temperature
        payload.update(kwargs)

        # Buffer for accumulating tool calls during streaming
        current_tool_call: Dict[str, Any] = {}
        has_tool_calls = tools is not None
        in_tool_call = False

        for event_json in self._client.post_stream("/messages", data=payload):
            event = json.loads(event_json)
            event_type = event.get("type")

            # Handle text content
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if text:
                        yield {
                            "type": "content",
                            "content": text
                        }
                elif delta.get("type") == "input_json_delta":
                    # Accumulating tool call arguments
                    if in_tool_call and "partial_json" in delta:
                        if "arguments" not in current_tool_call.get("function", {}):
                            current_tool_call["function"]["arguments"] = ""
                        current_tool_call["function"]["arguments"] += delta["partial_json"]

            # Tool call starts
            elif event_type == "content_block_start" and has_tool_calls:
                content_block = event.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    in_tool_call = True
                    current_tool_call = {
                        "id": content_block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": content_block.get("name", ""),
                            "arguments": ""
                        }
                    }

            # Tool call ends
            elif event_type == "content_block_stop" and in_tool_call:
                in_tool_call = False
                # Parse the accumulated arguments as JSON if possible
                try:
                    import json
                    args_str = current_tool_call["function"].get("arguments", "")
                    if args_str:
                        current_tool_call["function"]["arguments"] = json.loads(args_str)
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
                
                yield {
                    "type": "tool_call",
                    "tool_call": current_tool_call,
                    "finish_reason": "tool_calls"
                }
                current_tool_call = {}

    def close(self):
        self._client.close()
