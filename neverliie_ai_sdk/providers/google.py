import json
import os
from typing import List, Dict, Any, Iterator, Union
from ._base import BaseProvider
from .._client import HttpClient
from .._types import Message, ChatCompletionResponse, Tools, ToolChoice, StreamingEvent


class Google(BaseProvider):
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, base_url: str = None):
        super().__init__(api_key, base_url)
        self._client = HttpClient(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers={"content-type": "application/json"},
        )

    def _to_google_contents(self, messages: List[Message]) -> List[Dict[str, Any]]:
        contents = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system", "tool"] else "model"
            parts = []

            if msg.get("content"):
                parts.append({"text": msg["content"]})

            # Handle tool_calls from assistant
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    fc_part = {
                        "functionCall": {
                            "name": tool_call["function"]["name"],
                            "args": tool_call["function"].get("arguments", {})
                        }
                    }
                    if tool_call.get("thought_signature"):
                        fc_part["thoughtSignature"] = tool_call["thought_signature"]
                    parts.append(fc_part)

            # Handle tool response
            if msg.get("role") == "tool":
                parts = [{
                    "functionResponse": {
                        "name": msg.get("name", "unknown"),
                        "response": {"result": msg["content"]}
                    }
                }]

            if parts:
                contents.append({
                    "role": role,
                    "parts": parts,
                })
        return contents

    def _convert_tools_to_google(self, tools: Tools) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Google format."""
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                function_declarations.append({
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters", {"type": "object", "properties": {}})
                })
        return [{"functionDeclarations": function_declarations}]

    def _convert_tool_choice_to_google(self, tool_choice: ToolChoice) -> Any:
        """Convert OpenAI tool_choice to Google format."""
        if tool_choice == "auto":
            return "AUTO"
        elif tool_choice == "none":
            return "NONE"
        elif tool_choice == "required":
            return "ANY"
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [tool_choice["function"]["name"]]
                }
            }
        return None

    def chat(
        self,
        messages: Union[str, List[Message]],
        model: str = "gemini-1.5-flash",
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> ChatCompletionResponse:
        normalized_messages = self._normalize_messages(messages)
        contents = self._to_google_contents(normalized_messages)

        payload: Dict[str, Any] = {"contents": contents}

        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        if tools is not None:
            payload["tools"] = self._convert_tools_to_google(tools)
        if tool_choice is not None:
            converted = self._convert_tool_choice_to_google(tool_choice)
            if converted:
                payload["toolConfig"] = {"functionCallingConfig": {"mode": converted}} if isinstance(converted, str) else converted

        payload.update(kwargs)

        endpoint = f"/models/{model}:generateContent?key={self.api_key}"
        response = self._client.post(endpoint, data=payload)

        return self._normalize_response(response, model)

    def _normalize_response(self, response: Dict[str, Any], model: str) -> ChatCompletionResponse:
        content = ""
        tool_calls = []

        if response.get("candidates"):
            candidate = response["candidates"][0]
            if candidate.get("content", {}).get("parts"):
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        content += part["text"]
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append({
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": fc.get("name"),
                                "arguments": fc.get("args", {})
                            },
                            "thought_signature": part.get("thoughtSignature")
                        })

        usage = response.get("usageMetadata", {})

        message = {
            "role": "assistant",
            "content": content if content else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": response.get("candidates", [{}])[0].get("finishReason"),
            }],
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        }

    def chat_stream(
        self,
        messages: Union[str, List[Message]],
        model: str = "gemini-1.5-flash",
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
        contents = self._to_google_contents(normalized_messages)

        payload: Dict[str, Any] = {"contents": contents}

        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config

        if tools is not None:
            payload["tools"] = self._convert_tools_to_google(tools)
        if tool_choice is not None:
            converted = self._convert_tool_choice_to_google(tool_choice)
            if converted:
                payload["toolConfig"] = {"functionCallingConfig": {"mode": converted}} if isinstance(converted, str) else converted

        payload.update(kwargs)

        endpoint = f"/models/{model}:streamGenerateContent?key={self.api_key}&alt=sse"

        # Buffer for tracking tool calls
        tool_calls_emitted = set()
        has_tool_calls = tools is not None

        for chunk_json in self._client.post_stream(endpoint, data=payload):
            chunk = json.loads(chunk_json)
            
            if chunk.get("candidates"):
                candidate = chunk["candidates"][0]
                
                # Handle text content
                if candidate.get("content", {}).get("parts"):
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            yield {
                                "type": "content",
                                "content": part["text"]
                            }
                        
                        # Handle function calls (tool calls)
                        elif "functionCall" in part and has_tool_calls:
                            fc = part["functionCall"]
                            tool_id = f"call_{fc.get('name', 'unknown')}"
                            
                            if tool_id not in tool_calls_emitted:
                                tool_calls_emitted.add(tool_id)
                                yield {
                                    "type": "tool_call",
                                    "tool_call": {
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": fc.get("name"),
                                            "arguments": fc.get("args", {})
                                        },
                                        "thought_signature": part.get("thoughtSignature")
                                    },
                                    "finish_reason": "tool_calls"
                                }

    def close(self):
        self._client.close()
