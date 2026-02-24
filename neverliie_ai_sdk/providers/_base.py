from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Union
from .._types import Message, ChatCompletionResponse, Tools, ToolChoice, StreamingEvent


class BaseProvider(ABC):
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or self.default_base_url

    @property
    @abstractmethod
    def default_base_url(self) -> str:
        pass

    @abstractmethod
    def chat(
        self,
        messages: Union[str, List[Message]],
        model: str,
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        **kwargs
    ) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: Union[str, List[Message]],
        model: str,
        tools: Tools = None,
        tool_choice: ToolChoice = None,
        **kwargs
    ) -> Iterator[StreamingEvent]:
        pass

    def _normalize_messages(self, messages: Union[str, List[Message]]) -> List[Message]:
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        return messages

    def _normalize_streaming_content(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
        return ""
