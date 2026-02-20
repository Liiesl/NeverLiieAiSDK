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
