from typing import TypedDict, List, Optional, Union, Literal, Dict, Any


class ToolFunction(TypedDict, total=False):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(TypedDict, total=False):
    type: str
    function: ToolFunction


class ToolCallFunction(TypedDict, total=False):
    name: str
    arguments: str


class ToolCall(TypedDict, total=False):
    id: str
    type: str
    function: ToolCallFunction
    thought_signature: str


class Message(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str]
    tool_calls: List[ToolCall]
    tool_call_id: str
    name: str


class ChatCompletionChoice(TypedDict, total=False):
    index: int
    message: Message
    finish_reason: str
    delta: Dict[str, Any]


class Usage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(TypedDict, total=False):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class StreamChunk(TypedDict, total=False):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]


class StreamContent(TypedDict):
    type: Literal["content"]
    content: str


class StreamToolCall(TypedDict):
    type: Literal["tool_call"]
    tool_call: ToolCall
    finish_reason: Optional[str]


StreamingEvent = Union[StreamContent, StreamToolCall]

ChatMessages = Union[str, List[Message]]

ToolChoice = Union[Literal["auto", "none", "required"], Dict[str, Any]]

Tools = List[Tool]
