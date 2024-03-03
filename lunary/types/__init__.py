from typing import Literal, TypedDict


OpenAIMessageRole = Literal["system", "user", "assistant", "tool", "function"]

class OpenAIMessage(TypedDict):
    content: str
    role: OpenAIMessageRole 

class LunaryMessage(TypedDict):
    content: str
    role: OpenAIMessageRole
    tool_calls: list[dict[str, str]] | None # TODO: stronger typing
    tool_call_id: str | None
    function_call: dict[str, str] | None # TODO: stronger typing

__all__ = ["OpenAIMessage"]