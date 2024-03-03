from langchain_core.messages import AnyMessage, ToolMessage
from lunary.types import OpenAIMessageRole, LunaryMessage
from typing import cast

# TODO: chunks

def get_openai_role(message: AnyMessage) -> OpenAIMessageRole:
  if message.type == "chat":
    return message.role #TODO: handle the case where it's a role not supported by OpenAI 
  if message.type == "human":
    return "user"
  elif message. type == "ai":
    return "assistant"
  else:
    return message.type 


# TODO: return type
def parse_lc_message(lc_message: AnyMessage):
  if not isinstance(lc_message.content, str): # type: ignore
    # From https://python.langchain.com/docs/modules/model_io/concepts#messages, 
    # the value is not a string when it's a multi-modal input
    # We do not support such case for now, as few LLMs support it.
    print("[Lunary] Message content needs to be a string.")
    return  

  content = lc_message.content 
  role = get_openai_role(lc_message)


  '''
    # Tool calls:
    ## OpenAI
    - AIMessage, with tool/function calls in the additional_kwargs
      - function calls (deprecated): {'function_call': {'arguments': '{"source_path":"foo","destination_path":"bar"}', 'name': 'move_file'}}
      - tool calls: {'tool_calls': [{'id': 'call_NHQS3LWtajYKQ17rUikXHYOB', 'function': {'arguments': '{"source_path":"foo","destination_path":"bar"}', 'name': 'move_file'}, 'type': 'function'}]}
    
    ## Langchain Tools
    - ToolMessage (with a `tool_call_id` property). 

  '''
  # TODO: When is additional_kwargs["name"] used?
  # TODO: verify the exactitude of the comment above

  tool_calls = cast(list[dict[str, str]] | None, lc_message.additional_kwargs.get('tool_calls', None)) # type: ignore
  function_call = cast(dict[str, str] | None, lc_message.additional_kwargs.get('function_call', None)) # type: ignore

  tool_call_id = None
  if isinstance(lc_message, ToolMessage):
    tool_call_id = lc_message.tool_call_id

  message: LunaryMessage =  {
    "content": content,
    "role": role,
    "tool_calls": tool_calls,
    "tool_call_id": tool_call_id,
    "function_call": function_call,
  }

  return message



