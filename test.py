# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_openai import ChatOpenAI

# class A:
#   def __init__(self):
#     self.a = 1
#     pass


# chat = ChatOpenAI()
# res = chat.invoke([message])

# print(res)

from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

tools = [MoveFileTool()]
functions = [convert_to_openai_function(t) for t in tools]

message = model.invoke(
    [HumanMessage(content="move file foo to bar")], functions=functions
)

print(message)

model_with_tools = model.bind_tools(tools)
res = model_with_tools.invoke([HumanMessage(content="move file foo to bar")])

print(res.additional_kwargs.get("tool_calls"))
print(type(res.additional_kwargs.get("tool_calls")))