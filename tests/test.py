import openai
import asyncio
from llmonitor import monitor, tool, agent
# from tests import agent

openai.api_key = "..."

monitor(openai)


@agent("My great agent", user_id="123", tags=["test", "test2"])
def my_agent(a, b, c, test, test2):
    tool1("hello")
    output = openai.ChatCompletion.create(
        model="gpt-3.5-turboo",
        messages=[{"role": "user", "content": "Hello world"}],
        user_id="123",
    )
    print(output)
    tool2()
    return "Agent output"


@tool(name="tool 1", user_id="123")
def tool1(a):
    return "Output 1"


@tool()
def tool2():
    return "Output 2"


my_agent(1, 2, 3, test="sdkj", test2="sdkj")

print("Doneˆ")
