import lunary
from openai import OpenAI
import os
import pytest

@pytest.fixture
def client():
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def test_monitor(client):
    lunary.monitor(client)

def test_identify_user(client):
    with lunary.users.identify("user1", user_props={"email": "123@gle.com"}):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello world"}]
        )
        assert completion.choices[0].message.content is not None

@lunary.agent("My great agent", user_id="123", tags=["test", "test2"])
def my_agent(client, a, b, c, test, test2):
    tool1_output = tool1("hello")
    output = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}],
    )
    print(output)
    tool2_output = tool2()
    return f"Agent output: {tool1_output}, {tool2_output}"

@lunary.tool(name="tool 1", user_id="123")
def tool1(a):
    return "Output 1"

@lunary.tool()
def tool2():
    return "Output 2"

def test_my_agent(client):
    result = my_agent(client, 1, 2, 3, test="sdkj", test2="sdkj")
    assert "Agent output: Output 1, Output 2" in result

def test_tool1():
    result = tool1("hello")
    assert result == "Output 1"

def test_tool2():
    result = tool2()
    assert result == "Output 2"