import os
import asyncio
from anthropic import Anthropic, AsyncAnthropic

import lunary
from lunary.anthrophic import monitor, parse_message, agent


def sync_non_streaming():
    client = Anthropic()
    monitor(client)

    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-opus-20240229",
    )
    print(message.content)


async def async_non_streaming():
    client = monitor(AsyncAnthropic())

    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-opus-20240229",
    )
    print(message.content)


def sync_streaming():
    client = monitor(Anthropic())

    stream = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-opus-20240229",
        stream=True,
    )
    for event in stream:
        print(event)


async def async_streaming():
    client = monitor(AsyncAnthropic())

    stream = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-opus-20240229",
        stream=True,
    )
    async for event in stream:
        print(event)


def sync_stream_helper():
    client = Anthropic()
    monitor(client)

    with client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-opus-20240229",
    ) as stream:
        for event in stream:
            print(event)


async def async_stream_helper():
    client = monitor(AsyncAnthropic())

    async with client.messages.stream(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Say hello there!",
            }
        ],
        model="claude-3-opus-20240229",
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)
        print()

    message = await stream.get_final_message()
    print(message.to_json())


def extra_arguments():
    client = Anthropic()
    monitor(client)

    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-opus-20240229",
        tags=["translate"],
        user_id="user123",
        user_props={
            "name": "John Doe",
        },
        metadata={
            "test": "hello",
            "isTest": True,
            "testAmount": 123,
        },
    )
    print(message.content)


def anthrophic_bedrock():
    from anthropic import AnthropicBedrock

    client = AnthropicBedrock()

    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello!",
            }
        ],
        model="anthropic.claude-3-sonnet-20240229-v1:0",
    )
    print(message)


def tool_calls():
    from anthropic import Anthropic
    from anthropic.types import ToolParam, MessageParam

    client = monitor(Anthropic())

    user_message: MessageParam = {
        "role": "user",
        "content": "What is the weather in San Francisco, California?",
    }
    tools: list[ToolParam] = [
        {
            "name": "get_weather",
            "description": "Get the weather for a specific location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
    ]

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[user_message],
        tools=tools,
    )
    print(f"Initial response: {message.model_dump_json(indent=2)}")

    assert message.stop_reason == "tool_use"

    tool = next(c for c in message.content if c.type == "tool_use")
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            user_message,
            {"role": message.role, "content": message.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool.id,
                        "content": [{"type": "text", "text": "The weather is 73f"}],
                    }
                ],
            },
        ],
        tools=tools,
    )
    print(f"\nFinal response: {response.model_dump_json(indent=2)}")


async def async_tool_calls():
    client = monitor(AsyncAnthropic())
    async with client.messages.stream(
        max_tokens=1024,
        model="claude-3-haiku-20240307",
        tools=[
            {
                "name": "get_weather",
                "description": "Get the weather at a specific location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Unit for the output",
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        messages=[{"role": "user", "content": "What is the weather in SF?"}],
    ) as stream:
        async for event in stream:
            if event.type == "input_json":
                print(f"delta: {repr(event.partial_json)}")
                print(f"snapshot: {event.snapshot}")


@agent("DemoAgent")
def reconcilliation_tool_calls():
    from anthropic import Anthropic
    from anthropic.types import ToolParam, MessageParam

    thread = lunary.open_thread()
    client = monitor(Anthropic())

    user_message: MessageParam = {
        "role": "user",
        "content": "What is the weather in San Francisco, California?",
    }
    tools: list[ToolParam] = [
        {
            "name": "get_weather",
            "description": "Get the weather for a specific location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
    ]

    message_id = thread.track_message(user_message)

    with lunary.parent(message_id):
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[user_message],
            tools=tools,
            parent=message_id,
        )
        print(f"Initial response: {message.model_dump_json(indent=2)}")

        assert message.stop_reason == "tool_use"

        tool = next(c for c in message.content if c.type == "tool_use")

        for item in parse_message(message): thread.track_message(item)

        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            parent=message_id,
            messages=[
                user_message,
                {"role": message.role, "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool.id,
                            "content": [{"type": "text", "text": "The weather is 73f"}],
                        }
                    ],
                },
            ],
            tools=tools,
        )
        print(f"\nFinal response: {response.model_dump_json(indent=2)}")

        for item in parse_message(response):
            thread.track_message(item)

    return response


# sync_non_streaming()
# asyncio.run(async_non_streaming())

# sync_streaming()
# asyncio.run(async_streaming())

# extra_arguments()

# sync_stream_helper()
# asyncio.run(async_stream_helper())

# # anthrophic_bedrock()

# tool_calls()
# asyncio.run(async_tool_calls())

reconcilliation_tool_calls()
