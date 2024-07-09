import os
import asyncio
from dotenv import load_dotenv
from anthropic import Anthropic, AsyncAnthropic

from lunary.anthrophic import monitor

load_dotenv()


def test_sync_non_streaming():
    client = Anthropic()
    monitor(client)

    message = client.messages.create(
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Hello, Claude",
        }],
        model="claude-3-opus-20240229",
    )
    print(message.content)


async def test_async_non_streaming():
    client = monitor(AsyncAnthropic())

    message = await client.messages.create(
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Hello, Claude",
        }],
        model="claude-3-opus-20240229",
    )
    print(message.content)


def test_sync_streaming():
    client = monitor(Anthropic())

    stream = client.messages.create(
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Hello, Claude",
        }],
        model="claude-3-opus-20240229",
        stream=True,
    )
    for event in stream:
        print(event)


async def test_async_streaming():
    client = monitor(AsyncAnthropic())

    stream = await client.messages.create(
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Hello, Claude",
        }],
        model="claude-3-opus-20240229",
        stream=True,
    )
    async for event in stream:
        print(event)


def test_sync_stream_helper():
    client = Anthropic()
    monitor(client)

    with client.messages.stream(
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Hello, Claude",
        }],
        model="claude-3-opus-20240229",
    ) as stream:
        for event in stream:
            print(event)

async def test_async_stream_helper():
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
        async for event in stream:
            print(event)

    message = await stream.get_final_message()
    print(message.to_json())


def test_extra_arguments():
    client = Anthropic()
    monitor(client)

    message = client.messages.create(
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Hello, Claude",
        }],
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


# test_sync_non_streaming()
# test_asyncio.run(async_non_streaming())

# test_sync_streaming()
# test_asyncio.run(async_streaming())

# test_extra_arguments()

# test_sync_stream_helper()
asyncio.run(test_async_stream_helper())