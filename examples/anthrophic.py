
import os
import asyncio
from dotenv import load_dotenv
from anthropic import Anthropic, AsyncAnthropic

from lunary.anthrophic import monitor

load_dotenv()

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


# sync_non_streaming()
# asyncio.run(async_non_streaming())

# sync_streaming()
asyncio.run(async_streaming())

