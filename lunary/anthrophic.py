import typing as t
from functools import partial
from inspect import iscoroutine
from contextlib import AsyncContextDecorator, ContextDecorator

from . import (
    track_event,
    run_context,
    run_manager,
    logging,
    logger,
    user_props_ctx,
    user_ctx,
    traceback,
    tags_ctx,
)

try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message, ToolParam, MessageParam
    from anthropic.lib.streaming import (
        MessageStreamManager,
        AsyncMessageStreamManager,
        MessageStream,
        AsyncMessageStream,
    )
except ImportError:
    raise ImportError("Anthrophic SDK not installed!") from None

PARAMS_TO_CAPTURE = [
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "n",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    # Additional params
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
]


def __prop(
    target: t.Union[t.Dict, t.Any],
    property_or_keys: t.Union[t.List, str],
    default_value: t.Any = None
):
    if isinstance(property_or_keys, list):
        value = target
        for key in property_or_keys:
            value = __prop(value, key)
            if not value: return default_value
        return value

    if isinstance(target, dict):
        return target.get(property_or_keys, default_value)
    return getattr(target, property_or_keys, default_value)


def __parse_tools(tools: list[ToolParam]):
    return [
        {
            "type": "function",
            "function": {
                "name": __prop(tool, "name"),
                "description": __prop(tool, "description"),
                "inputSchema": __prop(tool, "input_schema"),
            },
        }
        for tool in tools
    ]

def __params_parser(params: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    return {
        key: __parse_tools(value) if key == "tools" else value
        for key, value in params.items()
        if key in PARAMS_TO_CAPTURE
    }


def parse_message(message: MessageParam):
    role = __prop(message, "role", "system")
    content = __prop(message, "content", message)

    print(role, content)

    if isinstance(content, str):
        yield {"role": role, "content": content}
    elif isinstance(content, list):
        for item in content:
            if __prop(item, "type") == "text":
                yield {"content": __prop(item, "text"), "role": role}
            elif __prop(item, "type") == "tool_use":
                yield {
                    "functionCall": {
                        "name": __prop(item, "name"),
                        "arguments": __prop(item, "input"),
                    },
                    "toolCallId": __prop(item, "id"),
                }
            elif __prop(item, "type") == "tool_result":
                yield {
                    "role": "tool",
                    "tool_call_id": __prop(item, "tool_use_id"),
                    "content": __prop(item, "content"),
                }

    else:
        error = f"Invalid 'content' type for message: {message}"
        raise ValueError(error)


def __input_parser(kwargs: t.Dict):
    inputs = []

    if kwargs.get("system"):
        system = kwargs.get("system")
        if isinstance(system, str):
            inputs.append({"role": "system", "content": kwargs["system"]})
        elif isinstance(system, list):
            for item in kwargs["system"]:
                if __prop(item, "type") == "text":
                    inputs.append({"role": "system", "content": __prop(item, "text")})

    for message in kwargs.get("messages", []):
        inputs.extend(parse_message(message))

    return {"input": inputs, "name": kwargs.get("model")}

def __output_parser(output: t.Any, stream: bool = False):
    return {
        "name": __prop(output, "model"),
        "tokensUsage": {
            "completion": __prop(output, ["usage", "output_tokens"]),
            "prompt": __prop(output, ["usage", "input_tokens"]),
        },
        "output": [
            (
                {"content": __prop(content, "text"), "role": __prop(output, "role")}
                if __prop(content, "type") == "text"
                else {
                    "functionCall": {
                        "name": __prop(content, "name"),
                        "arguments": __prop(content, "input"),
                    },
                    "toolCallId": __prop(content, "id"),
                }
            )
            for content in __prop(output, "content", output)
        ],
    }


class Stream:

    def __init__(self, stream: MessageStream, handler: "StreamHandler"):
        self.__stream = stream
        self.__handler = handler

        self.__messages = []

        # Method wrappers
        self._iterator = self.__iterator__()
        self.text_stream = self.__stream_text__()

    def __getattr__(self, name):
        return getattr(self.__stream, name)

    def __iter__(self):
        return self.__iterator__()

    def __iterator__(self):
        for event in self.__stream.__stream__():
            if event.type == "message_start":
                self.__messages.append(
                    {
                        "role": event.message.role,
                        "model": event.message.model,
                        "usage": {
                            "input": event.message.usage.input_tokens,
                            "output": event.message.usage.output_tokens,
                        },
                        "content": [],
                    }
                )
            if event.type == "message_delta":
                if len(self.__messages) >= 1:
                    message = self.__messages[-1]
                    message["usage"]["output"] = event.usage.output_tokens

            if event.type == "message_stop":
                # print("\n\n ** ", list(__parse_message_content(event.message)))
                pass

            if event.type == "content_block_start":
                if len(self.__messages) >= 1:
                    message = self.__messages[-1]

                    if event.content_block.type == "text":
                        message["content"].insert(
                            event.index,
                            {
                                "type": event.content_block.type,
                                "content": event.content_block.text,
                            },
                        )
                    else:
                        message["content"].insert(
                            event.index,
                            {
                                "functionCall": {
                                    "name": event.content_block.name,
                                    "arguments": event.content_block.input,
                                },
                                "toolCallId": event.content_block.id,
                            },
                        )

            if event.type == "content_block_delta":
                if len(self.__messages) >= 1:
                    message = self.__messages[-1]
                    event_content = message["content"][event.index]

                    if event.delta.type == "text_delta":
                        event_content["content"] += event.delta.text
                    # else:
                    #     functionCall = event_content.get("functionCall")
                    #     event_content.update(
                    #         {
                    #             "functionCall": {
                    #                 "name": functionCall["name"],
                    #                 "arguments": (
                    #                     functionCall["arguments"]
                    #                     + event.delta.partial_json
                    #                 ),
                    #             }
                    #         }
                    #     )

            if event.type == "content_block_stop":
                if hasattr(event, "content_block") and len(self.__messages) >= 1:
                    message = self.__messages[-1]
                    event_content: dict = message["content"][event.index]

                    if event.content_block.type == "text":
                        event_content["content"] = event.content_block.text
                    elif event.content_block.type == "tool_use":
                        event_content.update(
                            {
                                "functionCall": {
                                    "name": event.content_block.name,
                                    "arguments": event.content_block.input,
                                },
                                "toolCallId": event.content_block.id,
                            }
                        )
                    else:
                        raise Exception("Invalid `content_block` type")

            yield event

        output = []

        for message in self.__messages:
            for item in message.get("content"):
                content = item.get("content")
                if isinstance(content, str):
                    output.append({"role": message["role"], "content": content})
                elif isinstance(content, list):
                    for sub_item in content:
                        output.append(sub_item)
                else:
                    output.append(item)

        track_event(
            output=output,
            event_name="end",
            run_type=self.__handler.__type__,
            run_id=self.__handler.__run_id__,
            name=self.__handler.__name__,
            token_usage={
                "completion": sum(
                    [message["usage"]["output"] for message in self.__messages]
                ),
                "prompt": sum(
                    [message["usage"]["input"] for message in self.__messages]
                ),
            },
        )

    def __stream_text__(self) -> t.Iterator[str]:
        for chunk in self.__iterator__():
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text


class AsyncStream:

    def __init__(self, stream: AsyncMessageStream, handler: "AsyncStreamHandler"):
        self.__stream = stream
        self.__handler = handler

        self.__messages = []

        # Method wrappers
        self._iterator = self.__iterator__()
        self.text_stream = self.__stream_text__()

    def __getattr__(self, name):
        return getattr(self.__stream, name)

    def __aiter__(self):
        return self.__iterator__()

    async def __iterator__(self):
        async for event in self.__stream.__stream__():
            if event.type == "message_start":
                self.__messages.append(
                    {
                        "role": event.message.role,
                        "model": event.message.model,
                        "usage": {
                            "input": event.message.usage.input_tokens,
                            "output": event.message.usage.output_tokens,
                        },
                        "content": [],
                    }
                )
            if event.type == "message_delta":
                if len(self.__messages) >= 1:
                    message = self.__messages[-1]
                    message["usage"]["output"] = event.usage.output_tokens

            if event.type == "message_stop":
                # print("\n\n ** ", list(__parse_message_content(event.message)))
                pass

            if event.type == "content_block_start":
                if len(self.__messages) >= 1:
                    message = self.__messages[-1]

                    if event.content_block.type == "text":
                        message["content"].insert(
                            event.index,
                            {
                                "type": event.content_block.type,
                                "content": event.content_block.text,
                            },
                        )
                    else:
                        message["content"].insert(
                            event.index,
                            {
                                "functionCall": {
                                    "name": event.content_block.name,
                                    "arguments": event.content_block.input,
                                },
                                "toolCallId": event.content_block.id,
                            },
                        )

            if event.type == "content_block_delta":
                if len(self.__messages) >= 1:
                    message = self.__messages[-1]
                    event_content = message["content"][event.index]

                    if event.delta.type == "text_delta":
                        event_content["content"] += event.delta.text
                    # else:
                    #     functionCall = event_content.get("functionCall")
                    #     event_content.update(
                    #         {
                    #             "functionCall": {
                    #                 "name": functionCall["name"],
                    #                 "arguments": (
                    #                     functionCall["arguments"]
                    #                     + event.delta.partial_json
                    #                 ),
                    #             }
                    #         }
                    #     )

            if event.type == "content_block_stop":
                if hasattr(event, "content_block") and len(self.__messages) >= 1:
                    message = self.__messages[-1]
                    event_content: dict = message["content"][event.index]

                    if event.content_block.type == "text":
                        event_content["content"] = event.content_block.text
                    elif event.content_block.type == "tool_use":
                        event_content.update(
                            {
                                "functionCall": {
                                    "name": event.content_block.name,
                                    "arguments": event.content_block.input,
                                },
                                "toolCallId": event.content_block.id,
                            }
                        )
                    else:
                        raise Exception("Invalid `content_block` type")

            yield event

        output = []

        for message in self.__messages:
            for item in message.get("content"):
                content = item.get("content")
                if isinstance(content, str):
                    output.append({"role": message["role"], "content": content})
                elif isinstance(content, list):
                    for sub_item in content:
                        output.append(sub_item)
                else:
                    output.append(item)

        track_event(
            output=output,
            event_name="end",
            run_type=self.__handler.__type__,
            run_id=self.__handler.__run_id__,
            name=self.__handler.__name__,
            token_usage={
                "completion": sum(
                    [message["usage"]["output"] for message in self.__messages]
                ),
                "prompt": sum(
                    [message["usage"]["input"] for message in self.__messages]
                ),
            },
        )

    async def __stream_text__(self) -> t.AsyncIterator[str]:
        async for chunk in self.__iterator__():
            if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                yield chunk.delta.text


class StreamHandler:

    __stream_manager: MessageStreamManager

    def __init__(
        self, method: t.Callable, run_id: str, name: str, type: str, *args, **kwargs
    ):
        self.__method = method
        self.__args = args
        self.__kwargs = kwargs

        self.__run_id__ = run_id
        self.__name__ = name
        self.__type__ = type

        self.__stream_manager = self.__method(*self.__args, **self.__kwargs)

    def __enter__(self):
        stream = self.__stream_manager.__enter__()
        return Stream(stream, self)

    def __exit__(self, *_):
        self.__stream_manager.__exit__(None, None, None)

    def __getattr__(self, name):
        return getattr(self.__stream_manager, name)

    def __iter__(self):
        stream = Stream(self.__stream_manager, self)
        return stream.__iterator__()


class AsyncStreamHandler:

    __stream_manager: AsyncMessageStreamManager

    def __init__(
        self, method: t.Callable, run_id: str, name: str, type: str, *args, **kwargs
    ):
        self.__method = method
        self.__args = args
        self.__kwargs = kwargs

        self.__run_id__ = run_id
        self.__name__ = name
        self.__type__ = type

        self.__stream_manager = self.__method(*self.__args, **self.__kwargs)

    def __await__(self):

        async def _():
            if iscoroutine(self.__stream_manager):
                self.__stream_manager = await self.__stream_manager
            return self

        return _().__await__()

    async def __aenter__(self):
        if iscoroutine(self.__stream_manager):
            self.__stream_manager = await self.__stream_manager

        stream = await self.__stream_manager.__aenter__()
        return AsyncStream(stream, self)

    async def __aexit__(self, *_):
        await self.__stream_manager.__aexit__(None, None, None)

    def __getattr__(self, name):
        return getattr(self.__stream_manager, name)

    def __aiter__(self):
        stream = AsyncStream(self.__stream_manager, self)
        return stream.__iterator__()


def __metadata_parser(metadata):
    return {x: metadata[x] for x in metadata if x in ["user_id"]}


def __wrap_sync(
    method: t.Callable,
    type: t.Optional[str] = None,
    user_id: t.Optional[str] = None,
    user_props: t.Optional[dict] = None,
    tags: t.Optional[dict] = None,
    name: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
    input_parser=__input_parser,
    output_parser=__output_parser,
    params_parser=__params_parser,
    stream_handler=StreamHandler,
    metadata_parser=__metadata_parser,
    contextify_stream: t.Optional[t.Callable] = None,
    *args,
    **kwargs,
):
    output = None

    parent_run_id = kwargs.pop("parent", None)
    run = run_manager.start_run(run_id, parent_run_id)

    with run_context(run.id):
        try:
            try:
                params = params_parser(kwargs)
                metadata = kwargs.get("metadata")
                parsed_input = input_parser(kwargs)

                if metadata:
                    kwargs["metadata"] = metadata_parser(metadata)

                track_event(
                    type,
                    "start",
                    run_id=run.id,
                    parent_run_id=parent_run_id,
                    input=parsed_input["input"],
                    name=name or parsed_input["name"],
                    user_id=(kwargs.pop("user_id", None) or user_ctx.get() or user_id),
                    user_props=(
                        kwargs.pop("user_props", None)
                        or user_props
                        or user_props_ctx.get()
                    ),
                    params=params,
                    metadata=metadata,
                    tags=(kwargs.pop("tags", None) or tags or tags_ctx.get()),
                    template_id=(
                        kwargs.get("extra_headers", {}).get("Template-Id", None)
                    ),
                    is_openai=False,
                )
            except Exception as e:
                return logging.exception(e)

            if contextify_stream or kwargs.get("stream") == True:
                return stream_handler(
                    method, run.id, name or parsed_input["name"], type, *args, **kwargs
                )

            try:
                output = method(*args, **kwargs)
            except Exception as e:
                track_event(
                    type,
                    "error",
                    run.id,
                    error={"message": str(e), "stack": traceback.format_exc()},
                )
                raise e from None

            try:
                parsed_output = output_parser(output, kwargs.get("stream", False))

                print(parsed_input, parsed_output, output)

                track_event(
                    type,
                    "end",
                    run.id,
                    # In case need to compute tokens usage server side
                    name=name or parsed_input["name"],
                    output=parsed_output["output"],
                    token_usage=parsed_output["tokensUsage"],
                )
                return output
            except Exception as e:
                logger.exception(e)(e)
            finally:
                return output
        finally:
            run_manager.end_run(run.id)


async def __wrap_async(
    method: t.Callable,
    type: t.Optional[str] = None,
    user_id: t.Optional[str] = None,
    user_props: t.Optional[dict] = None,
    tags: t.Optional[dict] = None,
    name: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
    input_parser=__input_parser,
    output_parser=__output_parser,
    params_parser=__params_parser,
    stream_handler=AsyncStreamHandler,
    metadata_parser=__metadata_parser,
    contextify_stream: t.Optional[bool] = False,
    *args,
    **kwargs,
):
    output = None

    parent_run_id = kwargs.pop("parent", None)
    run = run_manager.start_run(run_id, parent_run_id)

    with run_context(run.id):
        try:
            try:
                params = params_parser(kwargs)
                metadata = kwargs.get("metadata")
                parsed_input = input_parser(kwargs)

                if metadata:
                    kwargs["metadata"] = metadata_parser(metadata)

                track_event(
                    type,
                    "start",
                    run_id=run.id,
                    parent_run_id=parent_run_id,
                    input=parsed_input["input"],
                    name=name or parsed_input["name"],
                    user_id=(kwargs.pop("user_id", None) or user_ctx.get() or user_id),
                    user_props=(
                        kwargs.pop("user_props", None)
                        or user_props
                        or user_props_ctx.get()
                    ),
                    params=params,
                    metadata=metadata,
                    tags=(kwargs.pop("tags", None) or tags or tags_ctx.get()),
                    template_id=(
                        kwargs.get("extra_headers", {}).get("Template-Id", None)
                    ),
                    is_openai=False,
                )
            except Exception as e:
                return logging.exception(e)

            if contextify_stream or kwargs.get("stream") == True:
                return await stream_handler(
                    method, run.id, name or parsed_input["name"], type, *args, **kwargs
                )

            try:
                output = await method(*args, **kwargs)
            except Exception as e:
                track_event(
                    type,
                    "error",
                    run.id,
                    error={"message": str(e), "stack": traceback.format_exc()},
                )
                raise e from None

            try:
                parsed_output = output_parser(output, kwargs.get("stream", False))

                track_event(
                    type,
                    "end",
                    run.id,
                    # In case need to compute tokens usage server side
                    name=name or parsed_input["name"],
                    output=parsed_output["output"],
                    token_usage=parsed_output["tokensUsage"],
                )
                return output
            except Exception as e:
                logger.exception(e)(e)
            finally:
                return output
        finally:
            run_manager.end_run(run.id)


if t.TYPE_CHECKING:
    ClientType = t.TypeVar("ClientType")


def monitor(client: "ClientType") -> "ClientType":
    if isinstance(client, Anthropic):
        client.messages.create = partial(
            __wrap_sync, client.messages.create, type="llm"
        )
        client.messages.stream = partial(
            __wrap_sync,
            client.messages.stream,
            type="llm",
            stream_handler=StreamHandler,
            contextify_stream=True,
        )
    elif isinstance(client, AsyncAnthropic):
        client.messages.create = partial(
            __wrap_async, client.messages.create, type="llm"
        )
        client.messages.stream = partial(
            __wrap_sync,
            client.messages.stream,
            type="llm",
            stream_handler=AsyncStreamHandler,
            contextify_stream=True,
        )
    else:
        raise Exception("Invalid argument. Expected instance of Anthropic Client")
    return client


def agent(name=None, user_id=None, user_props=None, tags=None):
    def decorator(fn):
        return partial(
            __wrap_sync,
            fn, "agent",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            tags=tags
        )

    return decorator


def chain(name=None, user_id=None, user_props=None, tags=None):
    def decorator(fn):
        return partial(
            __wrap_sync,
            fn, "chain",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            tags=tags
        )

    return decorator


def tool(name=None, user_id=None, user_props=None, tags=None):
    def decorator(fn):
        return partial(
            __wrap_sync,
            fn, "tool",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            tags=tags
        )

    return decorator

