import typing as t
from functools import partial
from inspect import iscoroutine
from contextlib import AsyncContextDecorator, ContextDecorator

from . import track_event, run_context, run_manager, logging, logger, user_props_ctx, user_ctx, traceback, tags_ctx, filter_params

try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message
    from anthropic.lib.streaming import MessageStreamManager, AsyncMessageStreamManager
except ImportError:
    raise ImportError("Anthrophic SDK not installed!") from None


class sync_context_wrapper(ContextDecorator):

    def __init__(self, stream):
        self.__stream = stream

    def __enter__(self):
        return self.__stream

    def __exit__(self, *_):
        return


class async_context_wrapper(AsyncContextDecorator):

    def __init__(self, stream):
        self.__stream = stream

    async def __aenter__(self):
        return self.__stream

    async def __aexit__(self, *_):
        return


def __input_parser(kwargs: t.Dict):
    return {"input": kwargs.get("messages"), "name": kwargs.get("model")}


def __output_parser(output: t.Union[Message], stream: bool = False):
    if isinstance(output, Message):
        return {
            "name":
            output.model,
            "tokensUsage":
            output.usage,
            "output": [{
                "content": content.text,
                "role": output.role
            } for content in output.content],
        }
    else:
        return {
            "name": None,
            "tokenUsage": None,
            "output": getattr(output, "content", output)
        }


def __stream_handler(method, run_id, name, type, *args, **kwargs):
    messages = []
    original_stream = None
    stream = method(*args, **kwargs)

    if isinstance(stream, MessageStreamManager):
        original_stream = stream
        stream = original_stream.__enter__()

    for event in stream:
        if event.type == "message_start":
            messages.append({
                "role": event.message.role,
                "model": event.message.model
            })
        if event.type == "message_delta":
            if len(messages) >= 1:
                message = messages[-1]
                message["usage"] = {"tokens": event.usage.output_tokens}

        if event.type == "message_stop": pass
        if event.type == "content_block_start":
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = event.content_block.text

        if event.type == "content_block_delta":
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = message.get("output",
                                                "") + event.delta.text

        if event.type == "content_block_stop":
            pass

        yield event

    if original_stream:
        original_stream.__exit__(None, None, None)

    track_event(
        type,
        "end",
        run_id,
        name=name,
        output=[{
            "role": message["role"],
            "content": message["output"]
        } for message in messages],
        token_usage=sum([message["usage"]["tokens"] for message in messages]),
    )


async def __async_stream_handler(method, run_id, name, type, *args, **kwargs):
    messages = []
    original_stream = None
    stream = method(*args, **kwargs)

    if iscoroutine(stream):
        stream = await stream

    if isinstance(stream, AsyncMessageStreamManager):
        original_stream = stream
        stream = await original_stream.__aenter__()

    async for event in stream:
        if event.type == "message_start":
            messages.append({
                "role": event.message.role,
                "model": event.message.model
            })
        if event.type == "message_delta":
            if len(messages) >= 1:
                message = messages[-1]
                message["usage"] = {"tokens": event.usage.output_tokens}

        if event.type == "message_stop": pass
        if event.type == "content_block_start":
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = event.content_block.text

        if event.type == "content_block_delta":
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = message.get("output",
                                                "") + event.delta.text

        if event.type == "content_block_stop":
            pass

        yield event

    if original_stream:
        await original_stream.__aexit__(None, None, None)

    track_event(
        type,
        "end",
        run_id,
        name=name,
        output=[{
            "role": message["role"],
            "content": message["output"]
        } for message in messages],
        token_usage=sum([message["usage"]["tokens"] for message in messages]),
    )


def __metadata_parser(metadata):
    return {x: metadata[x] for x in metadata if x in ["user_id"]}


def __wrap_sync(method: t.Callable,
                type: t.Optional[str] = None,
                user_id: t.Optional[str] = None,
                user_props: t.Optional[dict] = None,
                tags: t.Optional[dict] = None,
                name: t.Optional[str] = None,
                run_id: t.Optional[str] = None,
                input_parser=__input_parser,
                output_parser=__output_parser,
                stream_handler=__stream_handler,
                metadata_parser=__metadata_parser,
                contextify_stream: t.Optional[t.Callable] = None,
                *args,
                **kwargs):
    output = None

    parent_run_id = kwargs.pop("parent", None)
    run = run_manager.start_run(run_id, parent_run_id)

    with run_context(run.id):
        try:
            try:
                params = filter_params(kwargs)
                metadata = kwargs.get("metadata")
                parsed_input = input_parser(kwargs)

                if metadata:
                    kwargs["metadata"] = metadata_parser(metadata)

                track_event(type,
                            "start",
                            run_id=run.id,
                            parent_run_id=parent_run_id,
                            input=parsed_input["input"],
                            name=name or parsed_input["name"],
                            user_id=(kwargs.pop("user_id", None)
                                     or user_ctx.get() or user_id),
                            user_props=(kwargs.pop("user_props", None)
                                        or user_props or user_props_ctx.get()),
                            params=params,
                            metadata=metadata,
                            tags=(kwargs.pop("tags", None) or tags
                                  or tags_ctx.get()),
                            template_id=(kwargs.get("extra_headers", {}).get(
                                "Template-Id", None)),
                            is_openai=False)
            except Exception as e:
                logging.exception(e)

            if contextify_stream or kwargs.get("stream") == True:
                generator = stream_handler(method, run.id, name
                                           or parsed_input["name"], type,
                                           *args, **kwargs)
                if contextify_stream:
                    return contextify_stream(generator)
                else: return generator

            try:
                output = method(*args, **kwargs)
            except Exception as e:
                track_event(
                    type,
                    "error",
                    run.id,
                    error={
                        "message": str(e),
                        "stack": traceback.format_exc()
                    },
                )
                raise e from None

            try:
                parsed_output = output_parser(output,
                                              kwargs.get("stream", False))

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


async def __wrap_async(method: t.Callable,
                       type: t.Optional[str] = None,
                       user_id: t.Optional[str] = None,
                       user_props: t.Optional[dict] = None,
                       tags: t.Optional[dict] = None,
                       name: t.Optional[str] = None,
                       run_id: t.Optional[str] = None,
                       input_parser=__input_parser,
                       output_parser=__output_parser,
                       stream_handler=__async_stream_handler,
                       metadata_parser=__metadata_parser,
                       contextify_stream: t.Optional[bool] = False,
                       *args,
                       **kwargs):
    output = None

    parent_run_id = kwargs.pop("parent", None)
    run = run_manager.start_run(run_id, parent_run_id)

    with run_context(run.id):
        try:
            try:
                params = filter_params(kwargs)
                metadata = kwargs.get("metadata")
                parsed_input = input_parser(kwargs)

                if metadata:
                    kwargs["metadata"] = metadata_parser(metadata)

                track_event(type,
                            "start",
                            run_id=run.id,
                            parent_run_id=parent_run_id,
                            input=parsed_input["input"],
                            name=name or parsed_input["name"],
                            user_id=(kwargs.pop("user_id", None)
                                     or user_ctx.get() or user_id),
                            user_props=(kwargs.pop("user_props", None)
                                        or user_props or user_props_ctx.get()),
                            params=params,
                            metadata=metadata,
                            tags=(kwargs.pop("tags", None) or tags
                                  or tags_ctx.get()),
                            template_id=(kwargs.get("extra_headers", {}).get(
                                "Template-Id", None)),
                            is_openai=False)
            except Exception as e:
                logging.exception(e)

            if contextify_stream or kwargs.get("stream") == True:
                generator = stream_handler(method, run.id, name
                                           or parsed_input["name"], type,
                                           *args, **kwargs)
                if contextify_stream:
                    return contextify_stream(generator)
                else: return generator

            try:
                output = await method(*args, **kwargs)
            except Exception as e:
                track_event(
                    type,
                    "error",
                    run.id,
                    error={
                        "message": str(e),
                        "stack": traceback.format_exc()
                    },
                )
                raise e from None

            try:
                parsed_output = output_parser(output,
                                              kwargs.get("stream", False))

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
        client.messages.create = partial(__wrap_sync,
                                         client.messages.create,
                                         type="llm")
        client.messages.stream = partial(__wrap_sync,
                                         client.messages.stream,
                                         type="llm",
                                         contextify_stream=sync_context_wrapper)
    elif isinstance(client, AsyncAnthropic):
        client.messages.create = partial(__wrap_async,
                                         client.messages.create,
                                         type="llm")
        client.messages.stream = partial(__wrap_sync,
                                         client.messages.stream,
                                         type="llm",
                                         stream_handler=__async_stream_handler,
                                         contextify_stream=async_context_wrapper)
    else:
        raise Exception(
            "Invalid argument. Expected instance of Anthropic Client")
    return client
