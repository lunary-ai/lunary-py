import typing as t
from functools import partial
from . import track_event, run_context, run_manager, logging, logger, user_props_ctx, user_ctx, traceback, tags_ctx, filter_params

try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message
except ImportError:
    raise ImportError("Anthrophic SDK not installed!") from None


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
    stream = method(*args, **kwargs)

    for event in stream:
        if event.type == "message_start":
            # print(event.message.model)
            messages.append({
                "role": event.message.role,
                "model": event.message.model
            })
        if event.type == "message_delta":
            # print("*", event.usage.output_tokens)
            if len(messages) >= 1:
                message = messages[-1]
                message["usage"] = {"tokens": event.usage.output_tokens}

        if event.type == "message_stop": pass
        if event.type == "content_block_start":
            # print("* START")
            # print(event.content_block.text)
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = event.content_block.text

        if event.type == "content_block_delta":
            # print(event.delta.text, end="")
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = message.get("output",
                                                "") + event.delta.text

        if event.type == "content_block_stop":
            # print("* END")
            pass

        yield event

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
    stream = await method(*args, **kwargs)

    async for event in stream:
        if event.type == "message_start":
            # print(event.message.model)
            messages.append({
                "role": event.message.role,
                "model": event.message.model
            })
        if event.type == "message_delta":
            # print("*", event.usage.output_tokens)
            if len(messages) >= 1:
                message = messages[-1]
                message["usage"] = {"tokens": event.usage.output_tokens}

        if event.type == "message_stop": pass
        if event.type == "content_block_start":
            # print("* START")
            # print(event.content_block.text)
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = event.content_block.text

        if event.type == "content_block_delta":
            # print(event.delta.text, end="")
            if len(messages) >= 1:
                message = messages[-1]
                message["output"] = message.get("output",
                                                "") + event.delta.text

        if event.type == "content_block_stop":
            # print("* END")
            pass

        yield event

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

            if kwargs.get("stream") == True:
                return stream_handler(method, run.id, name
                                      or parsed_input["name"], type, *args,
                                      **kwargs)

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
                            is_openai=True)
            except Exception as e:
                logging.exception(e)

            if kwargs.get("stream") == True:
                return stream_handler(method, run.id, name
                                            or parsed_input["name"], type,
                                            *args, **kwargs)

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
        client.messages.create = partial(__wrap_sync, client.messages.create,
                                         "llm")
    elif isinstance(client, AsyncAnthropic):
        client.messages.create = partial(__wrap_async, client.messages.create,
                                         "llm")
    else:
        raise Exception(
            "Invalid argument. Expected instance of Anthropic Client")
    return client
