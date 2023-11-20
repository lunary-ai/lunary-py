import asyncio, uuid, os, warnings
from pkg_resources import parse_version
from importlib.metadata import version, PackageNotFoundError
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone


from .parsers import (
    default_input_parser,
    default_output_parser,
)
from .openai_utils import OpenAIUtils
from .event_queue import EventQueue
from .consumer import Consumer
from .users import user_ctx, user_props_ctx
from .tags import tags_ctx, tags  # DO NOT REMOVE `tags` import

run_ctx = ContextVar("run_ids", default=None)

queue = EventQueue()
consumer = Consumer(queue)

consumer.start()


def track_event(
    run_type,
    event_name,
    run_id,
    parent_run_id=None,
    name=None,
    input=None,
    output=None,
    error=None,
    token_usage=None,
    user_id=None,
    user_props=None,
    tags=None,
    extra=None,
    metadata=None,
    app_id=None,
):
    # Load here in case load_dotenv done after
    APP_ID = app_id or os.environ.get("LLMONITOR_APP_ID")
    VERBOSE = os.environ.get("LLMONITOR_VERBOSE")

    if not APP_ID:
        return warnings.warn("LLMONITOR_APP_ID is not set, not sending events")

    if parent_run_id:
        parent_run_id = str(parent_run_id)

    if run_ctx.get() is not None and str(run_id) != str(run_ctx.get()):
        parent_run_id = str(run_ctx.get())

    event = {
        "event": event_name,
        "type": run_type,
        "app": APP_ID,
        "name": name,
        "userId": user_id,
        "userProps": user_props,
        "tags": tags or tags_ctx.get(),
        "runId": str(run_id),
        "parentRunId": parent_run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input,
        "output": output,
        "error": error,
        "extra": extra,
        "runtime": "llmonitor-py",
        "tokensUsage": token_usage,
        "metadata": metadata,
    }

    if VERBOSE:
        print("llmonitor_add_event", event)

    queue.append(event)


def handle_internal_error(e):
    print("[LLMonitor] Error: ", e)


def wrap(
    fn,
    type=None,
    name=None,
    user_id=None,
    user_props=None,
    tags=None,
    input_parser=default_input_parser,
    output_parser=default_output_parser,
):
    def sync_wrapper(*args, **kwargs):
        output = None
        try:
            parent_run_id = run_ctx.get()
            run_id = uuid.uuid4()
            token = run_ctx.set(run_id)
            parsed_input = input_parser(*args, **kwargs)

            track_event(
                type,
                "start",
                run_id,
                parent_run_id,
                input=parsed_input["input"],
                name=name or parsed_input["name"],
                user_id=kwargs.pop("user_id", None) or user_ctx.get() or user_id,
                user_props=kwargs.pop("user_props", None)
                or user_props
                or user_props_ctx.get(),
                tags=kwargs.pop("tags", None) or tags or tags_ctx.get(),
                extra=parsed_input.get("extra", None),
            )
        except Exception as e:
            handle_internal_error(e)

        if kwargs.get("stream") == True:
            stream = fn(*args, **kwargs)
            original_iterator = iter(stream)

            def wrapped_iterator():
                choices = []
                tokens = 0

                for chunk in original_iterator:
                    tokens += 1
                    choice = chunk.choices[0]
                    index = choice.index

                    content = choice.delta.content
                    role = choice.delta.role
                    function_call = choice.delta.function_call

                    if len(choices) <= index:
                        choices.append(
                            {
                                "message": {
                                    "role": role,
                                    "content": content,
                                    "function_call": {},
                                }
                            }
                        )

                    if content:
                        choices[index]["message"]["content"] += content

                    if role:
                        choices[index]["message"]["role"] = role

                    if hasattr(function_call, "name"):
                        choices[index]["message"]["function_call"][
                            "name"
                        ] = function_call.name

                    if hasattr(function_call, "arguments"):
                        choices[index]["message"]["function_call"].setdefault(
                            "arguments", ""
                        )
                        choices[index]["message"]["function_call"][
                            "arguments"
                        ] += function_call.arguments

                output = OpenAIUtils.parse_message(choices[0]["message"])

                # TODO: add error handling
                track_event(
                    type,
                    "end",
                    run_id,
                    name=name or parsed_input["name"],
                    output=output,
                    token_usage=None,
                )

                yield chunk

            return (item for item in wrapped_iterator())

        try:
            output = fn(*args, **kwargs)
        except Exception as e:
            track_event(
                type,
                "error",
                run_id,
                error={"message": str(e), "stack": traceback.format_exc()},
            )

            # rethrow error
            raise e

        try:
            parsed_output = output_parser(output, kwargs.get("stream", False))

            track_event(
                type,
                "end",
                run_id,
                # Need name in case need to compute tokens usage server side,
                name=name or parsed_input["name"],
                output=parsed_output["output"],
                token_usage=parsed_output["tokensUsage"],
            )
            return output
        except Exception as e:
            handle_internal_error(e)

        run_ctx.reset(token)
        return output

    async def async_wrapper(*args, **kwargs):
        output = None
        try:
            parent_run_id = run_ctx.get()
            run_id = uuid.uuid4()
            token = run_ctx.set(run_id)
            parsed_input = input_parser(*args, **kwargs)

            track_event(
                type,
                "start",
                run_id,
                parent_run_id,
                input=parsed_input["input"],
                name=name or parsed_input["name"],
                user_id=user_ctx.get() or user_id or kwargs.pop("user_id", None),
                user_props=user_props_ctx.get()
                or user_props
                or kwargs.pop("user_props", None),
                tags=tags,
                extra=parsed_input["extra"],
            )
        except Exception as e:
            handle_internal_error(e)

        try:
            output = await fn(*args, **kwargs)
        except Exception as e:
            track_event(
                type,
                "error",
                run_id,
                error={"message": str(e), "stack": traceback.format_exc()},
            )

            # rethrow error
            raise e

        try:
            parsed_output = output_parser(output, kwargs.get("stream", None))

            track_event(
                type,
                "end",
                run_id,
                # Need name in case need to compute tokens usage server side,
                name=name or parsed_input["name"],
                output=parsed_output["output"],
                token_usage=parsed_output["tokensUsage"],
            )
            return output
        except Exception as e:
            handle_internal_error(e)

        run_ctx.reset(token)
        return output

    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper


def async_wrap(
    fn,
    type=None,
    name=None,
    user_id=None,
    user_props=None,
    tags=None,
    input_parser=default_input_parser,
    output_parser=default_output_parser,
):
    async def async_wrapper(*args, **kwargs):
        output = None
        try:
            parent_run_id = run_ctx.get()
            run_id = uuid.uuid4()
            token = run_ctx.set(run_id)
            parsed_input = input_parser(*args, **kwargs)

            track_event(
                type,
                "start",
                run_id,
                parent_run_id,
                input=parsed_input["input"],
                name=name or parsed_input["name"],
                user_id=user_ctx.get() or user_id or kwargs.pop("user_id", None),
                user_props=user_props_ctx.get()
                or user_props
                or kwargs.pop("user_props", None),
                tags=kwargs.pop("tags", None) or tags or tags_ctx.get(),
                extra=parsed_input["extra"],
            )
        except Exception as e:
            handle_internal_error(e)

        try:
            output = await fn(*args, **kwargs)
        except Exception as e:
            track_event(
                type,
                "error",
                run_id,
                error={"message": str(e), "stack": traceback.format_exc()},
            )

            # rethrow error
            raise e

        try:
            parsed_output = output_parser(output, kwargs.get("stream", None))

            track_event(
                type,
                "end",
                run_id,
                # Need name in case need to compute tokens usage server side,
                name=name or parsed_input["name"],
                output=parsed_output["output"],
                token_usage=parsed_output["tokensUsage"],
            )
            return output
        except Exception as e:
            handle_internal_error(e)

        run_ctx.reset(token)
        return output

    return async_wrapper


def monitor(object):
    try:
        openai_version = parse_version(version("openai"))
        name = getattr(object, "__name__", getattr(type(object), "__name__", None))

        if openai_version >= parse_version("1.0.0") and openai_version < parse_version(
            "2.0.0"
        ):
            name = getattr(type(object), "__name__", None)
            if name == "openai" or name == "OpenAI" or name == "AzureOpenAI":
                try:
                    object.chat.completions.create = wrap(
                        object.chat.completions.create,
                        "llm",
                        input_parser=OpenAIUtils.parse_input,
                        output_parser=OpenAIUtils.parse_output,
                    )
                except Exception as e:
                    print(
                        "[LLMonitor] Please use `monitor(openai)` or `monitor(client)` after setting the OpenAI api key"
                    )

            elif name == "AsyncOpenAI":
                object.chat.completions.create = async_wrap(
                    object.chat.completions.create,
                    "llm",
                    input_parser=OpenAIUtils.parse_input,
                    output_parser=OpenAIUtils.parse_output,
                )
            else:
                print(
                    "[LLMonitor] Uknonwn OpenAI client. You can only use `monitor(openai)` or `monitor(client)`"
                )
        elif openai_version < parse_version("1.0.0"):
            object.ChatCompletion.create = wrap(
                object.ChatCompletion.create,
                "llm",
                input_parser=OpenAIUtils.parse_input,
                output_parser=OpenAIUtils.parse_output,
            )

            object.ChatCompletion.acreate = wrap(
                object.ChatCompletion.acreate,
                "llm",
                input_parser=OpenAIUtils.parse_input,
                output_parser=OpenAIUtils.parse_output,
            )

    except PackageNotFoundError:
        print("[LLMonitor] The `openai` package is not installed")


def agent(name=None, user_id=None, user_props=None, tags=None):
    def decorator(fn):
        return wrap(
            fn,
            "agent",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            tags=tags,
            input_parser=default_input_parser,
        )

    return decorator


def chain(name=None, user_id=None, user_props=None, tags=None):
    def decorator(fn):
        return wrap(
            fn,
            "chain",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            tags=tags,
            input_parser=default_input_parser,
        )

    return decorator


def tool(name=None, user_id=None, user_props=None, tags=None):
    def decorator(fn):
        return wrap(
            fn,
            "tool",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            tags=tags,
            input_parser=default_input_parser,
        )

    return decorator
