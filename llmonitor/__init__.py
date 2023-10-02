import asyncio, uuid, os, warnings
import traceback
from contextvars import ContextVar
from datetime import datetime
from dotenv import load_dotenv

from .parsers import (
    default_input_parser,
    default_output_parser,
)
from .openai_utils import OpenAIUtils
from .event_queue import EventQueue
from .consumer import Consumer
from .users import user_ctx, user_props_ctx


load_dotenv()
APP_ID = os.environ.get("LLMONITOR_APP_ID")
VERBOSE = os.environ.get("LLMONITOR_VERBOSE")
API_URL = os.environ.get("LLMONITOR_API_URL")


run_ctx = ContextVar("run_ids", default=None)


queue = EventQueue()
consumer = Consumer(queue, api_url=API_URL)

consumer.start()


def track_event(
    event_type,
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
):
    if not APP_ID:
        return warnings.warn("LLMONITOR_APP_ID is not set, not sending events")

    event = {
        "event": event_name,
        "type": event_type,
        "app": APP_ID,
        "name": name,
        "userId": user_id,
        "userProps": user_props,
        "tags": tags,
        "runId": str(run_id),
        "parentRunId": str(parent_run_id) if parent_run_id else None,
        "timestamp": str(datetime.utcnow()),
        "input": input,
        "output": output,
        "error": error,
        "runtime": "llmonitor-py",
        "tokensUsage": token_usage,
    }

    if VERBOSE:
        print("llmonitor_add_event", event)

    queue.append(event)


def wrap(
    fn,
    type=None,
    name=None,
    user_id=None,
    user_props=None,
    parent_run_id=None,
    tags=None,
    input_parser=default_input_parser,
    output_parser=default_output_parser,
):
    def sync_wrapper(*args, **kwargs):
        nonlocal parent_run_id
        try:
            parent_run_id = parent_run_id or run_ctx.get()
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
                user_props=user_props_ctx.get() or user_props or kwargs.pop("user_props", None),
                tags=tags,
            )
            output = fn(*args, **kwargs)
            parsed_output = output_parser(output)

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
            print(e)
            track_event(
                type,
                "error",
                run_id,
                error={"message": str(e), "stack": traceback.format_exc()},
            )
        finally:
            run_ctx.reset(token)

    async def async_wrapper(*args, **kwargs):
        try:
            parent_run_id = parent_run_id or run_ctx.get()
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
                user_props=user_props_ctx.get() or user_props or kwargs.pop("user_props", None),
                tags=tags,
            )
            output = await fn(*args, **kwargs)
            parsed_output = output_parser(output)

            track_event(
                type,
                "end",
                run_id,
                parent_run_id,
                # Need name in case need to compute tokens usage server side
                name=name or parsed_input["name"],
                output=parsed_output["output"],
                token_usage=parsed_output["tokensUsage"],
            )
            return output
        except Exception as e:
            print(e)
            track_event(
                type,
                "error",
                run_id,
                parent_run_id,
                error={"message": str(e), "stack": traceback.format_exc()},
            )
        finally:
            run_ctx.reset(token)

    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper


def monitor(object: OpenAIUtils):
    if object.__name__ == "openai":
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

    else:
        warnings.warn("You cannot monitor this object")


def agent(name=None, user_id=None, user_props=None, tags=None, parent_run_id=None):
    def decorator(fn):
        return wrap(
            fn,
            "agent",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            parent_run_id=parent_run_id,
            tags=tags,
            input_parser=default_input_parser,
        )

    return decorator


def tool(name=None, user_id=None, user_props=None, tags=None, parent_run_id=None):
    def decorator(fn):
        return wrap(
            fn,
            "tool",
            name=name or fn.__name__,
            user_id=user_id,
            user_props=user_props,
            parent_run_id=parent_run_id,
            tags=tags,
            input_parser=default_input_parser,
        )

    return decorator
