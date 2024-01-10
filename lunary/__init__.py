from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import set_span_in_context
from opentelemetry import trace
import uuid
import os
import warnings
import traceback
import logging
import copy
import json
import time
import chevron
from pkg_resources import parse_version
from importlib.metadata import version, PackageNotFoundError
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional


from .parsers import (
    default_input_parser,
    default_output_parser,
)
from .openai_utils import OpenAIUtils
from .event_queue import EventQueue
from .consumer import Consumer
# DO NOT REMOVE `identify` import
from .users import user_ctx, user_props_ctx, identify
from .tags import tags_ctx, tags  # DO NOT REMOVE `tags` import
from .thread import Thread

DEFAULT_API_URL = "https://app.lunary.ai"

logger = logging.getLogger(__name__)

run_ctx = ContextVar("run_ids", default=None)

queue = EventQueue()
consumer = Consumer(queue)

consumer.start()


provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("lunary")


def clean_nones(value):
    """
    Recursively remove all None values from dictionaries and lists, and returns
    the result as a new dictionary or list.
    """
    if isinstance(value, list):
        return [clean_nones(x) for x in value if x is not None]
    elif isinstance(value, dict):
        return {
            key: clean_nones(val)
            for key, val in value.items()
            if val is not None
        }
    else:
        return value


def track_event(
    run_type,
    event_name,
    run_id,
    parent_run_id=None,
    name=None,
    input=None,
    output=None,
    message=None,
    error=None,
    token_usage=None,
    user_id=None,
    user_props=None,
    tags=None,
    thread_tags=None,
    feedback=None,
    extra=None,
    template_id=None,
    metadata=None,
    app_id=None,
):
    # Load here in case load_dotenv done after
    APP_ID = (
        app_id or os.environ.get(
            "LUNARY_APP_ID") or os.environ.get("LLMONITOR_APP_ID")
    )
    VERBOSE = os.environ.get(
        "LUNARY_VERBOSE") or os.environ.get("LLMONITOR_VERBOSE")

    if not APP_ID:
        return warnings.warn("LUNARY_APP_ID is not set, not sending events")

    if parent_run_id:
        parent_run_id = str(parent_run_id)

    if run_ctx.get() is not None and str(run_id) != str(run_ctx.get()):
        parent_run_id = str(run_ctx.get())

    event = {
        "event": event_name,
        "type": run_type,
        "app": APP_ID,
        "name": name,
        "userId": user_id or user_ctx.get(),
        "userProps": user_props or user_props_ctx.get(),
        "tags": tags or tags_ctx.get(),
        "threadTags": thread_tags,
        "runId": str(run_id),
        "parentRunId": parent_run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "input": input,
        "output": output,
        "error": error,
        "extra": extra,
        "feedback": feedback,
        "runtime": "lunary-py",
        "tokensUsage": token_usage,
        "metadata": metadata,
        "templateId": template_id,
    }

    if VERBOSE:
        print("[Lunary] Add event:", json.dumps(clean_nones(event), indent=4))
        print("\n")

    queue.append(event)


def handle_internal_error(e):
    logging.info("[Lunary] Error: ", e)


def stream_handler(fn, run_id, name, type, *args, **kwargs):
    stream = fn(*args, **kwargs)

    choices = []
    tokens = 0

    for chunk in stream:
        tokens += 1
        choice = chunk.choices[0]
        index = choice.index

        content = choice.delta.content
        role = choice.delta.role
        function_call = choice.delta.function_call
        tool_calls = choice.delta.tool_calls

        if len(choices) <= index:
            choices.append(
                {
                    "message": {
                        "role": role,
                        "content": content or "",
                        "function_call": {},
                        "tool_calls": [],
                    }
                }
            )

        if content:
            choices[index]["message"]["content"] += content

        if role:
            choices[index]["message"]["role"] = role

        if hasattr(function_call, "name"):
            choices[index]["message"]["function_call"]["name"] = function_call.name

        if hasattr(function_call, "arguments"):
            choices[index]["message"]["function_call"].setdefault(
                "arguments", "")
            choices[index]["message"]["function_call"][
                "arguments"
            ] += function_call.arguments

        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                existing_call_index = next(
                    (
                        index
                        for (index, tc) in enumerate(
                            choices[index]["message"]["tool_calls"]
                        )
                        if tc.index == tool_call.index
                    ),
                    -1,
                )

            if existing_call_index == -1:
                choices[index]["message"]["tool_calls"].append(tool_call)

            else:
                existing_call = choices[index]["message"]["tool_calls"][
                    existing_call_index
                ]
                if hasattr(tool_call, "function") and hasattr(
                    tool_call.function, "arguments"
                ):
                    existing_call.function.arguments += tool_call.function.arguments

        yield chunk

    output = OpenAIUtils.parse_message(choices[0]["message"])
    track_event(
        type,
        "end",
        run_id,
        name=name,
        output=output,
        token_usage={"completion": tokens, "prompt": None},
    )
    return

async def async_stream_handler(fn, run_id, name, type, *args, **kwargs):
    stream = await fn(*args, **kwargs)

    choices = []
    tokens = 0

    async for chunk in stream:
        tokens += 1
        choice = chunk.choices[0]
        index = choice.index

        content = choice.delta.content
        role = choice.delta.role
        function_call = choice.delta.function_call
        tool_calls = choice.delta.tool_calls

        if len(choices) <= index:
            choices.append(
                {
                    "message": {
                        "role": role,
                        "content": content or "",
                        "function_call": {},
                        "tool_calls": [],
                    }
                }
            )

        if content:
            choices[index]["message"]["content"] += content

        if role:
            choices[index]["message"]["role"] = role

        if hasattr(function_call, "name"):
            choices[index]["message"]["function_call"]["name"] = function_call.name

        if hasattr(function_call, "arguments"):
            choices[index]["message"]["function_call"].setdefault(
                "arguments", "")
            choices[index]["message"]["function_call"][
                "arguments"
            ] += function_call.arguments

        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                existing_call_index = next(
                    (
                        index
                        for (index, tc) in enumerate(
                        choices[index]["message"]["tool_calls"]
                    )
                        if tc.index == tool_call.index
                    ),
                    -1,
                )

            if existing_call_index == -1:
                choices[index]["message"]["tool_calls"].append(tool_call)

            else:
                existing_call = choices[index]["message"]["tool_calls"][
                    existing_call_index
                ]
                if hasattr(tool_call, "function") and hasattr(
                        tool_call.function, "arguments"
                ):
                    existing_call.function.arguments += tool_call.function.arguments

        yield chunk

    output = OpenAIUtils.parse_message(choices[0]["message"])
    track_event(
        type,
        "end",
        run_id,
        name=name,
        output=output,
        token_usage={"completion": tokens, "prompt": None},
    )
    return

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
        with tracer.start_as_current_span(uuid.uuid4()) as run:
            output = None
            try:
                run_id = trace.get_current_span().context.span_id
                parent_run_id = kwargs.pop("parent", None) or getattr(
                    trace.get_current_span().parent, "span_id", None
                )
                parsed_input = input_parser(*args, **kwargs)

                track_event(
                    type,
                    "start",
                    run_id,
                    parent_run_id,
                    input=parsed_input["input"],
                    name=name or parsed_input["name"],
                    user_id=kwargs.pop(
                        "user_id", None) or user_ctx.get() or user_id,
                    user_props=kwargs.pop("user_props", None)
                    or user_props
                    or user_props_ctx.get(),
                    tags=kwargs.pop("tags", None) or tags or tags_ctx.get(),
                    extra=kwargs.get("extra", None),
                    template_id=kwargs.pop("templateId", None),
                )
            except Exception as e:
                handle_internal_error(e)

            if kwargs.get("stream") == True:
                return stream_handler(
                    fn, run_id, name or parsed_input["name"], type, *args, **kwargs
                )

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
                parsed_output = output_parser(
                    output, kwargs.get("stream", False))

                track_event(
                    type,
                    "end",
                    run_id,
                    name=name
                    or parsed_input[
                        "name"
                    ],  # Need name in case need to compute tokens usage server side
                    output=parsed_output["output"],
                    token_usage=parsed_output["tokensUsage"],
                )
                return output
            except Exception as e:
                handle_internal_error(e)
            finally:
                return output

    return sync_wrapper


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
    async def wrapper(*args, **kwargs):
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(uuid.uuid4()) as run:
                output = None
                try:
                    run_id = trace.get_current_span().context.span_id
                    parent_run_id = getattr(
                        trace.get_current_span().parent, "span_id", None
                    )
                    parsed_input = input_parser(*args, **kwargs)

                    track_event(
                        type,
                        "start",
                        run_id,
                        parent_run_id,
                        input=parsed_input["input"],
                        name=name or parsed_input["name"],
                        user_id=kwargs.pop(
                            "user_id", None
                        ) or user_ctx.get() or user_id,
                        user_props=kwargs.pop("user_props", None)
                                   or user_props
                                   or user_props_ctx.get(),
                        tags=kwargs.pop("tags", None) or tags or tags_ctx.get(),
                        extra=parsed_input.get("extra", None),
                        template_id=kwargs.pop("templateId", None),
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
                    parsed_output = output_parser(
                        output, kwargs.get("stream", False)
                    )

                    track_event(
                        type,
                        "end",
                        run_id,
                        name=name
                             or parsed_input[
                                 "name"
                             ],  # Need name in case need to compute tokens usage server side
                        output=parsed_output["output"],
                        token_usage=parsed_output["tokensUsage"],
                    )
                    return output
                except Exception as e:
                    handle_internal_error(e)
                finally:
                    return output

        def async_stream_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(uuid.uuid4()) as run:
                output = None
                try:
                    run_id = trace.get_current_span().context.span_id
                    parent_run_id = getattr(
                        trace.get_current_span().parent, "span_id", None
                    )
                    parsed_input = input_parser(*args, **kwargs)

                    track_event(
                        type,
                        "start",
                        run_id,
                        parent_run_id,
                        input=parsed_input["input"],
                        name=name or parsed_input["name"],
                        user_id=kwargs.pop(
                            "user_id", None
                        ) or user_ctx.get() or user_id,
                        user_props=kwargs.pop("user_props", None)
                                   or user_props
                                   or user_props_ctx.get(),
                        tags=kwargs.pop("tags", None) or tags or tags_ctx.get(),
                        extra=parsed_input.get("extra", None),
                        template_id=kwargs.pop("templateId", None),
                    )
                except Exception as e:
                    handle_internal_error(e)

                return async_stream_handler(fn, run_id, name or parsed_input["name"], type, *args, **kwargs)

        if kwargs.get("stream") == True:
            return async_stream_wrapper(*args, **kwargs)

        else:
            return await async_wrapper(*args, **kwargs)

    return wrapper


def monitor(object):
    try:
        openai_version = parse_version(version("openai"))
        name = getattr(object, "__name__", getattr(
            type(object), "__name__", None))

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
                    logging.info(
                        "[Lunary] Please use `lunary.monitor(openai)` or `lunary.monitor(client)` after setting the OpenAI api key"
                    )

            elif name == "AsyncOpenAI":
                object.chat.completions.create = async_wrap(
                    object.chat.completions.create,
                    "llm",
                    input_parser=OpenAIUtils.parse_input,
                    output_parser=OpenAIUtils.parse_output,
                )
            else:
                logging.info(
                    "[Lunary] Uknonwn OpenAI client. You can only use `lunary.monitor(openai)` or `lunary.monitor(client)`"
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
        logging.info("[Lunary] The `openai` package is not installed")


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


try:
    import importlib.metadata
    import logging
    import os
    import traceback
    import warnings
    from contextvars import ContextVar
    from typing import Any, Dict, List, Union, cast
    from uuid import UUID

    import requests
    from langchain_core.agents import AgentFinish
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult
    from packaging.version import parse

    logger = logging.getLogger(__name__)

    DEFAULT_API_URL = "https://app.lunary.ai"

    user_ctx = ContextVar[Union[str, None]]("user_ctx", default=None)
    user_props_ctx = ContextVar[Union[str, None]]("user_props_ctx", default=None)

    spans: Dict[str, Any] = {}

    PARAMS_TO_CAPTURE = [
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "presence_penalty",
        "frequence_penalty",
        "seed",
        "function_call",
        "functions",
        "tools",
        "tool_choice",
        "response_format",
        "max_tokens",
        "logit_bias",
    ]


    class UserContextManager:
        """Context manager for Lunary user context."""

        def __init__(self, user_id: str, user_props: Any = None) -> None:
            user_ctx.set(user_id)
            user_props_ctx.set(user_props)

        def __enter__(self) -> Any:
            pass

        def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> Any:
            user_ctx.set(None)
            user_props_ctx.set(None)


    def identify(user_id: str, user_props: Any = None) -> UserContextManager:
        """Builds a Lunary UserContextManager

        Parameters:
            - `user_id`: The user id.
            - `user_props`: The user properties.

        Returns:
            A context manager that sets the user context.
        """
        return UserContextManager(user_id, user_props)


    def _serialize(obj: Any) -> Union[Dict[str, Any], List[Any], Any]:
        if hasattr(obj, "to_json"):
            return obj.to_json()

        if isinstance(obj, dict):
            return {key: _serialize(value) for key, value in obj.items()}

        if isinstance(obj, list):
            return [_serialize(element) for element in obj]

        return obj


    def _parse_input(raw_input: Any) -> Any:
        if not raw_input:
            return None

        # if it's an array of 1, just parse the first element
        if isinstance(raw_input, list) and len(raw_input) == 1:
            return _parse_input(raw_input[0])

        if not isinstance(raw_input, dict):
            return _serialize(raw_input)

        input_value = raw_input.get("input")
        inputs_value = raw_input.get("inputs")
        question_value = raw_input.get("question")
        query_value = raw_input.get("query")

        if input_value:
            return input_value
        if inputs_value:
            return inputs_value
        if question_value:
            return question_value
        if query_value:
            return query_value

        return _serialize(raw_input)


    def _parse_output(raw_output: dict) -> Any:
        if not raw_output:
            return None

        if not isinstance(raw_output, dict):
            return _serialize(raw_output)

        text_value = raw_output.get("text")
        output_value = raw_output.get("output")
        output_text_value = raw_output.get("output_text")
        answer_value = raw_output.get("answer")
        result_value = raw_output.get("result")

        if text_value:
            return text_value
        if answer_value:
            return answer_value
        if output_value:
            return output_value
        if output_text_value:
            return output_text_value
        if result_value:
            return result_value

        return _serialize(raw_output)


    def _parse_lc_role(
        role: str,
    ) -> str:
        if role == "human":
            return "user"
        else:
            return role


    def _get_user_id(metadata: Any) -> Any:
        if user_ctx.get() is not None:
            return user_ctx.get()

        metadata = metadata or {}
        user_id = metadata.get("user_id")
        if user_id is None:
            user_id = metadata.get("userId")  # legacy, to delete in the future
        return user_id


    def _get_user_props(metadata: Any) -> Any:
        if user_props_ctx.get() is not None:
            return user_props_ctx.get()

        metadata = metadata or {}
        return metadata.get("user_props", None)


    def _parse_lc_message(message: BaseMessage) -> Dict[str, Any]:
        keys = ["function_call", "tool_calls", "tool_call_id", "name"]
        parsed = {"text": message.content, "role": _parse_lc_role(message.type)}
        parsed.update(
            {
                key: cast(Any, message.additional_kwargs.get(key))
                for key in keys
                if message.additional_kwargs.get(key) is not None
            }
        )
        return parsed


    def _parse_lc_messages(messages: Union[List[BaseMessage], Any]) -> List[Dict[str, Any]]:
        return [_parse_lc_message(message) for message in messages]


    class LunaryCallbackHandler(BaseCallbackHandler):
        """Callback Handler for Lunary`.

        #### Parameters:
            - `app_id`: The app id of the app you want to report to. Defaults to
            `None`, which means that `LUNARY_APP_ID` will be used.
            - `api_url`: The url of the Lunary API. Defaults to `None`,
            which means that either `LUNARY_API_URL` environment variable
            or `https://app.lunary.ai` will be used.

        #### Raises:
            - `ValueError`: if `app_id` is not provided either as an
            argument or as an environment variable.
            - `ConnectionError`: if the connection to the API fails.


        #### Example:
        ```python
        from langchain.llms import OpenAI
        from langchain.callbacks import LunaryCallbackHandler

        handler = LunaryCallbackHandler()
        llm = OpenAI(callbacks=[handler],
                    metadata={"userId": "user-123"})
        llm.predict("Hello, how are you?")
        ```
        """

        __app_id: str

        def __init__(
            self,
            app_id: Union[str, None] = None,
            api_url: Union[str, None] = None,
            verbose: bool = False,
        ) -> None:
            super().__init__()
            try:
                import lunary

                self.__lunary_version = importlib.metadata.version("lunary")
                self.__track_event = lunary.track_event
                self.__tracer = lunary.tracer
                self.__set_span_in_context = lunary.trace.set_span_in_context
                self.__trace = lunary.trace

            except ImportError:
                logger.warning(
                    """[Lunary] To use the Lunary callback handler you need to 
                    have the `lunary` Python package installed. Please install it 
                    with `pip install lunary`"""
                )
                self.__has_valid_config = False
                return

            if parse(self.__lunary_version) < parse("0.0.32"):
                logger.warning(
                    f"""[Lunary] The installed `lunary` version is
                    {self.__lunary_version}
                    but `LunaryCallbackHandler` requires at least version 0.1.1
                    upgrade `lunary` with `pip install --upgrade lunary`"""
                )
                self.__has_valid_config = False

            self.__has_valid_config = True

            self.__api_url = (
                api_url
                or os.getenv("LUNARY_API_URL")
                or os.getenv("LLMONITOR_API_URL")
                or DEFAULT_API_URL
            )
            self.__verbose = (
                verbose
                or bool(os.getenv("LUNARY_VERBOSE"))
                or bool(os.getenv("LLMONITOR_VERBOSE"))
            )

            _app_id = app_id or os.getenv("LUNARY_APP_ID") or os.getenv("LLMONITOR_APP_ID")
            if _app_id is None:
                logger.warning(
                    """[Lunary] app_id must be provided either as an argument or 
                    as an environment variable"""
                )
                self.__has_valid_config = False
            else:
                self.__app_id = _app_id

            if self.__has_valid_config is False:
                return None

            try:
                res = requests.get(f"{self.__api_url}/api/app/{self.__app_id}")
                if not res.ok:
                    raise ConnectionError()
            except Exception:
                logger.warning(
                    f"""[Lunary] Could not connect to the Lunary API at
                    {self.__api_url}"""
                )

        def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            tags: Union[List[str], None] = None,
            metadata: Union[Dict[str, Any], None] = None,
            **kwargs: Any,
        ) -> None:
            try:
                run_id_str = str(run_id)
                if parent_run_id and spans.get(str(parent_run_id)):
                    parent = spans[str(parent_run_id)]
                    context = self.__set_span_in_context(parent)
                    span = self.__tracer.start_span("llm", context=context)
                    spans[run_id_str] = span
                else:
                    context = self.__set_span_in_context(self.__trace.get_current_span())
                    span = self.__tracer.start_span("llm", context=context)
                    parent_run_id = getattr(getattr(span, "parent", None), "span_id", None)
                    spans[run_id_str] = span

                user_id = _get_user_id(metadata)
                user_props = _get_user_props(metadata)

                params = kwargs.get("invocation_params", {})
                params.update(
                    serialized.get("kwargs", {})
                )  # Sometimes, for example with ChatAnthropic, `invocation_params` is empty

                name = (
                    params.get("model")
                    or params.get("model_name")
                    or params.get("model_id")
                )

                if not name and "anthropic" in params.get("_type"):
                    name = "claude-2"

                extra = {
                    param: params.get(param)
                    for param in PARAMS_TO_CAPTURE
                    if params.get(param) is not None
                }

                input = _parse_input(prompts)

                self.__track_event(
                    "llm",
                    "start",
                    user_id=user_id,
                    run_id=run_id_str,
                    parent_run_id=parent_run_id,
                    name=name,
                    input=input,
                    tags=tags,
                    extra=extra,
                    metadata=metadata,
                    user_props=user_props,
                    app_id=self.__app_id,
                )
            except Exception as e:
                warnings.warn(
                    f"""[Lunary] An error occurred in on_llm_start:
                    {e}\n{traceback.format_exc()}"""
                )

        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            tags: Union[List[str], None] = None,
            metadata: Union[Dict[str, Any], None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                context = self.__set_span_in_context(self.__trace.get_current_span())
                run_id_str = str(run_id)

                # Sometimes parent_run_id is set by langchain, but the
                # corresponding callback handler method is not called
                if parent_run_id and spans.get(str(parent_run_id)) is None:
                    parent_run_id = None

                if parent_run_id:
                    parent = spans[str(parent_run_id)]
                    context = self.__set_span_in_context(parent)
                    span = self.__tracer.start_span("llm", context=context)
                    spans[run_id_str] = span
                else:
                    context = self.__set_span_in_context(self.__trace.get_current_span())
                    span = self.__tracer.start_span("llm", context=context)
                    parent_run_id = getattr(getattr(span, "parent", None), "span_id", None)
                    spans[run_id_str] = span

                user_id = _get_user_id(metadata)
                user_props = _get_user_props(metadata)

                params = kwargs.get("invocation_params", {})
                params.update(
                    serialized.get("kwargs", {})
                )  # Sometimes, for example with ChatAnthropic, `invocation_params` is empty

                name = (
                    params.get("model")
                    or params.get("model_name")
                    or params.get("model_id")
                )

                if not name and "anthropic" in params.get("_type"):
                    name = "claude-2"

                extra = {
                    param: params.get(param)
                    for param in PARAMS_TO_CAPTURE
                    if params.get(param) is not None
                }

                input = _parse_lc_messages(messages[0])

                self.__track_event(
                    "llm",
                    "start",
                    user_id=user_id,
                    run_id=run_id_str,
                    parent_run_id=str(parent_run_id),
                    name=name,
                    input=input,
                    tags=tags,
                    extra=extra,
                    metadata=metadata,
                    user_props=user_props,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_chat_model_start: {e}")

        def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> None:
            try:
                span = spans.get(str(run_id))
                if span and hasattr(span, "is_recording") and span.is_recording():
                    span.end()

                token_usage = (response.llm_output or {}).get("token_usage", {})
                parsed_output: Any = [
                    _parse_lc_message(generation.message)
                    if hasattr(generation, "message")
                    else generation.text
                    for generation in response.generations[0]
                ]

                # if it's an array of 1, just parse the first element
                if len(parsed_output) == 1:
                    parsed_output = parsed_output[0]

                self.__track_event(
                    "llm",
                    "end",
                    run_id=run_id,
                    output=parsed_output,
                    token_usage={
                        "prompt": token_usage.get("prompt_tokens"),
                        "completion": token_usage.get("completion_tokens"),
                    },
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_llm_end: {e}")

        def on_tool_start(
            self,
            serialized: Dict[str, Any],
            input_str: str,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            tags: Union[List[str], None] = None,
            metadata: Union[Dict[str, Any], None] = None,
            **kwargs: Any,
        ) -> None:
            try:
                run_id_str = str(run_id)
                if parent_run_id:
                    parent = spans[str(parent_run_id)]
                    context = self.__set_span_in_context(parent)
                    span = self.__tracer.start_span("tool", context=context)
                    spans[run_id_str] = span
                else:
                    context = self.__set_span_in_context(self.__trace.get_current_span())
                    span = self.__tracer.start_span("tool", context=context)
                    parent_run_id = getattr(getattr(span, "parent", None), "span_id", None)
                    spans[run_id_str] = span

                user_id = _get_user_id(metadata)
                user_props = _get_user_props(metadata)
                name = serialized.get("name")

                self.__track_event(
                    "tool",
                    "start",
                    user_id=user_id,
                    run_id=run_id_str,
                    parent_run_id=str(parent_run_id),
                    name=name,
                    input=input_str,
                    tags=tags,
                    metadata=metadata,
                    user_props=user_props,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_tool_start: {e}")

        def on_tool_end(
            self,
            output: str,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            tags: Union[List[str], None] = None,
            **kwargs: Any,
        ) -> None:
            try:
                span = spans.get(str(run_id))
                if span and hasattr(span, "is_recording") and span.is_recording():
                    span.end()
                self.__track_event(
                    "tool",
                    "end",
                    run_id=run_id,
                    output=output,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_tool_end: {e}")

        def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            tags: Union[List[str], None] = None,
            metadata: Union[Dict[str, Any], None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                run_id_str = str(run_id)
                if parent_run_id and spans.get(str(parent_run_id)):
                    parent = spans[str(parent_run_id)]
                    context = self.__set_span_in_context(parent)
                    span = self.__tracer.start_span("chain", context=context)
                    spans[run_id_str] = span
                else:
                    context = self.__set_span_in_context(self.__trace.get_current_span())
                    span = self.__tracer.start_span("tool", context=context)
                    parent_run_id = getattr(getattr(span, "parent", None), "span_id", None)
                    spans[run_id_str] = span

                name = serialized.get("id", [None, None, None, None])[3]
                type = "chain"
                metadata = metadata or {}

                agentName = metadata.get("agent_name")
                if agentName is None:
                    agentName = metadata.get("agentName")

                if name == "AgentExecutor" or name == "PlanAndExecute":
                    type = "agent"
                if agentName is not None:
                    type = "agent"
                    name = agentName
                if parent_run_id is not None:
                    type = "chain"

                user_id = _get_user_id(metadata)
                user_props = _get_user_props(metadata)
                input = _parse_input(inputs)

                self.__track_event(
                    type,
                    "start",
                    user_id=user_id,
                    run_id=run_id_str,
                    parent_run_id=parent_run_id,
                    name=name,
                    input=input,
                    tags=tags,
                    metadata=metadata,
                    user_props=user_props,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_chain_start: {e}")

        def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans.get(str(run_id))
                if span and hasattr(span, "is_recording") and span.is_recording():
                    span.end()

                output = _parse_output(outputs)

                self.__track_event(
                    "chain",
                    "end",
                    run_id=run_id,
                    output=output,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_chain_end: {e}")

        def on_agent_finish(
            self,
            finish: AgentFinish,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans.get(str(run_id))
                if span and hasattr(span, "is_recording") and span.is_recording():
                    span.end()
                output = _parse_output(finish.return_values)

                self.__track_event(
                    "agent",
                    "end",
                    run_id=run_id,
                    output=output,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_agent_finish: {e}")

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans.get(str(run_id))
                if span and hasattr(span, "is_recording") and span.is_recording():
                    span.end()
                self.__track_event(
                    "chain",
                    "error",
                    run_id=run_id,
                    error={"message": str(error), "stack": traceback.format_exc()},
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_chain_error: {e}")

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans.get(str(run_id))
                if span and hasattr(span, "is_recording") and span.is_recording():
                    span.end()
                self.__track_event(
                    "tool",
                    "error",
                    run_id=run_id,
                    error={"message": str(error), "stack": traceback.format_exc()},
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_tool_error: {e}")

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans.get(str(run_id))
                if span and hasattr(span, "is_recording") and span.is_recording():
                    span.end()
                self.__track_event(
                    "llm",
                    "error",
                    run_id=run_id,
                    error={"message": str(error), "stack": traceback.format_exc()},
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[Lunary] An error occurred in on_llm_error: {e}")

except Exception as e:
    pass


def open_thread(id: Optional[str] = None, tags: Optional[List[str]] = None):
    return Thread(track_event=track_event, id=id, tags=tags)


def track_feedback(run_id: str, feedback: Dict[str, Any]):
    if not run_id or not isinstance(run_id, str):
        print("Lunary: No message ID provided to track feedback")
        return

    if not isinstance(feedback, dict):
        print("Lunary: Invalid feedback provided. Pass a valid object")
        return

    track_event(None, "feedback", run_id=run_id, feedback=feedback)



templateCache = {}

def get_raw_template(slug="kind-angle"):
    APP_ID = (
        os.environ.get(
            "LUNARY_APP_ID") or os.environ.get("LLMONITOR_APP_ID")
    )
    global templateCache
    now = time.time() * 1000
    cache_entry = templateCache.get(slug)

    if cache_entry and now - cache_entry['timestamp'] < 60000:
        return cache_entry['data']

    response = requests.get(f"{DEFAULT_API_URL}/api/v1/template?slug={slug}&app_id={APP_ID}", headers={"Content-Type": "application/json"})

    if not response.ok:
        raise Exception(f"Lunary: Error fetching template: {response.status_code} - {response.text}")

    data = response.json()
    templateCache[slug] = {'timestamp': now, 'data': data}
    return data


def render_template(slug, data):
    try:
        raw_template = get_raw_template(slug)
        template_id = copy.deepcopy(raw_template['id'])
        content = copy.deepcopy(raw_template['content'])
        extra = copy.deepcopy(raw_template['extra'])


        text_mode = isinstance(content, str)

        result = None
        if text_mode:
            rendered = chevron.render(content, data)
            result = {"text": rendered, "templateId": template_id, **extra}
            return result
        else:
            messages = []
            for message in content:
                message["content"] = chevron.render(message["content"], data)
                messages.append(message)
            result = {"messages": messages, "templateId": template_id, **extra}

            return result

    except Exception as e:
        print(e)


def get_dataset(dataset_id):
    APP_ID = os.environ.get("LUNARY_APP_ID") or os.environ.get("LLMONITOR_APP_ID")

    try:
        url = f"{DEFAULT_API_URL}/v1/projects/{APP_ID}/datasets/{dataset_id}"
        response = requests.get(url, headers={"Content-Type": "application/json"})

        raise Exception("Error")
        data = response.json()
        return data['runs']

    except Exception as e:
        print("Lunary: Error fetching dataset: you must be on the Unlimited or Enterprise plan to use this feature.")
