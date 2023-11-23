import os, warnings, traceback
from datetime import datetime, timezone
from .tags import tags_ctx, tags  # DO NOT REMOVE tags
from .users import user_ctx, user_props_ctx, identify  # DO NOT REMOVE identify
from .consumer import Consumer
from .event_queue import EventQueue
from .parsers import (
    default_input_parser,
    default_output_parser,
)

from opentelemetry import trace
from opentelemetry.trace import set_span_in_context
from opentelemetry.sdk.trace import TracerProvider, Span


provider = TracerProvider()
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llmonitor")

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
    # Load here in case load_dotenv done after `monitor`
    APP_ID = app_id or os.environ.get("LLMONITOR_APP_ID")
    VERBOSE = os.environ.get("LLMONITOR_VERBOSE")

    if not APP_ID:
        return warnings.warn("LLMONITOR_APP_ID is not set, not sending events")

    span = trace.get_current_span()
    span_id = str(run_id) or span.context.span_id


    parent_span_id = (
        str(parent_run_id) if parent_run_id else getattr(span, "parent", None)
    )

    event = {
        "event": event_name,
        "type": run_type,
        "app": APP_ID,
        "name": name,
        "userId": user_id,
        "userProps": user_props,
        "tags": tags or tags_ctx.get(),
        "runId": span_id,
        "parentRunId": parent_span_id,
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
        print("\n")

    queue.append(event)


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
    @tracer.start_as_current_span(name)
    def sync_wrapper(*args, **kwargs):
        try:
            span = trace.get_current_span()
            span_id = span.get_span_context().span_id
            parent_span_id = getattr(span, "parent", None)

            parsed_input = input_parser(*args, **kwargs)
            track_event(
                type,
                "start",
                run_id=span_id,
                parent_run_id=parent_span_id,
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
            print("[LLMonitor] Error: ", e)

        try:
            output = fn(*args, **kwargs)
        except Exception as e:
            track_event(
                type,
                "error",
                run_id=span_id,
                error={"message": str(e), "stack": traceback.format_exc()},
            )
            raise e  # rethrow error

        try:
            parsed_output = output_parser(output, kwargs.get("stream", False))

            track_event(
                type,
                "end",
                run_id=span_id,
                # Need name in case need to compute tokens usage server side,
                name=name or parsed_input["name"],
                output=parsed_output["output"],
                token_usage=parsed_output["tokensUsage"],
            )
            return output
        except Exception as e:
            print("[LLMonitor] Error: ", e)
        finally:
            return output

    return sync_wrapper


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


try:
    import logging
    import os
    import traceback
    import warnings
    from contextvars import ContextVar
    from typing import Any, Dict, List, Union, cast
    from uuid import UUID

    import requests
    from packaging.version import parse

    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema.agent import AgentAction, AgentFinish
    from langchain.schema.messages import BaseMessage
    from langchain.schema.output import LLMResult

    logger = logging.getLogger(__name__)

    DEFAULT_API_URL = "https://app.llmonitor.com"

    user_ctx = ContextVar[Union[str, None]]("user_ctx", default=None)
    user_props_ctx = ContextVar[Union[str, None]]("user_props_ctx", default=None)

    spans: Dict[str, Span] = {}

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
        """Context manager for LLMonitor user context."""

        def __init__(self, user_id: str, user_props: Any = None) -> None:
            user_ctx.set(user_id)
            user_props_ctx.set(user_props)

        def __enter__(self) -> Any:
            pass

        def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> Any:
            user_ctx.set(None)
            user_props_ctx.set(None)

    def identify(user_id: str, user_props: Any = None) -> UserContextManager:
        """Builds an LLMonitor UserContextManager

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

    def _parse_lc_messages(
        messages: Union[List[BaseMessage], Any]
    ) -> List[Dict[str, Any]]:
        return [_parse_lc_message(message) for message in messages]

    class LLMonitorCallbackHandler(BaseCallbackHandler):
        """Callback Handler for LLMonitor`.

        #### Parameters:
            - `app_id`: The app id of the app you want to report to. Defaults to
            `None`, which means that `LLMONITOR_APP_ID` will be used.
            - `api_url`: The url of the LLMonitor API. Defaults to `None`,
            which means that either `LLMONITOR_API_URL` environment variable
            or `https://app.llmonitor.com` will be used.

        #### Raises:
            - `ValueError`: if `app_id` is not provided either as an
            argument or as an environment variable.
            - `ConnectionError`: if the connection to the API fails.


        #### Example:
        ```python
        from langchain.llms import OpenAI
        from langchain.callbacks import LLMonitorCallbackHandler

        llmonitor_callback = LLMonitorCallbackHandler()
        llm = OpenAI(callbacks=[llmonitor_callback],
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
            self.__app_id = app_id or os.environ.get("LLMONITOR_APP_ID")
            super().__init__()

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
                run_id = str(run_id)
                if parent_run_id and spans.get(str(parent_run_id)):
                    parent = spans[str(parent_run_id)]
                    context = set_span_in_context(parent)
                    span = tracer.start_span("llm", context=context)
                    spans[run_id] = span
                else:
                    context = set_span_in_context(trace.get_current_span())
                    span = tracer.start_span("llm", context=context)
                    parent_run_id = span.parent.span_id
                    spans[run_id] = span

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

                track_event(
                    "llm",
                    "start",
                    user_id=user_id,
                    run_id=run_id,
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
                    f"[LLMonitor] An error occurred in on_llm_start: {e}\n{traceback.format_exc()}"
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
                context = set_span_in_context(trace.get_current_span())
                run_id = str(run_id)

                # Sometimes parent_run_id is set by langchain, but the corresponding callback handler method is not called
                if parent_run_id and spans.get(str(parent_run_id)) is None: 
                    parent_run_id = None


                if parent_run_id:
                    parent = spans[str(parent_run_id)]
                    context = set_span_in_context(parent)
                    span = tracer.start_span("llm", context=context)
                    spans[run_id] = span
                else:
                    context = set_span_in_context(trace.get_current_span())
                    span = tracer.start_span("llm", context=context)
                    parent_run_id = span.parent.span_id
                    spans[run_id] = span

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

                track_event(
                    "llm",
                    "start",
                    user_id=user_id,
                    run_id=run_id,
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
                logger.error(
                    f"[LLMonitor] An error occurred in on_chat_model_start: {e}"
                )

        def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> None:
            try:
                span = spans[str(run_id)]
                if not span.is_recording():
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

                track_event(
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
                logger.error(f"[LLMonitor] An error occurred in on_llm_end: {e}")

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
                run_id = str(run_id)
                if parent_run_id:
                    parent = spans[str(parent_run_id)]
                    context = set_span_in_context(parent)
                    span = tracer.start_span("tool", context=context)
                    spans[run_id] = span
                else:
                    context = set_span_in_context(trace.get_current_span())
                    span = tracer.start_span("tool", context=context)
                    parent_run_id = span.parent.span_id
                    spans[run_id] = span

                user_id = _get_user_id(metadata)
                user_props = _get_user_props(metadata)
                name = serialized.get("name")

                track_event(
                    "tool",
                    "start",
                    user_id=user_id,
                    run_id=run_id,
                    parent_run_id=str(parent_run_id),
                    name=name,
                    input=input_str,
                    tags=tags,
                    metadata=metadata,
                    user_props=user_props,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_tool_start: {e}")

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
                span = spans[str(run_id)]
                if not span.is_recording():
                    span.end()
                track_event(
                    "tool",
                    "end",
                    run_id=run_id,
                    output=output,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_tool_end: {e}")

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
                run_id = str(run_id)
                if parent_run_id and spans.get(str(parent_run_id)):
                    parent = spans[str(parent_run_id)]
                    context = set_span_in_context(parent)
                    span = tracer.start_span("chain", context=context)
                    spans[run_id] = span
                else:
                    context = set_span_in_context(trace.get_current_span())
                    span = tracer.start_span("tool", context=context)
                    parent_run_id = span.parent.span_id
                    spans[run_id] = span

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

                track_event(
                    type,
                    "start",
                    user_id=user_id,
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    name=name,
                    input=input,
                    tags=tags,
                    metadata=metadata,
                    user_props=user_props,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_chain_start: {e}")

        def on_chain_end(
            self,
            outputs: Dict[str, Any],
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans[str(run_id)]
                if not span.is_recording():
                    span.end()

                output = _parse_output(outputs)

                track_event(
                    "chain",
                    "end",
                    run_id=run_id,
                    output=output,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_chain_end: {e}")

        def on_agent_finish(
            self,
            finish: AgentFinish,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans[str(run_id)]
                if not span.is_recording():
                    span.end()
                output = _parse_output(finish.return_values)

                track_event(
                    "agent",
                    "end",
                    run_id=run_id,
                    output=output,
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_agent_finish: {e}")

        def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans[str(run_id)]
                if not span.is_recording():
                    span.end()
                track_event(
                    "chain",
                    "error",
                    run_id=run_id,
                    error={"message": str(error), "stack": traceback.format_exc()},
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_chain_error: {e}")

        def on_tool_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans[str(run_id)]
                if not span.is_recording():
                    span.end()
                track_event(
                    "tool",
                    "error",
                    run_id=run_id,
                    error={"message": str(error), "stack": traceback.format_exc()},
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_tool_error: {e}")

        def on_llm_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: Union[UUID, None] = None,
            **kwargs: Any,
        ) -> Any:
            try:
                span = spans[str(run_id)]
                if not span.is_recording():
                    span.end()
                track_event(
                    "llm",
                    "error",
                    run_id=run_id,
                    error={"message": str(error), "stack": traceback.format_exc()},
                    app_id=self.__app_id,
                )
            except Exception as e:
                logger.error(f"[LLMonitor] An error occurred in on_llm_error: {e}")

except Exception as e:
    ("[LLMonitor] Please install `langchain` to use LLMonitorCallbackHandler.")
