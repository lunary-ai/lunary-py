import json, logging

logger = logging.getLogger(__name__)

MONITORED_KEYS = [
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "max_tokens",
    "n",
    "present_penalty",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "tool_calls",
    "top_p",
]


class OpenAIUtils:
    @staticmethod
    def parse_role(role):
        if role == "assistant":
            return "ai"
        else:
            return role

    @staticmethod
    def get_property(object, property):
        if isinstance(object, dict):
            return None if not object.get(property) else object.get(property)
        else:
            return getattr(object, property, None)

    @staticmethod
    def parse_message(message):
        tool_calls = OpenAIUtils.get_property(message, "tool_calls")

        if tool_calls is not None:
            tool_calls_serialized = [
                json.loads(tool_call.model_dump_json(indent=2, exclude_unset=True))
                for tool_call in tool_calls
            ]
            tool_calls = tool_calls_serialized

        parsed_message = {
            "role": OpenAIUtils.get_property(message, "role"),
            "content": OpenAIUtils.get_property(message, "content"),
            "function_call": OpenAIUtils.get_property(message, "function_call"),
            "tool_calls": tool_calls,
        }
        return parsed_message

    @staticmethod
    def parse_input(*args, **kwargs):
        messages = [
            OpenAIUtils.parse_message(message) for message in kwargs["messages"]
        ]
        name = (
            kwargs.get("model", None)
            or kwargs.get("engine", None)
            or kwargs.get("deployment_id", None)
        )
        extra = {key: kwargs[key] for key in MONITORED_KEYS if key in kwargs}
        return {"name": name, "input": messages, "extra": extra}

    @staticmethod
    def parse_output(output, stream=False):
        try:
            return {
                "output": OpenAIUtils.parse_message(output.choices[0].message),
                "tokensUsage": {
                    "completion": output.usage.completion_tokens,
                    "prompt": output.usage.prompt_tokens,
                },
            }
        except Exception as e:
            logging.info("[Lunary] Error parsing output: ", e)
