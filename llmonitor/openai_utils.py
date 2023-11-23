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
        parsed_message = {
            "role": OpenAIUtils.get_property(message, "role"),
            "text": OpenAIUtils.get_property(message, "content"),
            "function_call": OpenAIUtils.get_property(message, "function_call"),
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
            print("[LLMonitor] Error parsing output: ", e)
