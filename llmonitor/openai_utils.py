MONITORED_KEYS = [
    "temperature",
    "functions",
    "max_tokens",
    "frequency_penalty",
    "stop",
    "presence_penalty",
    "function_call",
]


class OpenAIUtils:
    @staticmethod
    def parse_role(role):
        if role == "assistant":
            return "ai"
        else:
            return role

    @staticmethod
    def get_property(message, property):
        if isinstance(message, dict):
            return message.get(property)
        else:
            return getattr(message, property, None)

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
        print(messages)
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
            # if stream:
            #     message = []
            #     role = None
            #     first_chunk_parsed = False
            #     parsed_output = None
            #     for chunk in output:
            #         chunk_message = chunk["choices"][0]["delta"].get("content", "")
            #         message.append(chunk_message)
            #         if first_chunk_parsed == False:
            #             role = chunk["choices"][0]["delta"]["role"]
            #             parsed_output = chunk
            #             first_chunk_parsed = True

            #     parsed_output["choices"][0]["message"] = {
            #         "role": OpenAIUtils.parse_role(role),
            #         "text": "".join(message),
            #     }
            #     return {"output": parsed_output.choices[0].message, "tokensUsage": None}

            return {
                "output": OpenAIUtils.parse_message(output.choices[0].message),
                "tokensUsage": {
                    "completion": output.usage.completion_tokens,
                    "prompt": output.usage.prompt_tokens,
                },
            }
        except Exception as e:
            print("[LLMonitor] Error parsing output: ", e)
