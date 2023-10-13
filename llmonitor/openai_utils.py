MONITORED_KEYS = ["temperature", "functions", "max_tokens", "frequency_penalty", "stop", "presence_penalty", "function_call"]

class OpenAIUtils:
    @staticmethod
    def parse_role(role):
        if role == "assistant":
            return "ai"
        else:
            return role

    @staticmethod
    def parse_message(message):
        parsed_message = {
            "role": OpenAIUtils.parse_role(message["role"]), 
            "text": message["content"]
        }
        if "function_call" in message:
            parsed_message["functionCall"] = message["function_call"]
        return parsed_message

    @staticmethod
    def parse_input(*args, **kwargs):
        messages = [OpenAIUtils.parse_message(message) for message in kwargs["messages"]]
        name = kwargs.get('model', None) or kwargs.get('engine', None) or kwargs.get('deployment_id', None)
        extra = {key: kwargs[key] for key in MONITORED_KEYS if key in kwargs}
        return {
            "name": name,
            "input": messages,
            "extra": extra
        }

    @staticmethod
    def parse_output(output):
        try:
            message = output.choices[0].message

            return {
                "output": OpenAIUtils.parse_message(message),
                "tokensUsage": {
                    "completion": output.usage.completion_tokens,
                    "prompt": output.usage.prompt_tokens,
                },
            }
        except Exception as e:
            print("[LLMonitor] Error parsing output: ", e)
