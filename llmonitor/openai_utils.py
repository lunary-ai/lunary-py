class OpenAIUtils:
    @staticmethod
    def parse_role(role):
        if role == "assistant":
            return "ai"
        else:
            return role

    @staticmethod
    def parse_message(message):
        return {
            "role": OpenAIUtils.parse_role(message["role"]), 
            "text": message["content"]
        }

    @staticmethod
    def parse_input(*args, **kwargs):
        messages = [OpenAIUtils.parse_message(message) for message in kwargs["messages"]]
        name = kwargs.get('model', None) or kwargs.get('engine', None)
        return {
            "name": name,
            "input": messages,
        }

    @staticmethod
    def parse_output(output):
        try:
            message = output.choices[0].message

            text = None
            if hasattr(message, 'content') and message.content is not None:
                text = message.content
            elif hasattr(message, 'function_call') and message.function_call is not None:
                text = str(message.function_call)

            return {
                "output": {
                    "role": OpenAIUtils.parse_role(output.choices[0].message.role),
                    "text": text 
                },
                "tokensUsage": {
                    "completion": output.usage.completion_tokens,
                    "prompt": output.usage.prompt_tokens,
                },
            }
        except Exception as e:
            print("[LLMonitor] Error parsing output: ", e)
