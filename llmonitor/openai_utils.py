class OpenAIUtils:
    @staticmethod
    def parse_role(role):
        if role == "assistant":
            return "ai"
        else:
            return role

    @staticmethod
    def parse_message(message):
        return {"role": OpenAIUtils.parse_role(message["role"]), "text": message["content"]}

    @staticmethod
    def parse_input(*args, **kwargs):
        messages = [OpenAIUtils.parse_message(message) for message in kwargs["messages"]]
        name = kwargs["model"]
        return {
            "name": name,
            "input": messages,
        }

    @staticmethod
    def parse_output(output):
        return {
            "output": {
                "role": OpenAIUtils.parse_role(output.choices[0].message.role),
                "text": output.choices[0].message.content,
            },
            "tokensUsage": {
                "completion": output.usage.completion_tokens,
                "prompt": output.usage.prompt_tokens,
            },
        }
