def parse_role(role):
    if role == "assistant":
        return "ai"
    else:
        return role


def parse_message(message):
    return {"role": parse_role(message["role"]), "text": message["content"]}


def parse_input(*args, **kwargs):
    messages = [parse_message(message) for message in kwargs["messages"]]
    name = kwargs["model"]
    return {
        "name": name,
        "input": messages,
    }


def parse_output(output):
    return {
        "output": {
            "role": parse_role(output.choices[0].message.role),
            "text": output.choices[0].message.content,
        },
        "tokensUsage": {
            "completion": output.usage.completion_tokens,
            "prompt": output.usage.prompt_tokens,
        },
    }
