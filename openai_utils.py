def input_parser(input):
    messages = [
        {"role": message["role"], "text": message["content"]}
        for message in input["messages"]
    ]

    return {"name": input["model"], "input": messages}


def output_parser(output):
    return {
        "output": {
            "role": output.choices[0].message.role,
            "text": output.choices[0].message.content,
        },
        "tokensUsage": {
            "completion": output.usage.completion_tokens,
            "prompt": output.usage.prompt_tokens,
        },
    }
