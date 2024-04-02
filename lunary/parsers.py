from typing import Any, Dict

def default_input_parser(*args, **kwargs):
    def serialize(args, kwargs):
        if not args and not kwargs:
            return None

        if len(args) == 1 and not kwargs:
            return args[0]

        input = list(args)
        if kwargs:
            input.append(kwargs)

        return input

    return {"input": serialize(args, kwargs)}


def default_output_parser(output, *args, **kwargs):
    return {"output": getattr(output, "content", output), "tokensUsage": None}


PARAMS_TO_CAPTURE = [
  "frequency_penalty",
  "function_call",
  "functions",
  "logit_bias",
  "logprobs",
  "max_tokens",
  "n",
  "presence_penalty",
  "response_format",
  "seed",
  "stop",
  "temperature",
  "tool_choice",
  "tools",
  "top_logprobs",
  "top_p",        
  # Additional params
  "extra_headers",
  "extra_query",
  "extra_body",
  "timeout"
]

def filter_params(params: Dict[str, Any]) -> Dict[str, Any]:
    filtered_params = {key: value for key, value in params.items() if key in PARAMS_TO_CAPTURE}
    return filtered_params

