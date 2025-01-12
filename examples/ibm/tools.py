from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import lunary

model = ModelInference(
    model_id="meta-llama/llama-3-405b-instruct",
    credentials=Credentials(
        api_key = "pfc6XWOR0oX_oZY-c4axG-JKKzMhKz1NF04g9C8Idfz2",
        url = "https://us-south.ml.cloud.ibm.com"),
        project_id="c36b3e63-c64c-4f2c-8885-933244642424"
    )
lunary.monitor_ibm(model)

tools = [
  {
      "type": "function",
      "function": {
          "name": "get_weather",
          "parameters": {
              "type": "object",
              "properties": {
                  "location": {"type": "string"}
              },
          },
      },
  }
]
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like in Paris today?"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\": \"Paris\"}"
                }
            }
        ]
    },
    {
        "role": "tool",
        "tool_call_id": "call_abc123",
        "name": "get_weather",
        "content": "12 degrees celsius"
    },
    {
      "role": "user",
      "content": "Is it cold then?"
    }
]
response = model.chat(
  messages=messages, 
  tools=tools,
  tags=["baseball"], 
  user_id="1234", 
  user_props={"name": "Alice"}
)
print(response)