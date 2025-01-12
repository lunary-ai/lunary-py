from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import lunary

model = ModelInference(
    model_id="meta-llama/llama-3-1-8b-instruct",
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
    {"role": "user", "content": "What's the weather like in Paris today?"}
]
response = model.chat_stream(
  messages=messages, 
  tools=tools,
  # tags=["baseball"], 
  # user_id="1234", 
  # user_props={"name": "Alice"}
)

for chunk in response:
  print(chunk['choices'][0], '\n')