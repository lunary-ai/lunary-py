from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import lunary
import json

model = ModelInference(
    model_id="meta-llama/llama-3-1-8b-instruct",
    credentials=Credentials(
        api_key = "pfc6XWOR0oX_oZY-c4axG-JKKzMhKz1NF04g9C8Idfz2",
        url = "https://us-south.ml.cloud.ibm.com"),
        project_id="c36b3e63-c64c-4f2c-8885-933244642424"
    )

lunary.monitor_ibm(model)

print(model.model_id)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"}
]
response = model.chat(messages=messages)

# print(json.dumps(response, indent=4))
# print(generated_response['choices'][0]['message']['content'])
