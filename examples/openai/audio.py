import lunary
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

lunary.set_config(app_id="f94d32eb-42e2-430a-b277-3511c170267f", verbose=True)

lunary.monitor(client)

completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["audio", "text"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[{"role": "user", "content": "Hello world"}],
)

print(completion.choices[0])
